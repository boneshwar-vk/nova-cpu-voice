"""Gemma 4 E2B on the CPU as a small local, OpenAI-compatible endpoint (optional, standalone).

LiteRT-LM v0.17.1 raw session API (litert_capi.py), CPU backend only, one model process, kept loaded:
  * one live session: the system prompt is prefilled once at start (+ a 1-token warm-up decode);
    every request appends only what the KV cache lacks (normally just the new user turn);
  * a client-edited last reply (e.g. a barge-in keeps only the heard part) = rewind to that reply's
    pre-reply checkpoint + the heard text (no rebuild); anything else that diverges rewinds further;
  * POST /prefill: speculative prefill of a partial user turn; a chat request whose last user message
    equals it decodes at once (hit), else the speculation is rewound (miss);
  * thread count from --threads, multi-token prediction off, sampler from --top-k.

The voice server (app.py) uses the LiveGemma class below in-process; this HTTP wrapper is only for
trying the language model on its own.

Endpoints (127.0.0.1 only):
  POST /v1/chat/completions   OpenAI chat format; `stream: true` -> SSE chunks; `max_tokens` caps the reply
  POST /prefill               {"messages": [...history incl. system...], "partial": "user text so far"}
  GET  /stats                 last turn: ttft_ms, prefill_tokens, prefill_ms, decode_tps, mode, spec, threads
  GET  /                      a small chat page (TTFT + tok/s per reply)

Run:   python llm_cpu_server.py --port 8810 --threads 4
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE_ROOT = HERE / "models" / ".cache"

DEFAULT_SYSTEM = (
    "You are Nova, a warm voice assistant talking aloud. Start with a very short sentence, two to five words. "
    "Then one or two short sentences; if the question needs depth, explain clearly in up to six short sentences, "
    "then ask \"Want more detail?\" No lists.")


def S(t: str) -> str:
    return f"<|turn>user\n{t}<turn|>\n<|turn>model\n"


def clean(t: str) -> str:
    return (t or "").replace("<|turn>", "").replace("<turn|>", "").strip()


def norm(t: str) -> str:
    return " ".join((t or "").split())


def content_text(c) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict))
    return ""


class LiveGemma:
    """The KV mirror: turns = [(user, assistant|None)], tail in {closed, natural, open}."""

    def __init__(self, a):
        import litert_capi as R
        self.R = R
        self.a = a
        backend = getattr(a, "backend", "cpu")
        self.eng = R.Engine(model=a.model, backend=backend, max_tokens=a.max_tokens,
                            threads=a.threads if backend == "cpu" else None, mtp=False,
                            cache_dir=a.cache_dir)
        self.load_s = self.eng.load_s
        self.sess = None
        self.lock = threading.Lock()
        self.stats: dict = {}
        self.k = 0
        self.spec = None  # (history_key, partial_norm, label)
        self._new_session(a.system)
        # warm-up: one throwaway token on a checkpoint we rewind
        self.sess.checkpoint("warm")
        self.sess.prefill(S("Hi"))
        self.sess.decode(cancel_after=lambda st: len(st.chunks) >= 1)
        self.sess.rewind("sys")

    SPLIT = "\n\nAbout this person:\n"  # system = shared prefix (tools) + SPLIT + per-person part

    def _new_session(self, system: str):
        """Start the conversation over with `system`. When the shared prefix before SPLIT (Nova's tool
        declarations) is the one already in this session, rewind to it and prefill only the per-person
        part, so switching profile costs ~300 tokens instead of ~1,300."""
        t = time.perf_counter()
        pre, sep, rest = system.partition(self.SPLIT)
        if sep and self.sess and getattr(self, "_prefix", None) == pre:
            self.sess.rewind("prefix")
            self.sess.prefill(f"{sep}{rest}<turn|>\n")
            self.prefix_hit = True
        else:
            if self.sess:
                self.sess.close()
            self.sess = self.R.Session(self.eng, top_k=self.a.top_k, temperature=self.a.temperature, top_p=0.95, seed=1)
            if sep:
                self.sess.prefill(f"<|turn>system\n{pre}")
                self.sess.checkpoint("prefix")
                self._prefix = pre
                self.sess.prefill(f"{sep}{rest}<turn|>\n")
            else:
                self._prefix = None
                self.sess.prefill(f"<|turn>system\n{system}<turn|>\n")
            self.prefix_hit = False
        self.sys_ms = (time.perf_counter() - t) * 1000
        self.sess.checkpoint("sys")
        self.system = system
        self.turns: list[list] = []
        self.tail = "closed"
        self.cps = [("sys", [], "closed")]
        self.k = 0
        self.spec = None

    def _cp(self, label):
        self.sess.checkpoint(label)
        self.cps = [c for c in self.cps if c[0] != label] + [(label, [list(t) for t in self.turns], self.tail)]

    def _rewind(self, label):
        self.sess.rewind(label)
        i = [c[0] for c in self.cps].index(label)
        _, turns, tail = self.cps[i]
        self.turns = [list(t) for t in turns]
        self.tail = tail
        self.cps = self.cps[: i + 1]

    def _plan(self, history: list[tuple[str, str]], user: str) -> tuple[str, str]:
        """Rewind as needed; return (text to prefill, mode)."""
        m = 0
        while m < len(self.turns) and m < len(history):
            ku, ka = self.turns[m]
            hu, ha = history[m]
            if norm(ku) != norm(hu):
                break
            last = m == len(self.turns) - 1
            if ka is None:
                if not (last and self.tail == "open"):
                    break
            elif norm(ka) != norm(ha):
                break
            m += 1
        mode = "append"
        if m < len(self.turns):
            def ok(c) -> bool:
                if len(c[1]) <= m:
                    return all(norm(c[1][i][0]) == norm(self.turns[i][0]) for i in range(len(c[1])))
                # the turn's own pre-reply checkpoint r{k}: same user chunk, reply not yet started. Rewinding
                # there and appending the heard text avoids re-prefilling that whole user chunk after a barge-in.
                return (len(c[1]) == m + 1 and m < len(history) and c[2] == "open" and c[1][m][1] is None
                        and norm(c[1][m][0]) == norm(history[m][0])
                        and all(norm(c[1][i][0]) == norm(self.turns[i][0]) for i in range(m + 1)))
            cands = [c for c in self.cps if ok(c)]
            label = cands[-1][0] if cands else "sys"
            self._rewind(label)
            mode = f"rewind:{label}"
        # What the cache really holds now; the u{k} checkpoint must record this, not the replay below.
        self._snap = ([list(t) for t in self.turns], self.tail)
        b = []
        start = len(self.turns)
        if self.tail == "open" and start > 0 and start <= len(history):
            b.append(clean(history[start - 1][1]) + "<turn|>\n")
            self.turns[start - 1][1] = history[start - 1][1]
            mode += "+heard"
        for u, a in history[start:]:
            b.append(S(clean(u)) + clean(a) + "<turn|>\n")
            self.turns.append([u, a])
        if len(history) > start:
            mode += f"+replay{len(history) - start}"
        b.append(S(clean(user)))
        return "".join(b), mode

    def _prefill_turn(self, history, user):
        text, mode = self._plan(history, user)
        self.k += 1
        label = f"u{self.k}"
        self.sess.checkpoint(label)
        self.cps = [c for c in self.cps if c[0] != label] + [(label, self._snap[0], self._snap[1])]
        t = time.perf_counter()
        self.sess.prefill(text)
        ms = (time.perf_counter() - t) * 1000
        self.turns.append([user, None])
        self.tail = "open"
        self._cp(f"r{self.k}")
        return ms, self.eng.ntok(text), mode

    @staticmethod
    def split(messages):
        system = None
        hist, user = [], None
        pend = None
        for m in messages:
            role, txt = m.get("role"), content_text(m.get("content"))
            if role == "system":
                system = txt
            elif role == "user":
                if pend is not None:
                    hist.append((pend, ""))
                pend = txt
            elif role in ("assistant", "model"):
                if pend is not None:
                    hist.append((pend, txt))
                    pend = None
        user = pend
        return system, hist, user

    def prefill_partial(self, messages, partial, still_wanted=lambda: True):
        with self.lock:
            if not still_wanted():  # checked under the lock: a turn already generated after this was queued
                return {"skipped": "stale"}
            system, hist, _ = self.split(messages)
            if (system or self.a.system) != self.system:
                self._new_session(system or self.a.system)
            if self.spec:
                self._rewind(self.spec[2])
                self.spec = None
            label = f"u{self.k + 1}"
            ms, ntok, mode = self._prefill_turn(hist, partial)
            self.spec = (json.dumps(hist), norm(partial), label)
            return {"prefill_ms": round(ms, 1), "prefill_tokens": ntok, "mode": mode}

    def generate(self, messages, max_tokens, on_text, is_gone=lambda: False):
        with self.lock:
            t_req = time.perf_counter()
            system, hist, user = self.split(messages)
            if user is None:
                raise ValueError("no user message")
            if (system or self.a.system) != self.system:
                self._new_session(system or self.a.system)
            spec = None
            prefill_ms, ntok, mode = 0.0, 0, "spec-hit"
            if self.spec:
                hk, pn, label = self.spec
                self.spec = None
                if hk == json.dumps(hist) and pn == norm(user):
                    spec = True
                else:
                    spec = False
                    self._rewind(label)
            if spec is not True:
                prefill_ms, ntok, mode = self._prefill_turn(hist, user)
            cancelled = {"cap": False}

            def stop(st):
                if len(st.chunks) >= max_tokens:
                    cancelled["cap"] = True
                    return True
                return is_gone()
            st = self.sess.decode(cancel_after=stop, wait=False)
            sent = 0
            while True:
                fin = st.done.wait(0.01)
                while sent < len(st.chunks):
                    on_text(st.chunks[sent][1])
                    sent += 1
                if fin:
                    break
            reply = st.text()
            cut = bool(st.err) and "CANCEL" in (st.err or "").upper()
            if cut:
                # keep the KV consistent: back to the pre-reply state; the next request's history decides
                self._rewind(f"r{self.k}")
            else:
                self.turns[-1][1] = reply
                self.tail = "natural"
            n = len(st.chunks)
            ttft = (st.chunks[0][0] - t_req) * 1000 if n else None
            tps = (n - 1) / (st.chunks[-1][0] - st.chunks[0][0]) if n > 2 else None
            self.stats = {"ttft_ms": round(ttft, 1) if ttft else None, "prefill_ms": round(prefill_ms, 1),
                          "prefill_tokens": ntok, "decode_tokens": n, "decode_tps": round(tps, 2) if tps else None,
                          "mode": mode, "spec": spec, "cut": cut, "threads": self.a.threads, "top_k": self.a.top_k,
                          "kv_turns": len(self.turns), "error": None if cut else st.err}
            return reply, self.stats


PAGE = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Gemma CPU chat</title><style>
:root{--bg:#fafafa;--fg:#111;--mut:#666;--b:#ddd;--me:#e8f0fe}
@media (prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--mut:#999;--b:#333;--me:#1d2a3d}}
body{font:15px system-ui;background:var(--bg);color:var(--fg);max-width:760px;margin:0 auto;padding:16px}
#log div{padding:8px 10px;border:1px solid var(--b);border-radius:8px;margin:8px 0;white-space:pre-wrap}
.u{background:var(--me)} .m{font-size:12px;color:var(--mut)} form{display:flex;gap:8px}
input{flex:1;padding:8px;font:inherit;background:var(--bg);color:var(--fg);border:1px solid var(--b);border-radius:6px}
button{padding:8px 14px}</style></head><body>
<h3>Gemma 4 E2B · CPU · LiteRT-LM 0.17.1</h3><div class=m id=info></div><div id=log></div>
<form id=f><input id=q autocomplete=off placeholder="Ask something"><button>Send</button></form>
<script>
const msgs=[];const log=document.getElementById('log');
fetch('/stats').then(r=>r.json()).then(s=>info.textContent=`threads ${s.threads} · top-k ${s.top_k} · load ${s.load_s}s`);
let typing=null;
q.addEventListener('input',()=>{clearTimeout(typing);typing=setTimeout(()=>{if(q.value.trim().length>8)
 fetch('/prefill',{method:'POST',body:JSON.stringify({messages:msgs,partial:q.value.trim()})});},600)});
f.onsubmit=async e=>{e.preventDefault();const t=q.value.trim();if(!t)return;q.value='';
 msgs.push({role:'user',content:t});const u=document.createElement('div');u.className='u';u.textContent=t;log.append(u);
 const a=document.createElement('div');log.append(a);const m=document.createElement('div');m.className='m';log.append(m);
 const t0=performance.now();let first=null,n=0,txt='';
 const r=await fetch('/v1/chat/completions',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({messages:msgs,stream:true,max_tokens:200})});
 const rd=r.body.getReader();const dec=new TextDecoder();let buf='';
 while(true){const {value,done}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});
  let i;while((i=buf.indexOf('\\n\\n'))>=0){const line=buf.slice(0,i);buf=buf.slice(i+2);
   if(!line.startsWith('data: '))continue;const d=line.slice(6);if(d==='[DONE]')continue;
   const j=JSON.parse(d);const c=j.choices[0].delta.content||'';if(c){if(first===null)first=performance.now()-t0;n++;txt+=c;a.textContent=txt;}}}
 msgs.push({role:'assistant',content:txt});
 const s=await (await fetch('/stats')).json();
 m.textContent=`TTFT ${first?first.toFixed(0):'?'} ms (server ${s.ttft_ms} ms, prefill ${s.prefill_tokens} tok / ${s.prefill_ms} ms, ${s.mode}${s.spec===true?' · spec hit':s.spec===false?' · spec miss':''}) · ${s.decode_tps} tok/s`;};
</script></body></html>"""


def make_handler(g: LiveGemma):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            if self.path == "/":
                b = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif self.path == "/stats":
                self._json(200, {**g.stats, "threads": g.a.threads, "top_k": g.a.top_k,
                                 "load_s": round(g.load_s, 2), "system_prefill_ms": round(g.sys_ms, 1)})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": "gemma-4-e2b-cpu", "object": "model"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = self._body()
            except ValueError:
                return self._json(400, {"error": "bad json"})
            if self.path == "/prefill":
                try:
                    return self._json(200, g.prefill_partial(body.get("messages", []), body.get("partial", "")))
                except Exception as e:  # noqa: BLE001
                    return self._json(500, {"error": str(e)})
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            msgs = body.get("messages", [])
            max_tokens = int(body.get("max_tokens") or 256)
            rid = f"chatcmpl-{int(time.time() * 1000)}"
            if not body.get("stream"):
                try:
                    reply, stats = g.generate(msgs, max_tokens, lambda _t: None)
                except Exception as e:  # noqa: BLE001
                    return self._json(500, {"error": str(e)})
                return self._json(200, {"id": rid, "object": "chat.completion", "model": "gemma-4-e2b-cpu",
                                        "choices": [{"index": 0, "message": {"role": "assistant", "content": reply},
                                                     "finish_reason": "length" if stats["cut"] else "stop"}],
                                        "usage": {"prompt_tokens": stats["prefill_tokens"],
                                                  "completion_tokens": stats["decode_tokens"]},
                                        "x_stats": stats})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            gone = {"v": False}

            def emit(obj):
                try:
                    self.wfile.write(f"data: {json.dumps(obj) if not isinstance(obj, str) else obj}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    gone["v"] = True

            def on_text(t):
                emit({"id": rid, "object": "chat.completion.chunk", "model": "gemma-4-e2b-cpu",
                      "choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]})
            try:
                _, stats = g.generate(msgs, max_tokens, on_text, is_gone=lambda: gone["v"])
                emit({"id": rid, "object": "chat.completion.chunk", "model": "gemma-4-e2b-cpu",
                      "choices": [{"index": 0, "delta": {}, "finish_reason": "length" if stats["cut"] else "stop"}],
                      "x_stats": stats})
            except Exception as e:  # noqa: BLE001
                emit({"error": str(e)})
            emit("[DONE]")
            self.close_connection = True

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8810)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--model", default=str(HERE / "models" / "gemma-4-E2B-it.litertlm"))
    ap.add_argument("--system", default=DEFAULT_SYSTEM)
    ap.add_argument("--cache-dir", default=str(CACHE_ROOT / "litert_cpu"))
    a = ap.parse_args()
    Path(a.cache_dir).mkdir(parents=True, exist_ok=True)
    g = LiveGemma(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(g))
    print(f"llm_cpu_server: http://127.0.0.1:{a.port}/ (load {g.load_s:.1f}s, threads {a.threads}, top-k {a.top_k})",
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
