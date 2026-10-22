"""Gemma 4 E2B on the LiteRT-LM v0.17.1 raw session API, CPU only (the default LLM backend).

Reuses llm_cpu_server.LiveGemma:
  * one live session; the system prompt is prefilled once and checkpointed;
  * each turn prefills only what the KV cache lacks (normally just the new user chunk);
  * speculative prefill of the eager transcript at the soft endpoint (prefill_partial); a commit
    whose user chunk is identical decodes at once, anything else rewinds for free;
  * a barge-in keeps only the heard part of the reply (rewind to the pre-reply checkpoint + heard text).
One process-wide instance (single local user). Thread-safe through LiveGemma.lock.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
log = logging.getLogger("nova.session")

THREADS = int(os.environ.get("LITERT_THREADS", "4"))
DEVICE = "cpu"
KV_TOKENS = int(os.environ.get("LITERT_MAX_TOKENS", "3072"))  # 3072: room for ~960 tokens of tool declarations
CACHE_DIR = os.environ.get("LITERT_CACHE_DIR", str(HERE / "models" / ".cache" / "litert_cpu"))
MODEL = os.environ.get("LITERT_MODEL", str(HERE / "models" / "gemma-4-E2B-it.litertlm"))
KV_RESET_AT = int(KV_TOKENS * 0.8)  # fresh session (+ short history replay) before the KV cache fills

_inst = None
_inst_lock = threading.Lock()


class SessionLLM:
    def __init__(self, system: str, top_k: int = 40, temperature: float = 0.8) -> None:
        from llm_cpu_server import LiveGemma

        Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
        a = SimpleNamespace(model=MODEL, max_tokens=KV_TOKENS, threads=THREADS, cache_dir=CACHE_DIR, backend=DEVICE,
                            system=system, top_k=top_k, temperature=temperature)
        t0 = time.perf_counter()
        self.g = LiveGemma(a)
        self.system = system
        self.load_ms = (time.perf_counter() - t0) * 1000
        self._gone = threading.Event()
        self.last: dict = {}
        self.epoch = 0  # bumped by every stream(); a speculative prefill from an older epoch is stale
        self._drop_prefix: list[dict] = []
        self._sys_tok: tuple = (None, 80)  # (system text, its token count): the system turn is part of the KV cache
        self.drop = 0  # messages of the caller's (append-only) history that the fresh session left out
        log.info("session LLM ready: v0.17.1 raw session on CPU, %d threads, KV %d, load %.0f ms (system prefill %.0f ms)",
                 THREADS, KV_TOKENS, self.load_ms, self.g.sys_ms)

    @staticmethod
    def _messages(system: str, history: list[dict], user: str) -> list[dict]:
        return [{"role": "system", "content": system}] + list(history) + [{"role": "user", "content": user}]

    def _kv_tokens(self) -> int:
        try:
            if self._sys_tok[0] != self.g.system:
                self._sys_tok = (self.g.system, self.g.eng.ntok(self.g.system) + 8)
            return sum(self.g.eng.ntok(f"{u} {a or ''}") + 12 for u, a in self.g.turns) + self._sys_tok[1]
        except Exception:  # noqa: BLE001
            return 0

    def _hist(self, history: list[dict]) -> list[dict]:
        # `drop` belongs to the conversation it was computed for: any history that does not start with
        # the same messages (a new websocket or profile) is passed whole.
        if self.drop and (len(history) < self.drop or history[: self.drop] != self._drop_prefix):
            self.drop = 0
        return history[self.drop:]

    def _reset_if_full(self, history: list[dict], system: str, upcoming: int) -> None:
        """Fresh session (+ the last two turns replayed) before this turn could overflow the KV cache."""
        try:
            need = self._kv_tokens() + upcoming
        except Exception:  # noqa: BLE001
            need = 0
        if need > KV_TOKENS - 96:  # would not fit: start fresh now (in the user's pause when called from speculate)
            with self.g.lock:
                self.g._new_session(system)
            self.drop = max(0, len(history) - 4)  # replay only the last two turns, once
            self._drop_prefix = [dict(m) for m in history[: self.drop]]
            log.info("KV near full: fresh session, replaying %d messages", len(history) - self.drop)

    def _ntok(self, text: str) -> int:
        try:
            return self.g.eng.ntok(text)
        except Exception:  # noqa: BLE001
            return len(text) // 3

    def warm_system(self, system: str) -> None:
        """Switch the live session to a new system prompt now (one-time prefill), not on the first turn."""
        if system != self.g.system:
            with self.g.lock:
                self.g._new_session(system)
            self.drop = 0
            log.info("system prompt switched in %.0f ms (shared tool prefix reused: %s)", self.g.sys_ms, getattr(self.g, "prefix_hit", False))

    def speculate(self, history: list[dict], user: str, system: str | None = None, epoch: int | None = None) -> dict | None:
        """Prefill the eager user chunk while the turn detector is still deciding. With `epoch`, it is
        skipped once a turn has started generating since (a late prefill would roll that turn back)."""
        # Near the end of the KV cache: skip (the turn itself resets once, with room for the reply and notes).
        # Resetting here too raced with the turn's own reset and replayed whole conversations.
        if self._kv_tokens() + self._ntok(user) + 120 + 400 > KV_TOKENS - 96:
            log.info("spec prefill skipped: KV nearly full (the turn will start a fresh session)")
            return None
        history = self._hist(history)
        try:
            r = self.g.prefill_partial(self._messages(system or self.system, history, user), user,
                                       still_wanted=lambda: epoch is None or epoch == self.epoch)
            log.info("spec prefill: %s", r)
            return r
        except Exception as e:  # noqa: BLE001
            log.warning("spec prefill failed: %s", e)
            return None

    def stream(self, history: list[dict], user: str, max_tokens: int = 120,
               system: str | None = None) -> Iterator[str]:
        """Blocking iterator of text deltas; abandoning it cancels the decode."""
        system = system or self.system
        self._reset_if_full(history, system, self._ntok(user) + max_tokens + 400)  # + a possible notes turn
        history = self._hist(history)
        q: queue.Queue = queue.Queue()
        self.epoch += 1
        self._gone.clear()
        done = object()

        def run() -> None:
            try:
                _, st = self.g.generate(self._messages(system, history, user), max_tokens,
                                        on_text=q.put, is_gone=self._gone.is_set)
                self.last = st
                log.info("session turn: ttft %s ms, prefill %s tok / %s ms, %s, %s tok/s",
                         st.get("ttft_ms"), st.get("prefill_tokens"), st.get("prefill_ms"), st.get("mode"), st.get("decode_tps"))
            except Exception as e:  # noqa: BLE001
                log.warning("session generate failed (%s); resetting the session", e)
                try:
                    with self.g.lock:
                        self.g._new_session(system)
                except Exception:  # noqa: BLE001
                    pass
                q.put(e)
            finally:
                q.put(done)

        th = threading.Thread(target=run, name="session-llm", daemon=True)
        th.start()
        finished = False
        try:
            while True:
                item = q.get()
                if item is done:
                    finished = True
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            if not finished:
                self._gone.set()
                th.join(timeout=5)

    def cancel(self) -> None:
        self._gone.set()


def get(system: str) -> SessionLLM:
    global _inst
    with _inst_lock:
        if _inst is None:
            _inst = SessionLLM(system)
        return _inst


def available() -> bool:
    try:
        import litert_capi  # noqa: F401
        return Path(MODEL).exists()
    except Exception:  # noqa: BLE001
        return False
