"""Multi-run end-to-end voicebot latency bench for /ws/voice.

Per run it opens a NEW websocket, streams
a 16 kHz PCM16 clip as 512-sample frames paced at 32 ms real time (like a live
mic), then `silence_s` of zero frames to trigger the endpointer, and measures
client-perceived TTFA = last speech frame sent -> first binary audio chunk.
Inter-chunk dead air is flagged when chunk n+1 arrives later than chunk n's
playback duration (parsed from its WAV header). If the server sends
{"type": "trace", ...} (turn_trace.py) before audio_end, its fields are passed
through flattened as srv_<key> and every srv_*_ms field gets a p50 column.

Usage (from the repo root, server running; clips from bench/make_clips.py):
  .venv/bin/python bench/e2e_bench.py --runs 5 --tag cpu
  .venv/bin/python bench/e2e_bench.py --only question --runs 1
  .venv/bin/python bench/e2e_bench.py --selftest --runs 2   # no server needed
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

HERE = Path(__file__).resolve().parent      # bench/
ROOT = HERE.parent                          # clip / log paths are relative to the repo root
SR = 16000
FRAME = 512                                 # samples per frame (~32 ms), matches vad.py
FRAME_S = FRAME / SR
SILENCE = b"\x00\x00" * FRAME
ENV_KEYS = ("STT_BACKEND", "STT_LANGUAGE", "TTS_VOICE", "LITERT_THREADS")


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() or p.exists() else ROOT / p


def _pcm16_16k(path: Path) -> bytes:
    a, sr = sf.read(str(path), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        n = int(round(len(a) * SR / sr))
        a = np.interp(np.linspace(0, len(a), n, endpoint=False), np.arange(len(a)), a)
    return (np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes()


def _wav_secs(b: bytes) -> float:
    """Playback seconds of a WAV blob from its header (0.0 if not a WAV)."""
    if len(b) < 12 or b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        return 0.0
    pos, byte_rate = 12, 0
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], int.from_bytes(b[pos + 4:pos + 8], "little")
        if cid == b"fmt ":
            byte_rate = int.from_bytes(b[pos + 16:pos + 20], "little")
        elif cid == b"data":
            avail = len(b) - pos - 8
            n = avail if size in (0, 0xFFFFFFFF) else min(size, avail)
            return n / byte_rate if byte_rate else 0.0
        pos += 8 + size + (size & 1)
    return 0.0


async def _wait_any(events: list[asyncio.Event], timeout: float) -> bool:
    """True once any event is set within `timeout` seconds."""
    if any(e.is_set() for e in events):
        return True
    if timeout <= 0:
        return False
    tasks = [asyncio.create_task(e.wait()) for e in events]
    fired, pending = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    return bool(fired)


async def run_clip(url: str, clip_path: str | Path, silence_s: float, timeout_s: float = 20,
                   config: dict | None = None, lead_silence_s: float = 0.0) -> dict:
    """Drive one turn over a fresh websocket; see the module docstring for metrics.

    `lead_silence_s` of zero frames go first so the VAD's background gate can
    learn a quiet room (speech from frame 0 on a fresh socket is otherwise
    treated as background talk and dropped after ~2 s by vad.py's noise_reset)."""
    clip_path = _resolve(clip_path)
    r: dict = {"clip": clip_path.stem, "ttfa_ms": None, "first_audio_wall": None, "n_chunks": 0,
               "dead_air_violations": 0, "transcript": None, "reply": None,
               "audio_end_seen": False, "trace": None, "barge_in": 0, "error": None}
    chunks: list[tuple[float, float]] = []      # (arrival perf_counter, wav seconds)
    first_audio, done = asyncio.Event(), asyncio.Event()

    async def rx(ws) -> None:
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    chunks.append((time.perf_counter(), _wav_secs(msg)))
                    if not first_audio.is_set():
                        r["first_audio_wall"] = time.time()
                        first_audio.set()
                    continue
                m = json.loads(msg)
                t = m.get("type")
                if t == "transcript":
                    r["transcript"] = m.get("text")
                elif t == "reply":
                    r["reply"] = m.get("text")
                elif t == "trace":
                    r["trace"] = m
                elif t == "barge_in":
                    r["barge_in"] += 1
                elif t == "error":
                    r["error"] = m.get("message") or "error"
                    return
                elif t == "audio_end":
                    r["audio_end_seen"] = True
                    return
        except Exception as e:  # closed socket, bad JSON, ...
            r["error"] = r["error"] or f"{type(e).__name__}: {e}"
        finally:
            done.set()

    try:
        pcm = _pcm16_16k(clip_path)
        async with websockets.connect(url, max_size=None) as ws:
            if config:
                await ws.send(json.dumps(config))
            task = asyncio.create_task(rx(ws))
            for _ in range(int(lead_silence_s / FRAME_S)):  # room learning
                await ws.send(SILENCE)
                await asyncio.sleep(FRAME_S)
            for i in range(0, len(pcm), FRAME * 2):          # speech, real-time paced
                await ws.send(pcm[i:i + FRAME * 2])
                await asyncio.sleep(FRAME_S)
            t_speech_end = time.perf_counter()
            for _ in range(int(silence_s / FRAME_S)):         # trailing silence -> endpoint
                if done.is_set():
                    break
                await ws.send(SILENCE)
                await asyncio.sleep(FRAME_S)
            # First audio must arrive within timeout_s of the last speech frame.
            await _wait_any([first_audio, done], t_speech_end + timeout_s - time.perf_counter())
            if first_audio.is_set():
                r["ttfa_ms"] = round((chunks[0][0] - t_speech_end) * 1000, 1)
                if not await _wait_any([done], timeout_s):
                    r["error"] = r["error"] or "no audio_end"
            elif not done.is_set():
                r["error"] = "timeout"
            elif r["error"] is None:
                r["error"] = "no audio"
            task.cancel()
    except (OSError, websockets.WebSocketException, asyncio.TimeoutError) as e:
        r["error"] = r["error"] or f"{type(e).__name__}: {e}"

    r["n_chunks"] = len(chunks)
    r["dead_air_violations"] = sum(
        1 for i in range(1, len(chunks)) if chunks[i][0] - chunks[i - 1][0] > chunks[i - 1][1])
    for k, v in (r["trace"] or {}).items():
        if k != "type":
            r[f"srv_{k}"] = v
    return r


# ----------------------------------------------------------------------------- reporting

def _num(v) -> float | None:
    """Numeric view of a trace value: numbers as-is, number lists summed, else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, list) and v and all(_num(x) is not None for x in v):
        return float(sum(v))
    return None


def _p(vals: list[float], q: float) -> int | None:
    return round(float(np.percentile(vals, q))) if vals else None


def summarize(rows: list[dict]) -> tuple[str, dict]:
    """Markdown table per clip + overall; returns (table, overall {p50, p90})."""
    srv_cols: list[str] = []
    for r in rows:
        srv_cols += [k for k in r if k.startswith("srv_") and k.endswith("_ms")
                     and k not in srv_cols and _num(r[k]) is not None]
    groups = [(cid, [r for r in rows if r["id"] == cid]) for cid in dict.fromkeys(r["id"] for r in rows)]
    groups.append(("**overall**", rows))
    cols = ["clip", "kind", "n", "ttfa p50", "p90", "min", "max", "timeouts", "errors", "dead-air"]
    cols += [f"{c[4:]} p50" for c in srv_cols]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    overall: dict = {}
    for name, rs in groups:
        t = [r["ttfa_ms"] for r in rs if r["ttfa_ms"] is not None]
        kinds = sorted({r["kind"] for r in rs})
        cells = [name, kinds[0] if len(kinds) == 1 else "mixed", len(rs),
                 _p(t, 50), _p(t, 90), _p(t, 0), _p(t, 100),
                 sum(r["error"] == "timeout" for r in rs),
                 sum(bool(r["error"]) and r["error"] != "timeout" for r in rs),
                 sum(r["dead_air_violations"] for r in rs)]
        cells += [_p([x for x in (_num(r.get(c)) for r in rs) if x is not None], 50) for c in srv_cols]
        lines.append("| " + " | ".join("-" if c is None else str(c) for c in cells) + " |")
        overall = {"p50": cells[3], "p90": cells[4]}
    return "\n".join(lines), overall


def _run_line(r: dict, i: int, n: int) -> str:
    ttfa = "   -- " if r["ttfa_ms"] is None else f"{r['ttfa_ms']:6.0f}"
    trig = r.get("srv_commit_trigger") or "-"
    err = f"  ERR {r['error']}" if r["error"] else ""
    return (f"{r['id']:<12} run {i:>2}/{n}  ttfa {ttfa} ms  commit={trig:<7} "
            f"chunks={r['n_chunks']:<2} dead_air={r['dead_air_violations']}  {r['transcript']!r}{err}")


def report(rows: list[dict], args: argparse.Namespace, out: Path) -> int:
    table, overall = summarize(rows)
    try:
        shown = out.resolve().relative_to(ROOT)
    except ValueError:
        shown = out.name
    print(f"\n{len(rows)} runs appended to {shown}\n\n{table}")
    bad = [f"p{q}={overall[f'p{q}']} > gate {g:g}" for q, g in ((50, args.gate_p50), (90, args.gate_p90))
           if g is not None and (overall[f"p{q}"] is None or overall[f"p{q}"] > g)]
    if bad:
        print("\nGATE FAILED: " + "; ".join(bad))
        return 1
    return 0


# ----------------------------------------------------------------------------- runner

def load_clips(path: Path, only: str | None) -> list[dict]:
    clips = json.loads(path.read_text())
    if only:
        want = [s.strip() for s in only.split(",") if s.strip()]
        missing = set(want) - {c["id"] for c in clips}
        if missing:
            sys.exit(f"unknown clip id(s): {', '.join(sorted(missing))}")
        clips = [c for c in clips if c["id"] in want]
    return clips


async def bench(args: argparse.Namespace, clips: list[dict], url: str, out: Path) -> list[dict]:
    config = None
    if args.language or args.voice:
        config = {"type": "config"}
        if args.voice:
            config["voice"] = args.voice
        if args.language:
            config["language"] = args.language
    ctx = {"pipe_tag": os.environ.get("PIPE_TAG", "")}
    ctx.update({k: os.environ[k] for k in ENV_KEYS if os.environ.get(k)})
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with out.open("a") as f:
        for ci, clip in enumerate(clips):
            for i in range(1, args.runs + 1):
                r = await run_clip(url, clip["path"], clip["silence_s"], args.timeout, config, args.lead_silence)
                r.update(id=clip["id"], kind=clip["kind"], run=i, tag=args.tag,
                         timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"), **ctx)
                f.write(json.dumps(r) + "\n")
                f.flush()
                rows.append(r)
                print(_run_line(r, i, args.runs), flush=True)
                if args.pause and (ci < len(clips) - 1 or i < args.runs):
                    await asyncio.sleep(args.pause)     # let the server settle
    return rows


# ----------------------------------------------------------------------------- selftest

def _silence_wav(ms: int, sr: int = 22050) -> bytes:
    data = b"\x00\x00" * (sr * ms // 1000)
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", len(data)) + data)


async def _fake_voice_server(ws) -> None:
    """Stand-in for /ws/voice: endpoints after ~1.3 s of post-speech silence, then
    sends state/transcript/reply/trace, two 300 ms WAV chunks and audio_end."""
    heard, silent, t_speech_end = False, 0, 0.0
    async for msg in ws:
        if not isinstance(msg, bytes):
            continue                                    # config message
        if max(abs(v) for v in memoryview(msg).cast("h")) > 32:
            heard, silent, t_speech_end = True, 0, time.monotonic()
        elif heard:
            silent += 1
        if not heard or silent * FRAME_S < 1.3:
            continue
        heard = False                                   # one turn per connection
        t0 = time.monotonic()
        await ws.send(json.dumps({"type": "state", "value": "processing"}))
        await asyncio.sleep(0.15)
        await ws.send(json.dumps({"type": "transcript", "text": "selftest transcript"}))
        await ws.send(json.dumps({"type": "reply", "text": "Selftest reply."}))
        await ws.send(json.dumps({
            "type": "trace", "speech_end_t": t_speech_end, "soft_t": t_speech_end + 0.24,
            "smart_turn_ms": [70.0], "commit_t": t0, "commit_trigger": "soft",
            "stt_ms": 600, "stt_eager_used": True, "llm_ttft_ms": 400,
            "llm_first_sentence_ms": 900, "tts_first_ms": 500, "first_audio_t": time.monotonic(),
            "ttfa_ms": round((time.monotonic() - t_speech_end) * 1000, 1)}))
        await ws.send(json.dumps({"type": "state", "value": "speaking"}))
        await ws.send(_silence_wav(300))
        await asyncio.sleep(0.1)                        # < 300 ms, so no dead air
        await ws.send(_silence_wav(300))
        await ws.send(json.dumps({"type": "audio_end"}))


async def selftest(args: argparse.Namespace, clips: list[dict], out: Path) -> int:
    async with websockets.serve(_fake_voice_server, "127.0.0.1", 0, max_size=None) as server:
        port = server.sockets[0].getsockname()[1]
        print(f"selftest: fake /ws/voice on 127.0.0.1:{port}")
        rows = await bench(args, clips[:1], f"ws://127.0.0.1:{port}/ws/voice", out)
    return report(rows, args, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="ws://127.0.0.1:8800/ws/voice")
    ap.add_argument("--clips", default=str(HERE / "clips.json"))
    ap.add_argument("--only", metavar="ID,ID", help="subset of clip ids from --clips")
    ap.add_argument("--runs", type=int, default=10, help="runs per clip")
    ap.add_argument("--tag", default=os.environ.get("PIPE_TAG") or "run")
    ap.add_argument("--out", help="JSONL append path (default logs/e2e_<tag>.jsonl)")
    ap.add_argument("--gate-p50", type=float, metavar="MS")
    ap.add_argument("--gate-p90", type=float, metavar="MS")
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between runs")
    ap.add_argument("--timeout", type=float, default=20, help="s to wait for first audio")
    ap.add_argument("--lead-silence", type=float, default=1.0,
                    help="s of silence before the clip so the VAD's background gate learns a quiet room")
    ap.add_argument("--language", help='send {"type":"config","language":...} first, e.g. en')
    ap.add_argument("--voice", help="send config voice first, e.g. matcha")
    ap.add_argument("--selftest", action="store_true",
                    help="run one clip against an in-process fake server (no uvicorn needed)")
    args = ap.parse_args()

    clips = load_clips(_resolve(args.clips), args.only)
    if args.selftest:
        args.tag = args.tag if args.tag != "run" else "selftest"
        out = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "e2e_selftest.jsonl"
        sys.exit(asyncio.run(selftest(args, clips, out)))
    out = _resolve(args.out or f"logs/e2e_{args.tag}.jsonl")
    rows = asyncio.run(bench(args, clips, args.url, out))
    sys.exit(report(rows, args, out))


if __name__ == "__main__":
    main()
