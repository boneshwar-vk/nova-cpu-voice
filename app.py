"""Local voice assistant server: web page + /ws/voice, everything on the CPU, no network needed.

Run:
  .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8800
  open http://127.0.0.1:8800/
"""
from __future__ import annotations

import logging
import os
import socket
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import offline_engine
from realtime import voice_ws

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("nova.app")

HERE = Path(__file__).resolve().parent
app = FastAPI(title="Nova CPU voice")


@app.on_event("startup")
def _warm() -> None:
    """Load and warm every model at boot so the first turn runs at steady state (WARM=0 skips this)."""
    if os.environ.get("WARM", "1") != "1":
        return
    import numpy as np

    try:
        offline_engine.warmup(offline_engine.STT_LANGUAGE)
        # a realistic-length buffer: the first real turn then avoids shape-specific allocations
        offline_engine.stt_pcm((np.random.default_rng(0).standard_normal(16000 * 2) * 300).astype(np.int16).tobytes())
    except Exception:  # noqa: BLE001
        log.warning("STT warm-up skipped", exc_info=True)
    try:
        import open_tts

        r = open_tts.get(open_tts.DEFAULT_VOICE).synthesize("Sure, I can help you with that today.")
        log.info("TTS warm: voice %s (%d Hz, %d threads), %.0f ms",
                 r["voice"], r["sample_rate"], open_tts.TTS_THREADS, r["latency_ms"])
    except Exception:  # noqa: BLE001
        log.warning("TTS warm-up skipped", exc_info=True)
    try:
        from realtime import _get_semantic

        sem = _get_semantic()  # the first Smart Turn inference pays graph optimisation: do it now
        if sem is not None:
            sem.predict(np.zeros(3 * 16000, dtype=np.float32))
    except Exception:  # noqa: BLE001
        log.warning("turn detector warm-up skipped", exc_info=True)
    try:
        from realtime import _echo_corr, _subtract_echo

        # scipy.signal import + first FFT plan cost ~1 s; the first barge-in check must not pay it.
        rng = np.random.default_rng(0)
        mic, ref = rng.standard_normal(12800).astype(np.float32), rng.standard_normal(20800).astype(np.float32)
        _echo_corr(mic, ref)
        _subtract_echo(mic, ref)
    except Exception:  # noqa: BLE001
        log.warning("echo check warm-up skipped", exc_info=True)
    try:
        import llm_session
        import personas

        system = personas.runtime_system(personas.ORDER[0])
        s = llm_session.get(system)  # loads the engine + prefills the system prompt
        for _ in s.stream([], "Hi there."):  # one short real turn so the first user turn is steady-state
            pass
        with s.g.lock:
            s.g._new_session(system)
        log.info("LLM warmed (load %.0f ms)", s.load_ms)
    except Exception:  # noqa: BLE001
        log.warning("LLM warm-up skipped", exc_info=True)
    try:
        import embedder
        import memory_store

        if embedder.available():
            e = embedder.get()
            for p in memory_store.PROFILES:
                memory_store.get(p)
            _, ms = memory_store.get(memory_store.DEFAULT_PROFILE).retrieve("warm up the memory scan")
            log.info("memory warmed: embedder %s (%.0f ms load), retrieve %.2f ms", embedder.MODEL_NAME, e.load_ms, ms)
    except Exception as e:  # noqa: BLE001
        log.warning("memory warm-up failed: %s", e)
    try:
        offline_engine.stt_partial(np.zeros(32000, np.int16).tobytes())  # live-caption model
    except Exception as e:  # noqa: BLE001
        log.warning("partial STT warm-up failed: %s", e)
    try:
        import residency

        residency.start()  # KEEP_RESIDENT=1 (macOS): wire the warmed working set, see residency.py
    except Exception as e:  # noqa: BLE001
        log.warning("residency lock failed: %s", e)
    log.info("offline pipeline warmed (stt=%s llm=%s)", offline_engine.stt_label(), offline_engine.LLM_BACKEND)


def ok(data):
    return JSONResponse({"ok": True, "data": data})


def fail(code: str, message: str, status: int = 400):
    return JSONResponse({"ok": False, "error": {"code": code, "message": message}}, status)


@app.get("/api/pipeline")
def pipeline():
    """What the pipeline is running with (for the page's Details panel)."""
    import vad

    info = offline_engine.available()
    return ok({
        "denoise": {"on": vad.VAD_DENOISE, "model": "GTCRN (sherpa-onnx, VAD branch only)"},
        "vad": {"model": "Silero VAD v6", **{k: int(int(float(os.environ.get(e, d)) / 32) * 32) for k, e, d in
                (("soft_ms", "TURN_SOFT_MS", "200"), ("check_ms", "TURN_CHECK_MS", "700"), ("hard_ms", "TURN_HARD_MS", "1500"))}},
        "turn": {"model": "Smart Turn v3.2", "threads": vad.SMART_TURN_THREADS,
                 "soft_threshold": vad.SemanticEndpointer.THRESHOLD_SOFT,
                 "check_threshold": vad.SemanticEndpointer.THRESHOLD_CHECK,
                 "available": vad.SemanticEndpointer.available()},
        "stt": {"model": info.get("whisper"), "language": offline_engine.STT_LANGUAGE},
        "llm": {"model": info.get("model"), "runtime": info.get("runtime"),
                "max_tokens": offline_engine.LLM_MAX_TOKENS},
        "tts": _tts_info(),
        "memory": _memory_info(),
        "residency": __import__("residency").stats(),
    })


def _tts_info() -> dict:
    import open_tts

    v = open_tts.DEFAULT_VOICE
    eng = open_tts._engines.get(v)  # loaded at startup warm-up; never built here
    return {"voice": v, "model": open_tts.VOICES[v]["display"] + ", one sentence per synthesis",
            "sample_rate": eng.sample_rate if eng else None, "threads": open_tts.TTS_THREADS,
            "open_source": True, "loaded": eng is not None}


@app.get("/api/voices/open")
def voices_open():
    """Installed TTS voices for the page's voice picker."""
    import open_tts

    return ok({"default": open_tts.DEFAULT_VOICE,
               "voices": [{"id": k, "display": v["display"], "default": k == open_tts.DEFAULT_VOICE}
                          for k, v in open_tts.VOICES.items() if open_tts.available(k)]})


def _memory_info() -> dict:
    import embedder
    import memory_store
    from realtime import MEMORY_ON

    return {"on": MEMORY_ON, "embedder": embedder.MODEL_NAME, "dim": embedder.DIM,
            "search": "exact in-memory scan over SQLite-stored vectors",
            "top_k": memory_store.TOP_K, "min_cos": memory_store.MIN_COS,
            "profiles": memory_store.profiles() if MEMORY_ON else []}


def _profile(profile: str) -> str:
    import personas

    return profile if profile in personas.PERSONAS else personas.ORDER[0]


@app.get("/api/sessions")
def sessions_list(profile: str = ""):
    """Past conversations of a profile (newest first)."""
    import conv_store

    p = _profile(profile)
    return ok({"profile": p, "sessions": conv_store.sessions(p), "file": f"{p}_conversations.sqlite"})


@app.get("/api/sessions/{sid}")
def session_turns(sid: int, profile: str = ""):
    import conv_store

    p = _profile(profile)
    return ok({"profile": p, "id": sid, "turns": conv_store.turns(p, sid)})


@app.delete("/api/sessions")
def sessions_clear(profile: str = ""):
    """Delete all saved conversations of a profile."""
    import conv_store

    p = _profile(profile)
    return ok({"profile": p, "deleted": conv_store.delete(p)})


@app.delete("/api/sessions/{sid}")
def session_delete(sid: int, profile: str = ""):
    import conv_store

    p = _profile(profile)
    return ok({"profile": p, "deleted": conv_store.delete(p, sid)})


@app.get("/api/templates")
def templates_list(profile: str = ""):
    """The use-case templates and the one this profile uses."""
    import templates

    p = _profile(profile)
    return ok({"profile": p, "current": templates.current(p), "templates": templates.listing()})


class TemplateChoice(BaseModel):
    profile: str
    template: str


@app.post("/api/templates")
def templates_choose(c: TemplateChoice):
    """Pick a template (takes effect on the next connection)."""
    import personas
    import templates

    if c.profile not in personas.PERSONAS:
        return fail("BAD_PROFILE", "unknown profile")
    try:
        t = templates.choose(c.profile, c.template)
    except ValueError as e:
        return fail("BAD_TEMPLATE", str(e))
    return ok({"profile": c.profile, "current": t})


@app.get("/api/library")
def library(profile: str = ""):
    """The profile's document library (rag_store.py): titles, kinds, pages, passages."""
    p = _profile(profile)
    try:
        import rag_store

        return ok({"profile": p, "docs": rag_store.get(p).docs(), "file": f"{p}_library.sqlite"})
    except Exception as e:  # noqa: BLE001
        return ok({"profile": p, "docs": [], "file": "", "note": f"library unavailable: {e}"})


@app.get("/api/offline")
def offline_status():
    """True when the computer has no route to the internet. A UDP connect() only asks the OS for a
    route; no packet is sent."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))  # TEST-NET-1 address: never contacted
        routed = s.getsockname()[0] not in ("0.0.0.0", "127.0.0.1")
    except OSError:
        routed = False
    finally:
        s.close()
    return ok({"offline": not routed})


@app.get("/api/memory/profiles")
def memory_profiles():
    import memory_store

    return ok({"default": memory_store.DEFAULT_PROFILE, "profiles": memory_store.profiles()})


@app.get("/api/memory")
def memory_list(profile: str = ""):
    import memory_store

    try:
        s = memory_store.get(profile or None)
    except ValueError as e:
        return fail("BAD_PROFILE", str(e))
    return ok({"profile": s.profile, "file": s.path.name, "facts": s.list()})


@app.post("/api/memory/reset")
def memory_reset(profile: str = ""):
    import memory_store

    s = memory_store.get(profile or None)
    return ok({"profile": s.profile, "facts": s.reset()})


@app.delete("/api/memory/{fact_id}")
def memory_delete(fact_id: int, profile: str = ""):
    import memory_store

    s = memory_store.get(profile or None)
    if not s.delete(fact_id):
        return fail("NOT_FOUND", f"fact {fact_id} not in {s.profile}", 404)
    return ok({"profile": s.profile, "deleted": fact_id})


@app.websocket("/ws/voice")
async def ws_voice(ws: WebSocket):
    await voice_ws(ws)


@app.middleware("http")
async def _no_stale_pages(request, call_next):
    """Pages and their scripts must never come from a stale browser cache after an update."""
    resp = await call_next(request)
    p = request.url.path
    if p == "/" or p.endswith((".html", ".js", ".css")):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


app.mount("/", StaticFiles(directory=str(HERE / "static"), html=True), name="static")
