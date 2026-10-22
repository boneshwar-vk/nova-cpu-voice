"""Fully local speech recognition + language-model info for the voice pipeline (CPU only, no network).

  STT  : Moonshine v2 base (English) via sherpa-onnx; Moonshine v2 tiny for live display-only captions
  LLM  : Gemma 4 E2B int4 (.litertlm) on LiteRT-LM, CPU backend (llm_session.py)

Turn detection (vad.py) and TTS (open_tts.py) live in their own modules; realtime.py ties them together.
"""
from __future__ import annotations

import logging
import os
import re
import threading

import numpy as np

log = logging.getLogger("nova.engine")

LLM_BACKEND = "session"  # LiteRT-LM raw session on CPU (llm_session.py)
# English only by default: the shipped STT models are English.
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "en") or None
# sherpa-onnx STT spec "<kind>:<dir under models/>" (see stt_sherpa.py). Default = Moonshine v2 base:
# fast on CPU, punctuated output, robust to leading/trailing silence.
STT_BACKEND = os.environ.get(
    "STT_BACKEND", "moonshine_v2:sherpa-onnx-moonshine-base-en-quantized-2026-02-27")
# Voice replies: 1-3 short sentences; 120 tokens is ample and caps the tail.
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "120"))
# Rolling conversation memory sent to the LLM (user+assistant pairs).
LLM_HISTORY_TURNS = int(os.environ.get("LLM_HISTORY_TURNS", "6"))

_sherpa = None
_sherpa_lock = threading.Lock()


def _get_sherpa():
    global _sherpa
    if _sherpa is None:
        with _sherpa_lock:
            if _sherpa is None:
                from stt_sherpa import SherpaStt  # lazy: heavy import

                _sherpa = SherpaStt(STT_BACKEND)
    return _sherpa


# Live partial transcript while the person is still talking (display only): a separate,
# smaller Moonshine tiny instance with 2 threads, so it never queues behind or slows the
# final transcription (base). Off with PARTIAL_STT=0.
PARTIAL_STT = os.environ.get("PARTIAL_STT", "1") == "1"
PARTIAL_STT_BACKEND = os.environ.get(
    "PARTIAL_STT_BACKEND", "moonshine_v2:sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27")
_partial = None
_partial_lock = threading.Lock()


def stt_partial(pcm16: bytes, sr: int = 16000) -> str:
    """Quick display-only transcript of the audio so far (Moonshine tiny)."""
    global _partial
    if not PARTIAL_STT or len(pcm16) < sr:  # under ~0.5 s of audio: nothing useful yet
        return ""
    if _partial is None:
        with _partial_lock:
            if _partial is None:
                from stt_sherpa import SherpaStt

                _partial = SherpaStt(PARTIAL_STT_BACKEND, num_threads=2)
    audio = np.frombuffer(pcm16[: len(pcm16) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
    return collapse_repeats(_partial.transcribe(audio, sr))


def collapse_repeats(text: str) -> str:
    """Cut a decoder loop on audio that ends mid-word: "what is the name of the name of the name"
    -> "what is the name of". Display-only partials; the final transcript is untouched."""
    text = re.sub(r"\b([\w']{1,6})(?:[- ]\1){2,}\b[-\w]*", r"\1", text)  # "11-11-11-11" / "ha ha ha ha"
    text = re.sub(r"\s*\b\w*?([A-Za-z])\1{3,}\w*", "", text, flags=re.I).strip()  # "Kmw Lllllllll" (a stuck character)
    w = text.split()
    for i in range(len(w)):
        for n in (5, 4, 3, 2, 1):
            if i + 2 * n <= len(w) and [x.lower().strip(".,?!") for x in w[i:i + n]] == [x.lower().strip(".,?!") for x in w[i + n:i + 2 * n]] \
                    and (n > 1 or (i + 3 <= len(w) and w[i].lower() == w[i + 2].lower())):
                return " ".join(w[: i + n])
    return text


def stt_label() -> str:
    return STT_BACKEND


def warmup(language: str | None = None) -> None:
    """Pre-load the STT model so the first real turn isn't cold."""
    _get_sherpa().transcribe(np.zeros(8000, dtype=np.float32))


def _windows(audio: np.ndarray, sr: int, max_s: float = 8.0, min_s: float = 5.0) -> list[np.ndarray]:
    out = []
    while len(audio) > max_s * sr:
        lo, hi, w = int(min_s * sr), int(max_s * sr), int(0.2 * sr)
        seg = audio[lo:hi]
        n = (len(seg) - w) // (w // 4) + 1
        e = [float(np.mean(seg[i * (w // 4): i * (w // 4) + w] ** 2)) for i in range(max(n, 1))]
        cut = lo + int(np.argmin(e)) * (w // 4) + w // 2
        out.append(audio[:cut])
        audio = audio[cut:]
    out.append(audio)
    return [a for a in out if len(a) > 0.2 * sr]


def stt_pcm(pcm16: bytes, sr: int = 16000, language: str | None = None) -> str:
    """Transcribe raw PCM16 mono (16 kHz) locally."""
    if not pcm16 or len(pcm16) < 2:
        return ""
    if len(pcm16) % 2:  # np.int16 needs an even byte count
        pcm16 = pcm16[:-1]
    audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
    # Moonshine v2 returns nothing on clips longer than ~9 s: decode long turns in pieces cut at the
    # quietest moment between 5 and 8 s, so a long question is never silently dropped.
    return " ".join(t for t in (_get_sherpa().transcribe(w, sr) for w in _windows(audio, sr)) if t).strip()


def available() -> dict:
    """Readiness and labels for the UI."""
    import llm_session

    return {"ready": llm_session.available(), "llm_up": llm_session.available(),
            "model": os.path.basename(llm_session.MODEL).replace(".litertlm", ""),
            "runtime": f"LiteRT-LM 0.17.1 session · CPU {llm_session.THREADS} threads · speculative prefill",
            "device": "cpu", "whisper": stt_label()}
