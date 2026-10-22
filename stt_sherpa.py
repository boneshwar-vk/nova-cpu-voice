"""sherpa-onnx STT backends for the offline engine.

Selected with STT_BACKEND (see offline_engine.py):
  moonshine_v2:<dir>   Moonshine v2 (encoder_model.ort + decoder_model_merged.ort + tokens.txt)
  nemo_ctc:<dir>       NeMo CTC, e.g. Parakeet-TDT_CTC-110M int8 (model.int8.onnx + tokens.txt)
  sherpa_whisper:<dir> sherpa whisper export (<name>-encoder.int8.onnx / -decoder.int8.onnx / -tokens.txt)
<dir> is absolute or relative to models/. Only Moonshine v2 models are downloaded by
scripts/download_models.py; the other kinds work if you fetch their sherpa-onnx exports yourself.

Moonshine can repeat itself on truncated (eager) buffers, hence `looks_degenerate()` for the eager decode.
"""
from __future__ import annotations

import glob
import os
import re
import threading
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent / "models"
STT_THREADS = int(os.environ.get("STT_THREADS", "4"))


def _pick(d: Path, *patterns: str) -> str:
    """First file in d matching any pattern, preferring int8/quantized variants."""
    cands: list[str] = []
    for pat in patterns:
        cands += glob.glob(str(d / pat))
    if not cands:
        raise FileNotFoundError(f"no {patterns} in {d}")
    cands.sort(key=lambda p: (("int8" not in p) and ("quant" not in p), len(p)))
    return cands[0]


class SherpaStt:
    def __init__(self, spec: str, num_threads: int = STT_THREADS) -> None:
        import sherpa_onnx  # lazy: heavy import

        kind, _, path = spec.partition(":")
        d = Path(path) if os.path.isabs(path) else MODELS_DIR / path
        if not d.is_dir():
            raise FileNotFoundError(f"STT model dir not found: {d}")
        self.kind, self.dir = kind, d
        tokens = _pick(d, "tokens.txt", "*tokens.txt")
        R = sherpa_onnx.OfflineRecognizer
        if kind == "moonshine_v2":
            self.rec = R.from_moonshine_v2(
                encoder=_pick(d, "encoder*.ort", "encoder*.onnx"),
                decoder=_pick(d, "decoder*merged*.ort", "decoder*merged*.onnx", "decoder*.ort"),
                tokens=tokens, num_threads=num_threads)
        elif kind == "nemo_ctc":
            self.rec = R.from_nemo_ctc(model=_pick(d, "model*.onnx"), tokens=tokens,
                                       num_threads=num_threads)
        elif kind == "sherpa_whisper":
            self.rec = R.from_whisper(
                encoder=_pick(d, "*encoder*.onnx"), decoder=_pick(d, "*decoder*.onnx"),
                tokens=tokens, language="en", task="transcribe", num_threads=num_threads)
        else:
            raise ValueError(f"unknown sherpa STT kind {kind!r}")
        self._lock = threading.Lock()  # one recogniser, one decode at a time

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> str:
        with self._lock:
            s = self.rec.create_stream()
            s.accept_waveform(sr, audio.astype(np.float32))
            self.rec.decode_stream(s)
            return (s.result.text or "").strip()


# A 1-2 word unit repeated 3+ times ("were in simple were in simple were") or a
# 3-8 word unit repeated 2+ times ("I think photosynthesis is really fascinating"
# twice) is a decoder loop, not speech.
_REPEAT_SHORT = re.compile(r"\b((?:\w+\W+){1,2}?)(?:\1){2,}", re.IGNORECASE)
_REPEAT_LONG = re.compile(r"\b((?:\w+\W+){3,8}?)(?:\1){1,}", re.IGNORECASE)


def looks_degenerate(text: str, audio_s: float) -> bool:
    """Heuristic for the eager (truncated-buffer) decode: repeated n-grams or an
    implausible word rate mean the small model looped; the caller then waits
    for the full decode at commit instead of trusting this transcript."""
    if not text:
        return False
    t = re.sub(r"[^\w\s]", " ", text.lower()) + " "
    if _REPEAT_SHORT.search(t) or _REPEAT_LONG.search(t):
        return True
    words = len(text.split())
    return audio_s > 0 and words / audio_s > 6.0  # >6 words/s is not speech
