"""Open-source TTS voices via sherpa-onnx (CPU only).

Default = Piper en_US ljspeech medium ("ljspeech", 22.05 kHz, trained from scratch on the public-domain
LJ Speech data). Optional: Matcha LJSpeech + Vocos (`scripts/download_models.py --matcha`) and Piper
en_GB southern_english_female low ("sef", `--sef`; see THIRD_PARTY_MODELS.md for its licence caveat).
Piper voices use the noise/length scales from their own .onnx.json. Each sentence is synthesised in one call
(max_num_sentences=1) and returned as a PCM16 WAV. Licences: THIRD_PARTY_MODELS.md.
"""
from __future__ import annotations

import io
import re
import os
import threading
import time
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
M = HERE / "models" / "tts"
TTS_THREADS = int(os.environ.get("TTS_THREADS", "4"))

VOICES: dict[str, dict] = {
    "ljspeech": {"display": "LJSpeech (Piper, US)", "kind": "vits",
                 "dir": M / "vits-piper-en_US-ljspeech-medium", "model": "en_US-ljspeech-medium.onnx"},
    "sef": {"display": "Southern English female (Piper, UK)", "kind": "vits",
            "dir": M / "vits-piper-en_GB-southern_english_female-low", "model": "en_GB-southern_english_female-low.onnx"},
    "matcha": {"display": "Matcha LJSpeech (Matcha + Vocos, US)", "kind": "matcha",
               "dir": M / "matcha-icefall-en_US-ljspeech", "model": "model-steps-3.onnx",
               "vocoder": M / "vocos-22khz-univ.onnx"},
}
DEFAULT_VOICE = os.environ.get("TTS_VOICE", "ljspeech")
if DEFAULT_VOICE not in VOICES:
    DEFAULT_VOICE = "ljspeech"


def _piper_scales(d: Path, model: str) -> dict:
    """noise_scale / noise_scale_w / length_scale from the Piper voice's own config (upstream defaults)."""
    try:
        import json

        inf = json.loads((d / (model + ".json")).read_text()).get("inference", {})
        return {"noise_scale": float(inf.get("noise_scale", 0.667)), "noise_scale_w": float(inf.get("noise_w", 0.8)),
                "length_scale": float(inf.get("length_scale", 1.0))}
    except (OSError, ValueError):
        return {}


TARGET_RMS_DBFS = -20.0   # speech level all voices are calibrated to (gated RMS)
LIMIT_KNEE = 0.60         # soft limiter: linear below this, tanh above
LIMIT_CEIL = 0.89         # never exceeds -1 dBFS
TAIL_MIN_MS = 120.0       # minimum silence at the end of every sentence, so sentences never run together
CALIBRATION = ["Hello, it is nice to talk with you today.",
               "Your daughter's school play is on Friday at six in the evening.",
               "Remember to drink some water and take a short walk after lunch."]


def _gated_rms_db(a: np.ndarray, sr: int) -> float:
    """RMS over 20 ms frames within 30 dB of the loudest frame (speech only, pauses excluded)."""
    n = max(1, int(sr * 0.02)); k = len(a) // n
    if k == 0:
        return -120.0
    fr = 10 * np.log10(np.mean(a[: k * n].reshape(k, n) ** 2, axis=1) + 1e-12)
    voiced = fr[fr > fr.max() - 30]
    return float(10 * np.log10(np.mean(10 ** (voiced / 10)) + 1e-12))


def _soft_limit(a: np.ndarray) -> np.ndarray:
    x = np.abs(a); over = x > LIMIT_KNEE
    if not over.any():
        return a
    y = a.copy(); span = LIMIT_CEIL - LIMIT_KNEE
    y[over] = np.sign(a[over]) * (LIMIT_KNEE + span * np.tanh((x[over] - LIMIT_KNEE) / span))
    return y


def _shape_edges(a: np.ndarray, sr: int, lead_ms: float = 30.0, fade_ms: float = 20.0) -> np.ndarray:
    """Short silence lead-in, fade-in/out so joins never click, and at least TAIL_MIN_MS of tail silence."""
    n = int(sr * fade_ms / 1000)
    if len(a) > 2 * n and n > 0:
        r = np.linspace(0.0, 1.0, n, dtype=np.float32)
        a = a.copy(); a[:n] *= r; a[-n:] *= r[::-1]
    thr = max(float(np.abs(a).max()) if len(a) else 0.0, 1e-6) * 10 ** (-60 / 20)  # -60 dB: faint vocoder residue counts as silence
    loud = np.nonzero(np.abs(a) > thr)[0]
    trail_ms = (len(a) - 1 - loud[-1]) / sr * 1000 if len(loud) else 0.0
    pad = int(sr * max(0.0, TAIL_MIN_MS - trail_ms) / 1000)
    return np.concatenate([np.zeros(int(sr * lead_ms / 1000), np.float32), a, np.zeros(pad, np.float32)]).astype(np.float32)


class OpenTts:
    def __init__(self, voice: str = DEFAULT_VOICE, threads: int = TTS_THREADS) -> None:
        import sherpa_onnx as so

        if voice not in VOICES:
            raise ValueError(f"unknown voice {voice!r}; choose from {list(VOICES)}")
        v = VOICES[voice]; d = v["dir"]
        if v["kind"] == "vits":
            model = so.OfflineTtsModelConfig(
                vits=so.OfflineTtsVitsModelConfig(model=str(d / v["model"]), tokens=str(d / "tokens.txt"),
                                                  data_dir=str(d / "espeak-ng-data"), **_piper_scales(d, v["model"])),
                num_threads=threads, provider="cpu")
        else:
            model = so.OfflineTtsModelConfig(
                matcha=so.OfflineTtsMatchaModelConfig(acoustic_model=str(d / v["model"]), vocoder=str(v["vocoder"]),
                                                      tokens=str(d / "tokens.txt"), data_dir=str(d / "espeak-ng-data"),
                                                      noise_scale=0.667),
                num_threads=threads, provider="cpu")
        t0 = time.perf_counter()
        self.tts = so.OfflineTts(so.OfflineTtsConfig(model=model, max_num_sentences=1))
        self.voice, self.display, self.sample_rate = voice, v["display"], self.tts.sample_rate
        self._lock = threading.Lock()
        # Fixed per-voice gain (not per-sentence normalisation, which made loud sentences
        # quieter than their neighbours): calibrate the voice's speech level once.
        levels = [_gated_rms_db(np.asarray(self.tts.generate(t, sid=0, speed=1.0).samples, np.float32), self.sample_rate)
                  for t in CALIBRATION]
        self.gain = float(10 ** ((TARGET_RMS_DBFS - float(np.median(levels))) / 20))
        self.load_ms = (time.perf_counter() - t0) * 1000.0

    def synthesize(self, text: str, voice=None, speed: float = 1.0, **_ignored) -> dict:
        t0 = time.perf_counter()
        text = speakable(text)
        with self._lock:
            audio = self.tts.generate(text, sid=0, speed=float(speed))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        a = _soft_limit(np.asarray(audio.samples, np.float32) * self.gain)
        a = _shape_edges(a, audio.sample_rate)
        buf = io.BytesIO()
        sf.write(buf, a, audio.sample_rate, format="WAV", subtype="PCM_16")
        return {"wav": buf.getvalue(), "sample_rate": audio.sample_rate, "duration_ms": len(a) / audio.sample_rate * 1000.0,
                "latency_ms": latency_ms, "voice": self.voice}


# Respellings for words the voices mispronounce, applied to the spoken text only (captions keep the
# original). Add entries as {regex: respelling}.
RESPELL: dict[str, str] = {}


def speakable(text: str) -> str:
    for rx, say in RESPELL.items():
        text = re.sub(rx, say, text)
    return text


_engines: dict[str, OpenTts] = {}
_lock = threading.Lock()


def get(voice: str | None = None) -> OpenTts:
    v = voice if voice in VOICES else DEFAULT_VOICE
    with _lock:
        if v not in _engines:
            _engines[v] = OpenTts(v)
        return _engines[v]


def available(voice: str = DEFAULT_VOICE) -> bool:
    v = VOICES.get(voice)
    return bool(v) and (v["dir"] / v["model"]).exists()
