"""Silero VAD (v5) + a streaming two-stage turn detector, torch-free via
onnxruntime, plus the Smart Turn v3 semantic endpointer.

Feeds PCM16 16 kHz mono frames and emits staged silence events plus a per-push
`any_speech` flag (used for barge-in while the bot is talking):

  endpoint_soft  — soft_silence_ms of silence: cheap moment to ask the semantic
                   classifier "is this phrase complete?" and commit early
  endpoint       — end_silence_ms (the plain-silence commit point): semantic re-check,
                   or the commit itself when no classifier is available
  endpoint_hard  — max_silence_ms: unconditional commit (nobody talks this slow)

The caller decides when the turn actually commits and then calls
`turn_committed()`; the detector never ends a turn on its own anymore.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

import whisper_mel

_MODEL = Path(__file__).resolve().parent / "models" / "silero_vad.onnx"
_SMART_TURN = Path(__file__).resolve().parent / "models" / "smart-turn-v3.2-cpu.onnx"
_GTCRN = Path(__file__).resolve().parent / "models" / "gtcrn_simple.onnx"
# GTCRN speech enhancement on the VAD branch ONLY (STT and Smart Turn get raw
# audio): ~1.5 ms per 32 ms frame on a laptop CPU. VAD_DENOISE=0 disables.
VAD_DENOISE = os.environ.get("VAD_DENOISE", "1") == "1" and _GTCRN.exists()
SR = 16000
FRAME = 512  # samples per VAD frame at 16 kHz (~32 ms)
_CONTEXT = 64  # v5 prepends 64 samples of prior context to each frame
_FRAME_MS = FRAME / SR * 1000.0

# Adaptive background gate against "listening forever": Silero scores babble
# and TV as speech (~97 % / ~63 % of frames), so the silence counter never
# starts while people talk in the background. While no user turn is open we
# keep the level (dBFS) of recent frames; when most of them are speech-like
# (babble / TV / Nova's own echo) a frame counts as user speech only if it is
# also NOISE_GATE_DB above that background's NOISE_GATE_PCT percentile. In a
# quiet room the gate stays off, so clean-speech behaviour is unchanged.
NOISE_GATE = os.environ.get("NOISE_GATE", "1") == "1"
NOISE_GATE_DB = float(os.environ.get("NOISE_GATE_DB", "3"))
NOISE_GATE_PCT = float(os.environ.get("NOISE_GATE_PCT", "80"))
NOISE_GATE_WIN_S = float(os.environ.get("NOISE_GATE_WIN_S", "4.0"))
NOISE_GATE_MIN_SPEECHY = float(os.environ.get("NOISE_GATE_MIN_SPEECHY", "0.5"))
# With a speech-like background, frames this far below the user's own level
# (p90 of their speech frames this turn) are background, not the user.
NOISE_USER_DROP_DB = float(os.environ.get("NOISE_USER_DROP_DB", "10"))
# Smart Turn inference threads (bench/bench_stages.py measures the predict time).
SMART_TURN_THREADS = int(os.environ.get("SMART_TURN_THREADS", "4"))


class SileroVad:
    def __init__(self) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(_MODEL), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SR, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(_CONTEXT, dtype=np.float32)

    def prob(self, frame: np.ndarray) -> float:
        # v5: feed [context(64) + frame(512)] = 576 samples; carry last 64 fwd.
        x = np.concatenate([self._context, frame.astype(np.float32)])
        self._context = x[-_CONTEXT:]
        out, self._state = self.sess.run(
            ["output", "stateN"],
            {"input": x.reshape(1, -1), "state": self._state, "sr": self._sr},
        )
        return float(out[0, 0])


class GtcrnDenoiser:
    """sherpa-onnx streaming GTCRN (48 K params). Feed 512-sample frames, get
    denoised 512-sample frames back (the model's own hop is 256 samples, so
    output may lag the input by up to one frame — the VAD stage marks are
    frame-counted, so that only shifts them by ≤32 ms)."""

    def __init__(self) -> None:
        import sherpa_onnx  # lazy: heavy import

        cfg = sherpa_onnx.OnlineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=str(_GTCRN)),
                num_threads=1, provider="cpu"))
        self._d = sherpa_onnx.OnlineSpeechDenoiser(cfg)
        self._pending = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._d.reset()
        self._pending = np.zeros(0, dtype=np.float32)

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        """Return zero or more complete denoised 512-sample frames."""
        out = self._d.run(frame.astype(np.float32), SR)
        self._pending = np.concatenate([self._pending, np.asarray(out.samples, dtype=np.float32)])
        frames = []
        while len(self._pending) >= FRAME:
            frames.append(self._pending[:FRAME])
            self._pending = self._pending[FRAME:]
        return frames


class TurnDetector:
    """Stateful two-stage endpointer. `push(pcm16_bytes)` returns
    (events, any_speech). Commit is the caller's decision — call
    `turn_committed()` when a stage's verdict (or the hard cap) ends the turn.
    With `denoise` (default VAD_DENOISE) the frames are GTCRN-denoised before
    Silero; the caller's utterance buffer keeps the raw audio.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        min_speech_ms: float = 200,
        soft_silence_ms: float = float(os.environ.get("TURN_SOFT_MS", "200")),
        end_silence_ms: float = float(os.environ.get("TURN_CHECK_MS", "700")),
        max_silence_ms: float = float(os.environ.get("TURN_HARD_MS", "1500")),
        denoise: bool | None = None,
    ) -> None:
        self.vad = SileroVad()
        self.denoiser = GtcrnDenoiser() if (VAD_DENOISE if denoise is None else denoise) else None
        self.threshold = threshold
        self.min_speech_frames = max(1, int(min_speech_ms / _FRAME_MS))
        self.soft_silence_frames = max(1, int(soft_silence_ms / _FRAME_MS))
        self.end_silence_frames = max(1, int(end_silence_ms / _FRAME_MS))
        self.max_silence_frames = max(1, int(max_silence_ms / _FRAME_MS))
        self.reset()

    def reset(self) -> None:
        self.vad.reset()
        if self.denoiser is not None:
            self.denoiser.reset()
        self._buf = np.zeros(0, dtype=np.float32)
        self.in_speech = False
        self._last_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._soft_fired = False
        self._check_fired = False
        from collections import deque

        self._bg: deque = deque(maxlen=max(8, int(NOISE_GATE_WIN_S * 1000 / _FRAME_MS)))
        self.gate_db: float | None = None  # current gate level (None = off)
        self._onset_known = True
        self._gate_on = False
        self._user_lv: deque = deque(maxlen=200)

    def _gate(self) -> float | None:
        """Background gate level in dBFS, or None when the background is quiet."""
        if not NOISE_GATE or len(self._bg) < self._bg.maxlen // 2:
            return None
        lv = np.fromiter((x[0] for x in self._bg), dtype=np.float32)
        speechy = sum(1 for x in self._bg if x[1]) / len(self._bg)
        # Hysteresis: TV and babble have pauses; don't drop the gate in one.
        on = speechy >= (NOISE_GATE_MIN_SPEECHY / 2 if self._gate_on else NOISE_GATE_MIN_SPEECHY)
        self._gate_on = on
        if not on:
            return None
        return float(np.percentile(lv, NOISE_GATE_PCT)) + NOISE_GATE_DB

    def turn_committed(self) -> None:
        """The caller ended the turn — arm for the next utterance/barge-in."""
        self.in_speech = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._soft_fired = False
        self._check_fired = False

    def push(self, pcm16: bytes) -> tuple[list[str], bool]:
        samples = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self._buf = np.concatenate([self._buf, samples])
        events: list[str] = []
        any_speech = False
        while len(self._buf) >= FRAME:
            raw = self._buf[:FRAME]
            self._buf = self._buf[FRAME:]
            vad_frames = self.denoiser.push(raw) if self.denoiser is not None else [raw]
            for frame in vad_frames:
                self._step(frame, events)
                if self._last_speech:
                    any_speech = True
        return events, any_speech

    def _step(self, frame: np.ndarray, events: list[str]) -> None:
        if True:
            vad_speech = self.vad.prob(frame) >= self.threshold
            level = 10.0 * np.log10(float(np.mean(frame * frame)) + 1e-10)
            if not self.in_speech:
                # Background only adapts while no user turn is open (frozen
                # during the user's speech so their voice never raises it).
                if self._speech_frames == 0:
                    self.gate_db = self._gate()
                self._bg.append((level, vad_speech))
            elif not self._onset_known:
                # The "speech" began before we had learned the room (e.g. the
                # mic opened into background talk): keep learning; once the
                # gate says it was only background, drop it.
                self._bg.append((level, vad_speech))
                if len(self._bg) >= self._bg.maxlen // 2:
                    self._onset_known = True
                    g = self._gate()
                    if g is not None:
                        recent = [x[0] for x in list(self._bg)[-self.min_speech_frames * 2:]]
                        if max(recent) < g:
                            self.gate_db = g
                            self.turn_committed()
                            self._last_speech = False
                            events.append("noise_reset")
                            return
            speech = vad_speech and (self.gate_db is None or level >= self.gate_db)
            if self.in_speech and self.gate_db is not None:
                # Offset side: once we have heard the user, background voices
                # are the speech-like frames well below the user's own level.
                if speech:
                    self._user_lv.append(level)
                if len(self._user_lv) >= 12:
                    user = float(np.percentile(np.fromiter(self._user_lv, np.float32), 90))
                    if level < user - NOISE_USER_DROP_DB:
                        speech = False
            self._last_speech = speech
            if not self.in_speech:
                if speech:
                    self._speech_frames += 1
                    if self._speech_frames >= self.min_speech_frames:
                        self.in_speech = True
                        self._user_lv.clear()
                        self._onset_known = (self.gate_db is not None
                                             or len(self._bg) >= self._bg.maxlen // 2)
                        self._silence_frames = 0
                        self._soft_fired = False
                        self._check_fired = False
                        events.append("speech_start")
                else:
                    self._speech_frames = 0
            else:
                if speech:
                    self._silence_frames = 0
                    self._soft_fired = False
                    self._check_fired = False
                else:
                    self._silence_frames += 1
                    if (not self._soft_fired
                            and self._silence_frames >= self.soft_silence_frames):
                        self._soft_fired = True
                        events.append("endpoint_soft")
                    if (not self._check_fired
                            and self._silence_frames >= self.end_silence_frames):
                        self._check_fired = True
                        events.append("endpoint")
                    if self._silence_frames >= self.max_silence_frames:
                        self.turn_committed()
                        events.append("endpoint_hard")


class SemanticEndpointer:
    """Smart Turn v3 phrase-completion classifier (pipecat-ai/smart-turn-v3,
    BSD-2-Clause, 23 languages incl. en+hi). Audio-native: give it the raw
    utterance (last <=8 s), it returns P(the speaker is done talking) — no
    transcript needed, so it runs in parallel with eager STT.

    Preprocessing mirrors pipecat-ai/smart-turn inference.py: last 8 s (zero-padded at
    the START), zero-mean/unit-var normalize, Whisper log-mel (80 x 800) via
    whisper_mel.py (numpy, no torch/transformers dependency).
    """

    # Staged thresholds for the two-stage endpointer:
    # - soft (~200 ms of silence): commit only when VERY confident the phrase is
    #   complete — an early false commit is the worst failure mode.
    # - check (~700 ms): commit unless the verdict is clearly incomplete; only
    #   then may the wait extend to the hard cap (TURN_HARD_MS).
    THRESHOLD_SOFT = 0.9
    THRESHOLD_CHECK = 0.3

    def __init__(self) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, SMART_TURN_THREADS)
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(_SMART_TURN), sess_options=opts, providers=["CPUExecutionProvider"]
        )

    @staticmethod
    def available() -> bool:
        return _SMART_TURN.exists()

    def predict_pcm16(self, pcm16: bytes) -> float:
        audio = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        return self.predict(audio)

    def predict(self, audio: np.ndarray) -> float:
        n = 8 * SR
        if len(audio) > n:
            audio = audio[-n:]
        # Normalize over the REAL samples ONLY, then pad — the reference's
        # zero_mean_unit_var_norm excludes padding from the statistics.
        # Normalizing the padded buffer scaled every real sample by 1/sqrt(fill)
        # (a length-dependent brightness shift) and cost ~11 points of accuracy
        # on the labeled set (67.9% -> 78.6% at 0.5; false-commit@0.9
        # 21.4% -> 14.3%). Padding stays on the LEFT: the reference right-pads,
        # but left-padding measured strictly better (en 81% vs 62%) — keep it.
        audio = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
        if len(audio) < n:
            audio = np.pad(audio, (n - len(audio), 0))
        feats = whisper_mel.log_mel(audio.astype(np.float32))
        out = self.sess.run(
            ["logits"], {"input_features": feats[np.newaxis, :, :].astype(np.float32)}
        )[0]
        val = float(out[0, 0])
        # v3.0 exported a sigmoid; the head name says "logits" — normalize either
        # way so callers always get a probability.
        return val if 0.0 <= val <= 1.0 else 1.0 / (1.0 + np.exp(-val))
