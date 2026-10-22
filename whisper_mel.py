"""Whisper-style 80-bin log-mel spectrogram in plain numpy (the input format Smart Turn v3 expects).

16 kHz audio, 25 ms Hann window (n_fft 400), 10 ms hop, Slaney mel filterbank, log10 with an 8-decade
dynamic range clamp, scaled by (x + 4) / 4 - the same recipe as OpenAI Whisper's log_mel_spectrogram.
"""
from __future__ import annotations

import numpy as np

SR, N_FFT, HOP, N_MELS = 16000, 400, 160, 80


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    f = np.asarray(f, dtype=np.float64)
    lin = f / (200.0 / 3)
    log = 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0)
    return np.where(f >= 1000.0, log, lin)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    lin = m * (200.0 / 3)
    log = 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0))
    return np.where(m >= 15.0, log, lin)


def mel_filters(sr: int = SR, n_fft: int = N_FFT, n_mels: int = N_MELS) -> np.ndarray:
    """Slaney-normalised triangular filters, shape (n_mels, n_fft // 2 + 1)."""
    fft_hz = np.fft.rfftfreq(n_fft, 1.0 / sr)
    pts = _mel_to_hz(np.linspace(0.0, _hz_to_mel(sr / 2.0), n_mels + 2))
    lower = (fft_hz[None, :] - pts[:-2, None]) / np.diff(pts)[:-1, None]
    upper = (pts[2:, None] - fft_hz[None, :]) / np.diff(pts)[1:, None]
    w = np.maximum(0.0, np.minimum(lower, upper))
    w *= (2.0 / (pts[2:] - pts[:-2]))[:, None]
    return w.astype(np.float32)


_FILTERS = mel_filters()
_WINDOW = np.hanning(N_FFT + 1)[:-1].astype(np.float32)


def log_mel(audio: np.ndarray) -> np.ndarray:
    """(n_samples,) float32 at 16 kHz -> (80, n_samples // 160) float32."""
    x = np.pad(np.asarray(audio, dtype=np.float32), N_FFT // 2, mode="reflect")
    n = 1 + (len(x) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(x, (n, N_FFT), (HOP * x.strides[0], x.strides[0]))
    spec = np.fft.rfft(frames * _WINDOW, axis=-1).astype(np.complex64)
    power = (np.abs(spec[:-1]) ** 2).T  # drop the last frame, as Whisper does
    mel = _FILTERS @ power
    logm = np.log10(np.maximum(mel, 1e-10))
    logm = np.maximum(logm, logm.max() - 8.0)
    return ((logm + 4.0) / 4.0).astype(np.float32)
