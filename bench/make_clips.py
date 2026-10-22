"""Generate the benchmark/test clips with the shipped TTS voice (no recordings of real people).

Writes 16 kHz mono PCM16 WAVs to bench/clips/ and the clip list bench/clips.json used by
bench/e2e_bench.py and bench/bench_stages.py. Trailing silence is trimmed so that "end of the clip"
is the end of speech; e2e_bench.py then streams `silence_s` of silence (longer than the 1.5 s hard cap
of the turn detector, as a live microphone would) to trigger the endpointer.

    .venv/bin/python bench/make_clips.py [--voice ljspeech]
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

CLIPS = [
    {"id": "question", "kind": "complete", "silence_s": 2.0, "text": "What is a black hole, in simple words?"},
    {"id": "moon", "kind": "complete", "silence_s": 2.0, "text": "How far away is the moon?"},
    {"id": "roman", "kind": "complete", "silence_s": 2.0, "text": "Tell me something interesting about the Roman empire."},
    {"id": "gravity", "kind": "complete", "silence_s": 2.0, "text": "Can you explain gravity again?"},
    {"id": "incomplete", "kind": "incomplete", "silence_s": 3.0, "text": "So I was thinking about the"},
]
SR = 16000


def _trim(a: np.ndarray, sr: int, keep_ms: float = 20.0) -> np.ndarray:
    thr = max(float(np.abs(a).max()), 1e-6) * 10 ** (-40 / 20)
    loud = np.nonzero(np.abs(a) > thr)[0]
    if not len(loud):
        return a
    hi = min(len(a), loud[-1] + int(sr * keep_ms / 1000))
    return a[:hi]  # only the tail is trimmed: soft onsets ("h") stay intact


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default="ljspeech")
    a = ap.parse_args()
    import open_tts

    eng = open_tts.get(a.voice)
    out_dir = HERE / "clips"
    out_dir.mkdir(exist_ok=True)
    listing = []
    for c in CLIPS:
        r = eng.synthesize(c["text"])
        audio, sr = sf.read(io.BytesIO(r["wav"]), dtype="float32")
        if sr != SR:
            n = int(round(len(audio) * SR / sr))
            audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio).astype(np.float32)
        audio = _trim(audio, SR)
        path = out_dir / f"{c['id']}.wav"
        sf.write(str(path), audio, SR, subtype="PCM_16")
        listing.append({**c, "path": f"bench/clips/{path.name}", "voice": a.voice, "seconds": round(len(audio) / SR, 2)})
        print(f"{path.relative_to(ROOT)}  {len(audio) / SR:.2f} s  {c['text']!r}")
    (HERE / "clips.json").write_text(json.dumps(listing, indent=1) + "\n")
    print(f"wrote {len(listing)} clips and bench/clips.json")


if __name__ == "__main__":
    main()
