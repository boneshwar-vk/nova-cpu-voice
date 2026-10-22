#!/usr/bin/env python3
"""Download every model the pipeline needs from its original public upstream into ./models.

Idempotent: a file whose size and sha256 already match is skipped; an archive is skipped when its
extracted directory carries a `.complete` marker (delete models/<dir> to fetch it again). Standard library only.

    python scripts/download_models.py             # core models (about 2.9 GB, most of it Gemma)
    python scripts/download_models.py --matcha    # also the optional Matcha LJSpeech voice + Vocos vocoder
    python scripts/download_models.py --sef       # also the optional Piper "southern English female" voice
                                                  # (see THIRD_PARTY_MODELS.md for its licence caveat)
    GEMMA_PATH=/path/to/gemma-4-E2B-it.litertlm python scripts/download_models.py
                                                  # reuse a local Gemma file (checked, then symlinked)

Licences and sources: THIRD_PARTY_MODELS.md.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
HF = "https://huggingface.co"

# kind "file": url -> models/<dest>; kind "tar": .tar.bz2 extracted into models/<parent of dest>, creating <dest>.
CORE = [
    {"name": "Gemma 4 E2B it (LiteRT-LM, int4)", "kind": "file", "dest": "gemma-4-E2B-it.litertlm",
     "url": f"{HF}/litert-community/gemma-4-E2B-it-litert-lm/resolve/b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1/gemma-4-E2B-it.litertlm",
     "size": 2588147712, "sha256": "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c"},
    {"name": "Silero VAD v6.2", "kind": "file", "dest": "silero_vad.onnx",
     "url": "https://raw.githubusercontent.com/snakers4/silero-vad/be95df9152c0d7618fa1edfeb296fc3dae32376f/src/silero_vad/data/silero_vad.onnx",
     "size": 2327524, "sha256": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"},
    {"name": "GTCRN speech enhancement", "kind": "file", "dest": "gtcrn_simple.onnx",
     "url": f"{SHERPA}/speech-enhancement-models/gtcrn_simple.onnx",
     "size": 535638, "sha256": "e77603ac0c23dac3227dd2d7135b3a585cbee2679048aecfa886657d3ae1b534"},
    {"name": "Smart Turn v3.2 (CPU)", "kind": "file", "dest": "smart-turn-v3.2-cpu.onnx",
     "url": f"{HF}/pipecat-ai/smart-turn-v3/resolve/f766f81d3cfdf7737ac64aad813d91bbfd56bf93/smart-turn-v3.2-cpu.onnx",
     "size": 8679182, "sha256": "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"},
    {"name": "Moonshine v2 base en (int8)", "kind": "tar", "dest": "sherpa-onnx-moonshine-base-en-quantized-2026-02-27",
     "url": f"{SHERPA}/asr-models/sherpa-onnx-moonshine-base-en-quantized-2026-02-27.tar.bz2",
     "size": 111266225, "sha256": "43232c1d13013d37317163baec3135bd771a186a4356f28c889bab453bb0e891"},
    {"name": "Moonshine v2 tiny en (int8, live captions)", "kind": "tar", "dest": "sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27",
     "url": f"{SHERPA}/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2",
     "size": 29858559, "sha256": "9ec31b342d8fa3240c3b81b8f82e1cf7e3ac467c93ca5a999b741d5887164f8d"},
    {"name": "Piper en_US ljspeech medium (default voice)", "kind": "tar", "dest": "tts/vits-piper-en_US-ljspeech-medium",
     "url": f"{SHERPA}/tts-models/vits-piper-en_US-ljspeech-medium.tar.bz2",
     "size": 67169893, "sha256": "3dfb4b759d8be032a4903a9538d128b0fda2a06ab1de6cbc2d93a97e2dd83dba"},
    {"name": "all-MiniLM-L6-v2 int8 (memory/library embeddings)", "kind": "file", "dest": "embed/minilm/model_quantized.onnx",
     "url": f"{HF}/Xenova/all-MiniLM-L6-v2/resolve/751bff37182d3f1213fa05d7196b954e230abad9/onnx/model_quantized.onnx",
     "size": 22972370, "sha256": "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1"},
    {"name": "all-MiniLM-L6-v2 tokenizer", "kind": "file", "dest": "embed/minilm/tokenizer.json",
     "url": f"{HF}/Xenova/all-MiniLM-L6-v2/resolve/751bff37182d3f1213fa05d7196b954e230abad9/tokenizer.json",
     "size": 711661, "sha256": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"},
]
SEF = [
    {"name": "Piper en_GB southern_english_female low (optional voice)", "kind": "tar", "dest": "tts/vits-piper-en_GB-southern_english_female-low",
     "url": f"{SHERPA}/tts-models/vits-piper-en_GB-southern_english_female-low.tar.bz2",
     "size": 67073626, "sha256": "448e7723c66531c62004864382044b29703a2616699833ec7fe1205de9f549ce"},
]
MATCHA = [
    {"name": "Matcha LJSpeech acoustic model", "kind": "tar", "dest": "tts/matcha-icefall-en_US-ljspeech",
     "url": f"{SHERPA}/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2",
     "size": 76713830, "sha256": "82f599b04e3afb0003df0d493d026d5462e9f25a600d42d329775d18ffbb82df"},
    {"name": "Vocos 22 kHz vocoder", "kind": "file", "dest": "tts/vocos-22khz-univ.onnx",
     "url": f"{SHERPA}/vocoder-models/vocos-22khz-univ.onnx",
     "size": 53884024, "sha256": "0574a135aa1db2de6e181050db2ec528496cacd4a4701fc5d7faf9f9804c0081"},
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, out: Path, size: int) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "nova-cpu-voice-downloader"})
    with urllib.request.urlopen(req, timeout=60) as r, out.open("wb") as f:
        done, step = 0, max(size // 20, 1 << 20)
        nxt = step
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
            done += len(block)
            if done >= nxt:
                print(f"    {done / 1e6:8.1f} / {size / 1e6:.1f} MB", flush=True)
                nxt += step


def verify(path: Path, m: dict) -> None:
    got = path.stat().st_size
    if got != m["size"]:
        raise SystemExit(f"size mismatch for {m['name']}: {got} != {m['size']}")
    digest = sha256(path)
    if digest != m["sha256"]:
        raise SystemExit(f"sha256 mismatch for {m['name']}: {digest} != {m['sha256']}")


def safe_extract(archive: Path, into: Path) -> None:
    with tarfile.open(archive, "r:bz2") as t:
        for member in t.getmembers():
            target = (into / member.name).resolve()
            if not str(target).startswith(str(into.resolve()) + os.sep) or member.issym() or member.islnk():
                raise SystemExit(f"refusing unsafe archive member {member.name!r} in {archive.name}")
        t.extractall(into)


def get_file(m: dict) -> None:
    dest = MODELS / m["dest"]
    if dest.exists() and dest.stat().st_size == m["size"] and sha256(dest) == m["sha256"]:
        print(f"ok    {m['name']}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetch {m['name']}\n      {m['url']}")
    tmp = dest.with_name(dest.name + ".part")
    fetch(m["url"], tmp, m["size"])
    verify(tmp, m)
    tmp.replace(dest)


def get_tar(m: dict) -> None:
    dest = MODELS / m["dest"]
    if (dest / ".complete").exists():
        print(f"ok    {m['name']}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"fetch {m['name']}\n      {m['url']}")
    with tempfile.TemporaryDirectory(dir=dest.parent) as td:
        arc = Path(td) / "archive.tar.bz2"
        fetch(m["url"], arc, m["size"])
        verify(arc, m)
        ex = Path(td) / "x"
        ex.mkdir()
        safe_extract(arc, ex)
        top = [p for p in ex.iterdir()]
        if len(top) != 1 or not top[0].is_dir():
            raise SystemExit(f"unexpected archive layout for {m['name']}")
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(top[0]), dest)
    (dest / ".complete").write_text(m["sha256"] + "\n")


def link_gemma(m: dict, src: Path) -> None:
    dest = MODELS / m["dest"]
    if dest.is_symlink() and dest.resolve() == src.resolve():
        print(f"ok    {m['name']} (symlink to GEMMA_PATH)")
        return
    print(f"check {m['name']} at GEMMA_PATH")
    verify(src, m)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    dest.symlink_to(src.resolve())


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matcha", action="store_true", help="also download the optional Matcha LJSpeech voice")
    ap.add_argument("--sef", action="store_true", help="also download the optional Piper southern English female voice")
    ap.add_argument("--skip-gemma", action="store_true", help="do not download the language model")
    a = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    gemma_path = os.environ.get("GEMMA_PATH")
    for m in CORE + (SEF if a.sef else []) + (MATCHA if a.matcha else []):
        if m["dest"].endswith(".litertlm"):
            if a.skip_gemma:
                continue
            if gemma_path:
                link_gemma(m, Path(gemma_path).expanduser())
                continue
        (get_tar if m["kind"] == "tar" else get_file)(m)
    print(f"all models ready in {MODELS}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
