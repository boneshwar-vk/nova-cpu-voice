"""Per-stage CPU benchmark with the shipped configuration (no server needed).

Stop the voice server first: only one Gemma process should be loaded at a time.

  STT   Moonshine v2 base (sherpa-onnx): decode time of each bench clip (end of speech -> transcript)
  LLM   Gemma 4 E2B int4 on LiteRT-LM CPU through llm_session.py (the server's code path, same system
        prompt with tool declarations): time to first token (prefill of the new user turn + first token)
        and decode speed
  TTS   default voice (ljspeech): time to first audio of a reply (= synthesis of its first sentence, since
        each sentence is synthesised in one call) and real-time factor
  Turn  Smart Turn v3.2 predict time on an 8 s window; Silero VAD + GTCRN time per 32 ms frame

    .venv/bin/python bench/bench_stages.py --runs 5 --tag m2
Writes bench/results/stages_<tag>.json and prints a markdown table.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

PROMPTS = [
    "What is a black hole, in simple words?",
    "How far away is the moon?",
    "Tell me something interesting about the Roman empire.",
    "Can you explain gravity again?",
    "Give me three tips for sleeping better.",
]
REPLY = [  # typical spoken reply sentences
    "A black hole is a place in space where gravity is extremely strong.",
    "Nothing, not even light, can escape from it.",
    "The moon is about three hundred and eighty four thousand kilometres away.",
    "That is roughly thirty times the width of the Earth.",
    "The Roman empire built roads that are still used today.",
    "Gravity is the pull that every mass has on every other mass.",
]


def pct(xs, q):
    return round(float(np.percentile(xs, q)), 1) if xs else None


def stat(xs) -> dict:
    return {"n": len(xs), "p50": pct(xs, 50), "p90": pct(xs, 90), "min": pct(xs, 0), "max": pct(xs, 100)}


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2**20 if sys.platform == "darwin" else r / 1024, 1)


def machine() -> dict:
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            return ""
    info = {"os": f"{platform.system()} {platform.release()} {platform.machine()}", "python": platform.python_version(),
            "logical_cpus": os.cpu_count()}
    if sys.platform == "darwin":
        info.update(cpu=sh("sysctl", "-n", "machdep.cpu.brand_string"),
                    perf_cores=sh("sysctl", "-n", "hw.perflevel0.physicalcpu"),
                    eff_cores=sh("sysctl", "-n", "hw.perflevel1.physicalcpu"),
                    ram_gb=round(int(sh("sysctl", "-n", "hw.memsize") or 0) / 2**30, 1),
                    swap_used=sh("sysctl", "-n", "vm.swapusage"),
                    memory_free_pct=(re.findall(r"free percentage: (\d+)%", sh("memory_pressure")) or [None])[0])
    else:
        cpu = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                    if ln.startswith("model name")), "") if Path("/proc/cpuinfo").exists() else ""
        mem = Path("/proc/meminfo").read_text().split("\n")[0] if Path("/proc/meminfo").exists() else ""
        info.update(cpu=cpu, ram_gb=round(int(re.findall(r"\d+", mem)[0]) / 2**20, 1) if mem else None)
    info["load_avg"] = [round(x, 2) for x in os.getloadavg()]
    return info


def bench_stt(clips: list[dict], runs: int) -> dict:
    import soundfile as sf

    import offline_engine
    offline_engine.warmup()
    out, words, errs = [], 0, 0
    per_clip = {}
    for c in clips:
        a, sr = sf.read(str(ROOT / c["path"]), dtype="int16")
        pcm = a.tobytes()
        offline_engine.stt_pcm(pcm)  # shape warm-up
        ts = []
        for _ in range(runs):
            t = time.perf_counter()
            text = offline_engine.stt_pcm(pcm)
            ts.append((time.perf_counter() - t) * 1000)
        out += ts
        ref = re.findall(r"[a-z']+", c["text"].lower())
        hyp = re.findall(r"[a-z']+", text.lower())
        e = _edits(ref, hyp)
        words += len(ref)
        errs += e
        per_clip[c["id"]] = {"audio_s": round(len(a) / sr, 2), "decode_ms": stat(ts), "samples_ms": [round(x, 2) for x in ts],
                             "reference": c["text"], "transcript": text, "word_errors": e, "words": len(ref)}
    return {"model": offline_engine.STT_BACKEND, "decode_ms": stat(out), "wer": round(errs / max(words, 1), 3),
            "samples_ms": [round(x, 2) for x in out], "per_clip": per_clip}


def _edits(r: list[str], h: list[str]) -> int:
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)]


def bench_llm(runs: int, max_tokens: int) -> dict:
    import llm_session
    import personas

    system = personas.runtime_system(personas.ORDER[0])
    t = time.perf_counter()
    s = llm_session.get(system)
    load_ms = (time.perf_counter() - t) * 1000
    sys_tokens = s.g.eng.ntok(system)
    for _ in s.stream([], "Hi there.", 16):  # warm-up turn (not measured)
        pass
    ttft, tps, prefill_tok, decode_tok, rows = [], [], [], [], []
    for r in range(runs):
        for p in PROMPTS:
            reply = "".join(s.stream([], p, max_tokens))  # empty history: rewinds to the system prompt, prefills the user turn
            st = dict(s.last)
            ntok = s.g.eng.ntok(reply)
            rows.append({"prompt": p, "run": r + 1, "ttft_ms": st.get("ttft_ms"), "prefill_tokens": st.get("prefill_tokens"),
                         "prefill_ms": st.get("prefill_ms"), "decode_chunks": st.get("decode_tokens"), "reply_tokens": ntok,
                         "decode_tps": st.get("decode_tps"), "mode": st.get("mode"), "reply": reply})
            if st.get("ttft_ms") is not None:
                ttft.append(st["ttft_ms"])
            if st.get("decode_tps"):
                tps.append(st["decode_tps"])
            prefill_tok.append(st.get("prefill_tokens") or 0)
            decode_tok.append(ntok)
    return {"model": Path(llm_session.MODEL).name, "runtime": "LiteRT-LM 0.17.1 raw session, CPU",
            "threads": llm_session.THREADS, "kv_tokens": llm_session.KV_TOKENS, "max_tokens": max_tokens,
            "load_ms": round(load_ms), "system_prompt_tokens": sys_tokens, "system_prefill_ms": round(s.g.sys_ms),
            "ttft_ms": stat(ttft), "decode_tps": stat(tps), "user_turn_prefill_tokens": stat(prefill_tok),
            "reply_tokens": stat(decode_tok), "turns": rows}


def bench_tts(runs: int, voice: str) -> dict:
    import io

    import soundfile as sf

    import open_tts
    eng = open_tts.get(voice)
    first, lat, audio_s = [], [], 0.0
    synth_s = 0.0
    samples = []
    for _ in range(runs):
        for i, text in enumerate(REPLY):
            r = eng.synthesize(text)
            a, sr = sf.read(io.BytesIO(r["wav"]))
            lat.append(r["latency_ms"])
            samples.append({"text": text, "first_of_reply": i % 2 == 0, "synth_ms": round(r["latency_ms"], 2),
                            "audio_s": round(len(a) / sr, 4), "peak": round(float(np.abs(a).max()), 4)})
            if i % 2 == 0:  # each pair of sentences = one reply; its first sentence is the time to first audio
                first.append(r["latency_ms"])
            synth_s += r["latency_ms"] / 1000
            audio_s += len(a) / sr
    return {"voice": voice, "display": open_tts.VOICES[voice]["display"], "sample_rate": eng.sample_rate,
            "threads": open_tts.TTS_THREADS, "load_ms": round(eng.load_ms), "first_audio_ms": stat(first),
            "sentence_ms": stat(lat), "rtf": round(synth_s / max(audio_s, 1e-9), 4),
            "rtf_definition": "sum(synth_ms) / sum(audio_s) over all samples", "samples": samples}


def bench_frontend(runs: int) -> dict:
    import vad

    sem = vad.SemanticEndpointer()
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(8 * 16000) * 0.05).astype(np.float32)
    sem.predict(x)
    st = []
    for _ in range(max(runs, 5) * 4):
        t = time.perf_counter()
        sem.predict(x)
        st.append((time.perf_counter() - t) * 1000)
    td = vad.TurnDetector()
    frame = (rng.standard_normal(512) * 3000).astype(np.int16).tobytes()
    for _ in range(50):
        td.push(frame)
    ft = []
    for _ in range(500):
        t = time.perf_counter()
        td.push(frame)
        ft.append((time.perf_counter() - t) * 1000)
    return {"smart_turn_ms": stat(st), "smart_turn_samples_ms": [round(x, 2) for x in st],
            "vad_frame_samples_ms": [round(x, 3) for x in ft], "smart_turn_threads": vad.SMART_TURN_THREADS,
            "vad_frame_ms": stat(ft), "denoise": vad.VAD_DENOISE}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--tag", default="local")
    ap.add_argument("--voice", default=None, help="TTS voice (default: open_tts.DEFAULT_VOICE)")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--skip", default="", help="comma list of stt,llm,tts,frontend to skip")
    a = ap.parse_args()
    import offline_engine
    import open_tts

    skip = set(filter(None, a.skip.split(",")))
    clips = json.loads((HERE / "clips.json").read_text())
    clips = [c for c in clips if (ROOT / c["path"]).exists()]
    if not clips and "stt" not in skip:
        sys.exit("no clips: run bench/make_clips.py first")
    res = {"tag": a.tag, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "machine": machine(),
           "runs": a.runs}
    t0 = time.perf_counter()
    if "frontend" not in skip:
        res["frontend"] = bench_frontend(a.runs)
        print("frontend done", flush=True)
    if "stt" not in skip:
        res["stt"] = bench_stt(clips, a.runs)
        print("stt done", flush=True)
    if "tts" not in skip:
        res["tts"] = bench_tts(a.runs, a.voice or open_tts.DEFAULT_VOICE)
        print("tts done", flush=True)
    if "llm" not in skip:
        res["llm"] = bench_llm(a.runs, a.max_tokens or offline_engine.LLM_MAX_TOKENS)
        print("llm done", flush=True)
    res["peak_rss_mb"] = peak_rss_mb()
    res["seconds"] = round(time.perf_counter() - t0, 1)
    out = HERE / "results" / f"stages_{a.tag}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}\n")
    print("| stage | metric | p50 | p90 | n |\n|---|---|---|---|---|")
    rows = []
    if "stt" in res:
        rows.append(("STT (Moonshine v2 base)", "decode ms", res["stt"]["decode_ms"]))
    if "llm" in res:
        rows += [("LLM (Gemma 4 E2B, CPU)", "time to first token ms", res["llm"]["ttft_ms"]),
                 ("LLM (Gemma 4 E2B, CPU)", "decode tokens/s", res["llm"]["decode_tps"])]
    if "tts" in res:
        rows.append((f"TTS ({res['tts']['voice']})", "first audio ms", res["tts"]["first_audio_ms"]))
    if "frontend" in res:
        rows += [("Smart Turn v3.2", "predict ms", res["frontend"]["smart_turn_ms"]),
                 ("Silero + GTCRN", "ms per 32 ms frame", res["frontend"]["vad_frame_ms"])]
    for name, metric, s in rows:
        print(f"| {name} | {metric} | {s['p50']} | {s['p90']} | {s['n']} |")
    if "tts" in res:
        print(f"\nTTS real-time factor: {res['tts']['rtf']}")
    if "stt" in res:
        print(f"STT word error rate on the bench clips: {res['stt']['wer']}")
    print(f"peak RSS of this process: {res['peak_rss_mb']} MB")


if __name__ == "__main__":
    main()
