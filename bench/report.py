"""Combine bench/results/{e2e_<tag>.jsonl, server_rss_<tag>.txt, stages_<tag>.json} into one markdown summary.

    .venv/bin/python bench/report.py --tag local
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def p(xs, q):
    return f"{np.percentile(xs, q):.0f}" if xs else "–"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="local")
    a = ap.parse_args()
    st = json.loads((RES / f"stages_{a.tag}.json").read_text())
    rows = [json.loads(x) for x in (RES / f"e2e_{a.tag}.jsonl").read_text().splitlines() if x.strip()]
    rss = [int(x) for x in (RES / f"server_rss_{a.tag}.txt").read_text().split() if x.strip().isdigit()]
    m = st["machine"]
    out = [f"## CPU benchmark `{a.tag}` ({st['timestamp']})", "",
           f"Machine: {m.get('cpu')}, {m.get('ram_gb')} GB RAM, {m.get('os')}, Python {m.get('python')}; "
           f"swap at start of stage bench: {m.get('swap_used', 'n/a')}; load average {m.get('load_avg')}.", ""]
    comp = [r for r in rows if r.get("kind") == "complete" and r.get("ttfa_ms") is not None]
    inc = [r for r in rows if r.get("kind") == "incomplete" and r.get("ttfa_ms") is not None]
    errs = [r for r in rows if r.get("error")]
    out += ["| Stage | Metric | p50 | p90 | n |", "|---|---|---|---|---|"]
    c = [r["ttfa_ms"] for r in comp]
    s = [r["srv_ttfa_ms"] for r in comp if r.get("srv_ttfa_ms") is not None]
    out.append(f"| End to end | end of speech → first audio, client-measured (ms) | {p(c, 50)} | {p(c, 90)} | {len(c)} |")
    out.append(f"| End to end | end of speech → first audio, server trace (ms) | {p(s, 50)} | {p(s, 90)} | {len(s)} |")
    for key, label in (("srv_hangover_ms", "turn detection wait after speech (ms)"),
                       ("srv_stt_after_commit_ms", "STT after turn commit (ms)"),
                       ("srv_llm_ttft_ms", "LLM first token in the live turn (ms)"),
                       ("srv_tts_first_ms", "TTS first sentence in the live turn (ms)")):
        v = [r[key] for r in comp if r.get(key) is not None]
        out.append(f"| End to end | {label} | {p(v, 50)} | {p(v, 90)} | {len(v)} |")
    sd = st["stt"]["decode_ms"]
    lens = [c["audio_s"] for c in st["stt"]["per_clip"].values()]
    out.append(f"| STT | Moonshine v2 base decode of a {min(lens):.1f}–{max(lens):.1f} s clip (ms) | {sd['p50']:.0f} | {sd['p90']:.0f} | {sd['n']} |")
    lt, ld = st["llm"]["ttft_ms"], st["llm"]["decode_tps"]
    out.append(f"| LLM | Gemma 4 E2B time to first token (ms) | {lt['p50']:.0f} | {lt['p90']:.0f} | {lt['n']} |")
    out.append(f"| LLM | Gemma 4 E2B decode (tokens/s) | {ld['p50']:.1f} | {ld['p90']:.1f} | {ld['n']} |")
    tf = st["tts"]["first_audio_ms"]
    out.append(f"| TTS | {st['tts']['voice']} time to first audio (ms) | {tf['p50']:.0f} | {tf['p90']:.0f} | {tf['n']} |")
    out.append(f"| TTS | {st['tts']['voice']} real-time factor | {st['tts']['rtf']:.3f} | | |")
    fe = st["frontend"]
    out.append(f"| Turn | Smart Turn v3.2 predict (ms) | {fe['smart_turn_ms']['p50']:.0f} | {fe['smart_turn_ms']['p90']:.0f} | {fe['smart_turn_ms']['n']} |")
    out.append(f"| Memory | server peak RSS (MB) | {max(rss) / 1024:.0f} | | |" if rss else "| Memory | server peak RSS | – | | |")
    out += ["", f"End-to-end runs: {len(comp)} complete-question turns over {len({r['id'] for r in comp})} clips; "
            f"{len(errs)} runs with errors. Incomplete-phrase clip: committed by "
            f"{', '.join(sorted({str(r.get('srv_commit_trigger')) for r in inc})) or '–'} in {len(inc)} runs, "
            f"first audio p50 {p([r['ttfa_ms'] for r in inc], 50)} ms (not included above).",
            f"LLM: {st['llm']['threads']} threads, system prompt {st['llm']['system_prompt_tokens']} tokens "
            f"(prefilled once), user-turn prefill p50 {st['llm']['user_turn_prefill_tokens']['p50']:.0f} tokens, "
            f"reply cap {st['llm']['max_tokens']} tokens. Stage-bench process peak RSS {st['peak_rss_mb']:.0f} MB."]
    print("\n".join(out))


if __name__ == "__main__":
    main()
