#!/usr/bin/env python3
"""Summarize numeric fields of a JSONL (or JSON-in-log-line) file.

Usage:
  summarize.py FILE.jsonl [--group KEY[,KEY]] [--keys k1,k2] [--md] [--no-overview]

Each line holds a flat dict. Nested dicts are flattened to dotted keys; a list
of numbers is reduced to its per-line mean under `key.mean`. A line may carry a
text prefix before the JSON object (`[voice-turn] {...}`): parsing starts at
the first `{`. Unparsable lines are skipped silently
and counted in the footer.

For every numeric key (int/float; bool/None ignored) prints n, p50, p90, min,
max (numpy percentile, method="nearest"). `--group` prints one table per
distinct value of the group key(s); `--keys` restricts/orders the reported
keys; `--md` emits GitHub-markdown tables. Exits 0 even with no numeric keys.

With `--group`, an overview table comes first: one row per group with n and
p50 / p90 of every TTFA field present (`ttfa_ms`, `srv_ttfa_ms`,
`ttfa_content_ms`, `ttfa_ack_ms`, ...). E.g. `--group clip_id` or
`--group u_bucket` on logs/turns.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_DECODER = json.JSONDecoder()
HEADER = ("key", "n", "p50", "p90", "min", "max")


def parse_line(line: str) -> dict | None:
    """Return the first JSON object found in `line`, or None."""
    i = line.find("{")
    if i < 0:
        return None
    try:
        obj, _end = _DECODER.raw_decode(line[i:])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def flatten(obj: dict, prefix: str = "", out: dict | None = None) -> dict:
    out = {} if out is None else out
    for k, v in obj.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flatten(v, key + ".", out)
        elif isinstance(v, list):
            if v and all(is_num(x) for x in v):
                out[key + ".mean"] = float(np.mean(v))
        else:
            out[key] = v
    return out


def load(path: Path) -> tuple[list[dict], int]:
    rows: list[dict] = []
    skipped = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            obj = parse_line(line)
            if obj is None:
                skipped += 1
                continue
            rows.append(flatten(obj))
    return rows, skipped


def fmt(v) -> str:
    v = float(v)
    a = abs(v)
    if a >= 1e15:
        return f"{v:.4g}"
    if v.is_integer():
        return str(int(v))
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def stats(rows: list[dict], keys: list[str] | None, exclude: set[str]) -> list[tuple]:
    cols: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            if k not in exclude and is_num(v):
                cols[k].append(v)
    table = []
    for k in (keys if keys is not None else sorted(cols)):
        vals = cols.get(k)
        if not vals:
            continue
        a = np.asarray(vals, dtype=float)
        p50, p90 = np.percentile(a, [50, 90], method="nearest")
        table.append((k, len(a), p50, p90, a.min(), a.max()))
    return table


def render(table: list[tuple], md: bool) -> str:
    if not table:
        return "(no numeric keys)"
    cells = [HEADER] + [(k, str(n)) + tuple(fmt(x) for x in rest) for k, n, *rest in table]
    if md:
        lines = ["| " + " | ".join(HEADER) + " |", "|---|" + "---:|" * (len(HEADER) - 1)]
        lines += ["| " + " | ".join(c) + " |" for c in cells[1:]]
        return "\n".join(lines)
    widths = [max(len(c[i]) for c in cells) for i in range(len(HEADER))]
    return "\n".join(
        c[0].ljust(widths[0]) + "  " + "  ".join(x.rjust(w) for x, w in zip(c[1:], widths[1:]))
        for c in cells
    )


_TTFA = re.compile(r"(^|_)ttfa(_[a-z]+)?_ms$")


def overview(groups: dict[tuple, list[dict]], group_keys: list[str], md: bool) -> str:
    """One row per group: n and p50/p90 of each TTFA key present."""
    tkeys = sorted({k for rs in groups.values() for r in rs for k, v in r.items()
                    if _TTFA.search(k) and is_num(v)}, key=lambda k: (k.startswith("srv_"), k))
    if not tkeys:
        return ""
    head = ["/".join(group_keys) or "all", "n"] + [f"{k} {q}" for k in tkeys for q in ("p50", "p90")]
    body = []
    for g in sorted(groups, key=lambda t: tuple(_nat(x) for x in t)):
        rs = groups[g]
        row = [",".join(g), str(len(rs))]
        for k in tkeys:
            vals = [r[k] for r in rs if is_num(r.get(k))]
            if vals:
                p50, p90 = np.percentile(np.asarray(vals, float), [50, 90], method="nearest")
                row += [fmt(p50), fmt(p90)]
            else:
                row += ["-", "-"]
        body.append(row)
    if md:
        return "\n".join(["| " + " | ".join(head) + " |", "|---|" + "---:|" * (len(head) - 1)]
                         + ["| " + " | ".join(r) + " |" for r in body])
    w = [max(len(r[i]) for r in [head] + body) for i in range(len(head))]
    return "\n".join(r[0].ljust(w[0]) + "  " + "  ".join(x.rjust(wi) for x, wi in zip(r[1:], w[1:]))
                     for r in [head] + body)


def _nat(v: str) -> tuple:
    """Sort key: numeric-looking group values numerically, others as text."""
    try:
        return (0, float(v), "")
    except ValueError:
        return (1, 0.0, v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("file", type=Path, help="JSONL file (or log with JSON objects in lines)")
    ap.add_argument("--group", default="", help="comma-separated key(s) to group rows by")
    ap.add_argument("--keys", default="", help="comma-separated numeric keys to report (default: all)")
    ap.add_argument("--md", action="store_true", help="print GitHub-markdown tables")
    ap.add_argument("--no-overview", action="store_true",
                    help="with --group: skip the per-group TTFA p50/p90 overview table")
    args = ap.parse_args(argv)

    if not args.file.is_file():
        print(f"error: no such file: {args.file}", file=sys.stderr)
        return 1

    rows, skipped = load(args.file)
    group_keys = [k.strip() for k in args.group.split(",") if k.strip()]
    keys = [k.strip() for k in args.keys.split(",") if k.strip()] or None

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[tuple(str(r.get(k, "-")) for k in group_keys)].append(r)

    out: list[str] = []
    if group_keys and not args.no_overview:
        ov = overview(groups, group_keys, args.md)
        if ov:
            out.append("### TTFA by " + "/".join(group_keys) if args.md else
                       "== TTFA by " + "/".join(group_keys) + " (ms) ==")
            out.append(ov)
    for gvals in sorted(groups, key=lambda t: tuple(_nat(x) for x in t)):
        if group_keys:
            title = ", ".join(f"{k}={v}" for k, v in zip(group_keys, gvals))
            out.append(f"\n### {title}" if args.md else f"\n== {title} ==")
        out.append(render(stats(groups[gvals], keys, set(group_keys)), args.md))
    if not rows:
        out.append("(no rows)")
    tail = f"{len(rows)} rows, {skipped} skipped ({args.file.name})"
    out.append(f"\n_{tail}_" if args.md else f"\n{tail}")
    print("\n".join(out).lstrip("\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
