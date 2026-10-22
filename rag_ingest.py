"""Ingest a folder (or files) of your own documents into a profile's library, with live progress.

  .venv/bin/python rag_ingest.py --profile nova --dir ~/Documents/notes [--threads 2] [files ...]

Supported: PDF, Markdown, plain text, CSV and XLSX. The library is stored in data/<profile>_library.sqlite.
Safe beside a running server: embedding uses --threads ORT threads (default 2), numpy/BLAS are capped
to the same, and the process lowers its own priority (nice +10). No LLM is loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--dir", default=None, help="folder to ingest recursively")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--rebuild", action="store_true", help="delete every document in the profile's library first")
    ap.add_argument("--json", default=None, help="write the final report as JSON here")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()

    os.environ["EMBED_THREADS"] = str(a.threads)
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(a.threads)
    try:
        os.nice(a.nice)
    except OSError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import rag_store

    lib = rag_store.get(a.profile)
    if a.rebuild:
        n = len(lib.docs())
        lib.clear()
        print(f"rebuild: removed {n} docs from {lib.path}")
    last = {"line": ""}

    def progress(e: dict) -> None:
        ph = e["phase"]
        if ph == "done":
            r = e.get("report", {})
            print(f"\r  done  {e['file']}: {r.get('pages')} pages, {r.get('chunks')} chunks, {r.get('sentences')} sentences, "
                  f"{r.get('seconds')} s ({r.get('pages_per_s')} pages/s), headings={r.get('headings')}, no-text pages={r.get('scanned_pages')}",
                  flush=True)
            return
        if ph == "embed":
            what = f"{e.get('embedded', 0)}/{e.get('embed_total', 0)} vectors"
        elif ph == "index":
            what = f"{e.get('chunks')} chunks"
        else:
            what = f"page {e.get('page')}/{e.get('pages')}"
        eta = f" eta {e['eta_s']:.0f}s" if e.get("eta_s") else ""
        line = f"  {ph:<7} {e['file'][:48]:<48} {what:<24} {e.get('per_s', 0):>7.1f}/s{eta}"
        print("\r" + line.ljust(len(last["line"])), end="", flush=True)
        last["line"] = line

    t0 = time.perf_counter()
    if a.files:
        res = {"docs": 0, "skipped": 0, "pages": 0, "chunks": 0, "scanned_pages": 0, "errors": [], "files": []}
        for f in a.files:
            try:
                r = lib.ingest_file(f, progress=progress)
            except Exception as ex:  # noqa: BLE001
                res["errors"].append({"file": f, "error": str(ex)})
                continue
            res["files"].append(r)
            if r.get("skipped"):
                res["skipped"] += 1
            else:
                res["docs"] += 1
                for k in ("pages", "chunks", "scanned_pages"):
                    res[k] += r[k]
    else:
        if not a.dir:
            print("give --dir FOLDER or one or more files", file=sys.stderr)
            return 2
        d = a.dir
        print(f"ingesting {d} into profile {a.profile} ({lib.path}) with {a.threads} threads")
        res = lib.ingest_dir(d, progress=progress)
    secs = time.perf_counter() - t0
    print()
    for r in res["files"]:
        if r.get("skipped"):
            print(f"  skip  {r['file']} ({r.get('reason', 'already ingested')})")
    for c in res.get("deduped_csv", []):
        print(f"  skip  {c} (same sheet as an .xlsx)")
    for c in res.get("deduped_twins", []):
        print(f"  skip  {c} (twin of the .md with the same name; its pages are used for citations)")
    for e in res["errors"]:
        print(f"  ERROR {e['file']}: {e['error']}")
    st = lib.stats()
    print(f"report: {res['docs']} new docs, {res['skipped']} skipped, {res['pages']} pages, {res['chunks']} chunks in {secs:.1f} s"
          f" ({res['pages'] / max(secs, 1e-9):.1f} pages/s, {res['chunks'] / max(secs, 1e-9):.1f} chunks/s); pages without text: {res['scanned_pages']}")
    print(f"library: {st['docs']} docs, {st['indexed_chunks']} indexed chunks, {st['sentences']} sentences, "
          f"{st['disk_mb']} MB on disk, {st['vector_ram_mb']} MB vectors in RAM")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"result": res, "stats": st, "seconds": secs}, f, indent=1)
    return 1 if res["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
