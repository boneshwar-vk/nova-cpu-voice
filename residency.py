"""Optional (macOS): keep the server's working set in RAM on a memory-starved machine (KEEP_RESIDENT=1).

When the system is under memory pressure, macOS can swap an idle server down to a few MB; the next
turns then page the model back in and the first token is seconds late. After warm-up this module
mlock()s (wires) the process's own heap regions (model sessions, KV cache, buffers) and the mapped
XNNPACK packed-weight cache so they cannot be swapped or evicted. The per-layer-embedding table inside
the .litertlm (read sparsely) is left pageable. Regions are found with `vmmap` (once after warm-up,
then every RELOCK_S for new allocations). Costs RAM for other apps; off by default.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import re
import subprocess
import sys
import threading
import time

log = logging.getLogger("nova.residency")
KEEP_RESIDENT = os.environ.get("KEEP_RESIDENT", "0") == "1"
RELOCK_S = float(os.environ.get("RESIDENT_RELOCK_S", "120"))
LOCK_TYPES = ("MALLOC_LARGE", "MALLOC_SMALL", "MALLOC_TINY", "MALLOC_MEDIUM", "VM_ALLOCATE")
LOCK_FILES = (".xnnpack_cache",)
_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_RX = re.compile(r"^(\S.*?)\s+([0-9a-f]+)-([0-9a-f]+)\s+\[\s*([\d.]+[KMG])")
_locked: set[tuple[int, int]] = set()
_stats = {"locked_mb": 0.0, "regions": 0, "failed": 0, "passes": 0}


def _regions() -> list[tuple[str, int, int, str]]:
    out = subprocess.run(["vmmap", "-wide", str(os.getpid())], capture_output=True, text=True, timeout=60).stdout
    regs = []
    for line in out.splitlines():
        m = _RX.match(line)
        if not m:
            continue
        kind = m.group(1).strip()
        start, end = int(m.group(2), 16), int(m.group(3), 16)
        if "(empty)" in kind or "guard" in kind or "reserved" in line.lower():
            continue
        if any(kind.startswith(t) for t in LOCK_TYPES) or (kind.startswith("mapped file") and any(f in line for f in LOCK_FILES)):
            regs.append((kind, start, end, line))
    return regs


MAX_MB = float(os.environ.get("RESIDENT_MAX_MB", "2300"))  # never wire more than this (leave room for the OS)


def lock_pass() -> dict:
    t0 = time.perf_counter(); new_mb = 0.0
    regs = sorted(_regions(), key=lambda r: r[2] - r[1], reverse=True)  # biggest first: weights cache, KV, model heaps
    # Locked regions are never forgotten: once wired, vmmap no longer lists them the same way, so "not
    # listed" does not mean "freed". Counting them forever keeps the 2.3 GB cap conservative.
    for kind, start, end, _ in regs:
        if (start, end) in _locked:
            continue
        if _stats["locked_mb"] + (end - start) / 2**20 > MAX_MB:
            continue
        if _libc.mlock(start, end - start) == 0:
            _locked.add((start, end)); mb = (end - start) / 2**20
            _stats["locked_mb"] += mb; new_mb += mb; _stats["regions"] += 1
        else:
            _stats["failed"] += 1
    _stats["passes"] += 1
    log.info("residency: locked +%.0f MB (total %.0f MB in %d regions, %d not lockable) in %.1f s",
             new_mb, _stats["locked_mb"], _stats["regions"], _stats["failed"], time.perf_counter() - t0)
    return dict(_stats)


def start() -> None:
    if not KEEP_RESIDENT or sys.platform != "darwin":
        return
    lock_pass()

    def loop() -> None:
        while True:
            time.sleep(RELOCK_S)
            try:
                lock_pass()
            except Exception as e:  # noqa: BLE001
                log.warning("residency pass failed: %s", e)
    threading.Thread(target=loop, name="residency", daemon=True).start()


def stats() -> dict:
    return dict(_stats, on=KEEP_RESIDENT)
