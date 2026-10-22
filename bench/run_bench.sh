#!/usr/bin/env bash
# Full CPU benchmark with the shipped configuration:
#   1. start the server, play every bench clip RUNS times over /ws/voice (end-to-end time to first audio),
#      sampling the server's resident memory every 0.5 s;
#   2. stop the server (only one Gemma process at a time), then run the per-stage benchmark.
# Results go to bench/results/ (e2e_<tag>.jsonl, server_rss_<tag>.txt, stages_<tag>.json, summary_<tag>.md).
#
#   bench/run_bench.sh [--runs 5] [--tag local] [--port 8800]
# Extra environment (e.g. LITERT_THREADS=6) is passed to the server and to the stage benchmark.
set -euo pipefail
cd "$(dirname "$0")/.."
RUNS=5; TAG=local; PORT=8800
while [ $# -gt 0 ]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) echo "unknown argument $1"; exit 2 ;;
  esac
done
PY=.venv/bin/python
OUT=bench/results
mkdir -p "$OUT" logs
[ -f bench/clips.json ] && [ -f bench/clips/question.wav ] || "$PY" bench/make_clips.py
if "$PY" -c "import socket,sys; s=socket.socket(); sys.exit(s.connect_ex(('127.0.0.1', $PORT)) != 0)"; then
  echo "port $PORT is busy: stop the running server first"; exit 1
fi
if [ "$(uname)" = Darwin ]; then echo "swap: $(sysctl -n vm.swapusage)"; fi

echo "== starting server (tag $TAG)"
PIPE_TAG="bench_$TAG" "$PY" -m uvicorn app:app --host 127.0.0.1 --port "$PORT" > "logs/server_$TAG.log" 2>&1 &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null || true; [ -n "${SAMPLER:-}" ] && kill "$SAMPLER" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 300); do
  grep -q "Application startup complete" "logs/server_$TAG.log" 2>/dev/null && break
  kill -0 "$SRV" 2>/dev/null || { echo "server exited"; tail -20 "logs/server_$TAG.log"; exit 1; }
  sleep 1
done
( while kill -0 "$SRV" 2>/dev/null; do ps -o rss= -p "$SRV" 2>/dev/null || true; sleep 0.5; done ) > "$OUT/server_rss_$TAG.txt" &
SAMPLER=$!
sleep 2

echo "== end-to-end: $RUNS runs per clip"
rm -f "$OUT/e2e_$TAG.jsonl"
"$PY" bench/e2e_bench.py --runs "$RUNS" --tag "$TAG" --url "ws://127.0.0.1:$PORT/ws/voice" --out "$OUT/e2e_$TAG.jsonl" | tee "$OUT/e2e_$TAG.md"

kill "$SRV"; wait "$SRV" 2>/dev/null || true
sleep 1; kill "$SAMPLER" 2>/dev/null || true
PEAK_KB=$(sort -n "$OUT/server_rss_$TAG.txt" | tail -1)
echo "server peak RSS: $((PEAK_KB / 1024)) MB"

echo "== per-stage benchmark"
"$PY" bench/bench_stages.py --runs "$RUNS" --tag "$TAG" | tee "$OUT/stages_$TAG.md"
"$PY" bench/report.py --tag "$TAG" | tee "$OUT/summary_$TAG.md"
