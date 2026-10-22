## CPU benchmark `m2-8gb` (2026-10-22T16:46:59+00:00)

Machine: Apple M2, 8.0 GB RAM, Darwin 25.2.0 arm64, Python 3.11.5; swap at start of stage bench: total = 11264.00M  used = 10064.00M  free = 1200.00M  (encrypted); load average [5.53, 3.76, 3.05].

| Stage | Metric | p50 | p90 | n |
|---|---|---|---|---|
| End to end | end of speech → first audio, client-measured (ms) | 1523 | 2386 | 20 |
| End to end | end of speech → first audio, server trace (ms) | 1470 | 2303 | 20 |
| End to end | turn detection wait after speech (ms) | 577 | 818 | 20 |
| End to end | STT after turn commit (ms) | 8 | 51 | 20 |
| End to end | LLM first token in the live turn (ms) | 194 | 857 | 20 |
| End to end | TTS first sentence in the live turn (ms) | 171 | 209 | 20 |
| STT | Moonshine v2 base decode of a 1.4–3.0 s clip (ms) | 41 | 55 | 25 |
| LLM | Gemma 4 E2B time to first token (ms) | 333 | 396 | 25 |
| LLM | Gemma 4 E2B decode (tokens/s) | 30.0 | 33.0 | 25 |
| TTS | ljspeech time to first audio (ms) | 115 | 137 | 15 |
| TTS | ljspeech real-time factor | 0.030 | | |
| Turn | Smart Turn v3.2 predict (ms) | 67 | 83 | 20 |
| Memory | server peak RSS (MB) | 1565 | | |

End-to-end runs: 20 complete-question turns over 4 clips; 0 runs with errors. Incomplete-phrase clip: committed by hard cap in 5 runs, first audio p50 2131 ms (not included above).
LLM: 4 threads, system prompt 1206 tokens (prefilled once), user-turn prefill p50 16 tokens, reply cap 120 tokens. Stage-bench process peak RSS 1972 MB.
