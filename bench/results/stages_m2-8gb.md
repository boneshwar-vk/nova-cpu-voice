frontend done
stt done
tts done
llm done

wrote bench/results/stages_m2-8gb.json

| stage | metric | p50 | p90 | n |
|---|---|---|---|---|
| STT (Moonshine v2 base) | decode ms | 40.6 | 55.2 | 25 |
| LLM (Gemma 4 E2B, CPU) | time to first token ms | 333.0 | 395.6 | 25 |
| LLM (Gemma 4 E2B, CPU) | decode tokens/s | 30.0 | 33.0 | 25 |
| TTS (ljspeech) | first audio ms | 114.7 | 137.1 | 15 |
| Smart Turn v3.2 | predict ms | 66.6 | 82.6 | 20 |
| Silero + GTCRN | ms per 32 ms frame | 1.5 | 1.6 | 500 |

TTS real-time factor: 0.0296
STT word error rate on the bench clips: 0.364
peak RSS of this process: 1972.2 MB
