# nova-cpu-voice

Nova, a voice assistant that runs entirely on the CPU and works offline (Silero VAD + GTCRN → Smart Turn v3.2 → Moonshine v2 → Gemma 4 E2B on LiteRT-LM → Piper TTS, with a web page). A few optional tools (weather, news, web look-ups) use the internet when available; `TOOLS=0` turns all tools off.

## Install

Needs macOS 14+ on Apple silicon (tested) or Linux x86_64/aarch64 (should work, untested), Python 3.11, 8 GB of RAM and about 4.5 GB of free disk. Windows and Intel Macs are not supported.

```bash
git clone https://github.com/boneshwar-vk/nova-cpu-voice.git
cd nova-cpu-voice
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env  # only if you do not have uv
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/download_models.py                                     # about 2.8 GB, one time
```

## Use

```bash
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8800
```

Wait until the terminal prints `Application startup complete` (about half a minute while the models load), then open http://127.0.0.1:8800/, press **Start conversation**, allow the microphone and talk. Headphones make interrupting Nova more reliable. Stop the server with Ctrl+C.

## CPU numbers

Apple M2 (4P + 4E cores), 8 GB RAM, macOS 26.2, 22 Oct 2026, shipped config (Gemma on CPU with 4 threads, Moonshine v2 base, `ljspeech` voice, memory and tools on) under everyday load (other apps open, ~10 GB swap in use, load avg 5.5 / 3.8 / 3.1 over 1 / 5 / 15 min). Measured with `bench/run_bench.sh --runs 5 --tag m2-8gb`: end to end = 4 synthetic `ljspeech` questions × 5 runs, no errors; raw data in `bench/results/`.

| Stage | Metric | p50 | p90 | n |
|---|---|---|---|---|
| End to end | end of speech → first audio, measured by the client | 1523 ms | 2386 ms | 20 |
| End to end | of which: waiting for the turn detector to decide | 577 ms | 818 ms | 20 |
| End to end | of which: Gemma first token after the transcript (speculative prefill) | 194 ms | 857 ms | 20 |
| Speech to text | Moonshine v2 base, 1.4–3.1 s clip | 41 ms | 55 ms | 25 |
| Language model | Gemma 4 E2B int4, time to first token (new 16-token user turn) | 333 ms | 396 ms | 25 |
| Language model | Gemma 4 E2B int4, decode speed | 30.0 tokens/s | 33.0 tokens/s | 25 |
| Text to speech | `ljspeech` voice, time to first audio (first sentence) | 115 ms | 137 ms | 15 |
| Text to speech | `ljspeech` voice, real-time factor | 0.030 | | 30 sentences |
| Turn detection | Smart Turn v3.2, one prediction | 67 ms | 83 ms | 20 |
| Memory | server peak resident memory | 1565 MiB | | |

Third-party model licences and attribution: [THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md). Code: MIT, see [LICENSE](LICENSE).
