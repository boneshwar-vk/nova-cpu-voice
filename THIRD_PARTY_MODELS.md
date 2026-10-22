# Third-party models and assets

No model weights are stored in this repository. `scripts/download_models.py` fetches each file below
from its upstream source (pinned revision or release asset, checked by size and SHA-256) into `models/`.
Each model keeps its own licence. The code in this repository is MIT (see `LICENSE`).

| Model | Used for | Licence | Downloaded from |
|---|---|---|---|
| Gemma 4 E2B it, `.litertlm` (LiteRT-LM community build; mixed 2/4/8-bit weights, mostly int4) | language model | Apache-2.0 (`https://ai.google.dev/gemma/docs/gemma_4_license`) | `https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm` @ `b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1` |
| Silero VAD v6.2 (`silero_vad.onnx`) | voice activity detection | MIT | `https://github.com/snakers4/silero-vad` @ `be95df9152c0d7618fa1edfeb296fc3dae32376f` |
| GTCRN (`gtcrn_simple.onnx`) | denoising on the VAD branch | MIT (upstream `Xiaobin-Rong/gtcrn`) | sherpa-onnx release `speech-enhancement-models` |
| Smart Turn v3.2 CPU (`smart-turn-v3.2-cpu.onnx`) | end-of-turn detection | BSD-2-Clause | `https://huggingface.co/pipecat-ai/smart-turn-v3` @ `f766f81d3cfdf7737ac64aad813d91bbfd56bf93` |
| Moonshine v2 base and tiny, English, int8 (`sherpa-onnx-moonshine-{base,tiny}-en-quantized-2026-02-27`) | speech to text (base) and live captions (tiny) | MIT (English models, see the `LICENSE` file inside the archive; upstream `moonshine-ai/moonshine`) | sherpa-onnx release `asr-models` |
| Piper `en_US-ljspeech-medium` (`vits-piper-en_US-ljspeech-medium`), default voice "ljspeech" | text to speech | Model card: "Trained from scratch for 1000 epochs on medium quality settings using the LJ Speech dataset"; dataset licence: public domain. The `rhasspy/piper-voices` repository is tagged MIT. | sherpa-onnx release `tts-models`; original voice: `https://huggingface.co/rhasspy/piper-voices` (`en/en_US/ljspeech/medium`) |
| Piper `en_GB-southern_english_female-low` (`vits-piper-en_GB-southern_english_female-low`), optional voice "sef" | text to speech | Dataset OpenSLR SLR83, CC BY-SA 4.0; `rhasspy/piper-voices` repository tagged MIT; fine-tuned from `en_US-ryan-low`, whose RyanSpeech data is CC BY-NC-SA 4.0 (non-commercial lineage, commercial use unclear; see below) | sherpa-onnx release `tts-models` (only with `--sef`); original voice: `https://huggingface.co/rhasspy/piper-voices` (`en/en_GB/southern_english_female/low`) |
| Matcha-TTS LJSpeech (`matcha-icefall-en_US-ljspeech`), optional voice "matcha" | text to speech | model and code Apache-2.0 (`k2-fsa/icefall`); LJSpeech data public domain | sherpa-onnx release `tts-models` (only with `--matcha`) |
| Vocos 22 kHz (`vocos-22khz-univ.onnx`), used by the optional Matcha voice | vocoder | Apache-2.0 (upstream `https://huggingface.co/BSC-LT/vocos-mel-22khz`) | sherpa-onnx release `vocoder-models` (only with `--matcha`) |
| all-MiniLM-L6-v2, ONNX int8 (`model_quantized.onnx`, `tokenizer.json`) | embeddings for memory and the document library | Apache-2.0 (ONNX export of `sentence-transformers/all-MiniLM-L6-v2`) | `https://huggingface.co/Xenova/all-MiniLM-L6-v2` @ `751bff37182d3f1213fa05d7196b954e230abad9` |

The Piper and Matcha archives also contain `espeak-ng-data`, the phoneme data of eSpeak NG
(`https://github.com/espeak-ng/espeak-ng`), which is GPL-3.0-or-later. sherpa-onnx uses it to turn
text into phonemes for these voices. The sherpa-onnx binary wheels also compile in espeak-ng library code
(GPL-3.0-or-later). This repository only imports sherpa-onnx from PyPI and does not redistribute it; a bundled
build that ships sherpa-onnx (for example a packaged app) takes on the GPL-3.0 terms.

## Default voice

`en_US-ljspeech-medium` from the Piper voice collection (`https://github.com/rhasspy/piper`; training notes
linked from the model card: `https://brycebeattie.com/files/tts/`), converted for sherpa-onnx by the k2-fsa project. Its model card
(`https://huggingface.co/rhasspy/piper-voices/raw/main/en/en_US/ljspeech/medium/MODEL_CARD`) states the
dataset `https://keithito.com/LJ-Speech-Dataset/` with "License: public domain" and that the voice was
"Trained from scratch for 1000 epochs on medium quality settings using the LJ Speech dataset".

## Optional "sef" voice (CC BY-SA 4.0, see the caveat)

Downloaded only with `scripts/download_models.py --sef`. The voice is the Piper voice
`en_GB-southern_english_female-low`, trained on the southern English female speakers of:

> Isin Demirsahin, Oddur Kjartansson, Alexander Gutkin, Clara Rivera. "Open-source Multi-speaker Corpora
> of the English Accents in the British Isles." Proceedings of LREC 2020. OpenSLR SLR83,
> `https://www.openslr.org/83/`, licensed under Creative Commons Attribution-ShareAlike 4.0 International
> (`https://creativecommons.org/licenses/by-sa/4.0/`).

If you redistribute this voice or audio made with it, keep this attribution and the share-alike terms.
The `rhasspy/piper-voices` repository is tagged MIT, but the voice's model card says it was fine-tuned from
`en_US-ryan-low`, whose model card lists its training data (RyanSpeech) as CC BY-NC-SA 4.0. Because of this
non-commercial lineage, commercial use of "sef" is unclear; check it yourself before any commercial use.
The voice is a derived work of SLR83: Piper trained it on that data, and k2-fsa converted it to ONNX for
sherpa-onnx. The dataset is provided "as is", without warranties; see section 5 of the licence.

## Bundled web fonts

`static/fonts/` contains Latin subsets of EB Garamond (Georg Duffner, Octavio Pardo) and Inter
(Rasmus Andersson), both under the SIL Open Font License 1.1 (`static/fonts/OFL.txt`).

## Runtime libraries

Installed from PyPI by `requirements.txt`, each under its own licence: LiteRT-LM (`litert-lm-api`,
Apache-2.0), sherpa-onnx (Apache-2.0), ONNX Runtime (MIT), FastAPI (MIT), Starlette (BSD-3-Clause),
Pydantic (MIT), Uvicorn (BSD-3-Clause), websockets (BSD-3-Clause), NumPy and SciPy (BSD-3-Clause), soundfile
(BSD-3-Clause; its wheels bundle libsndfile, LGPL-2.1), tokenizers (Apache-2.0), pypdfium2 (Apache-2.0 or
BSD-3-Clause; PDFium binary under its own BSD-style licences), openpyxl (MIT). Development only: pytest (MIT),
reportlab (BSD).

## Online data sources (optional tools)

With tools on (the default) and an internet connection, some requests use these services. Each has its own
terms; check them before any commercial use, or set `TOOLS=0`.

- Open-Meteo (weather): data CC BY 4.0, "Weather data by Open-Meteo.com"; the free API is for non-commercial use.
- Wikipedia (summaries): text CC BY-SA 4.0.
- DuckDuckGo Instant Answers (web look-ups).
- Google News RSS (headlines).
- Frankfurter (exchange rates).

## Demo video

`demo/nova-backprop-demo.mp4` uses the Piper `en_US-ljspeech-medium` voice (Nova) and the Piper `en_US-norman-medium` voice (Sam, LibriVox data, public domain, trained from scratch: `https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/norman/medium`). The font is Inter (OFL 1.1).
