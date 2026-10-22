"""Per-turn latency trace for /ws/voice.

All absolute marks are `time.monotonic()` seconds; the derived *_ms fields are
what the benchmarks read. Perceived TTFA = first_audio_t - speech_end_t, where
speech_end_t is the arrival time of the last push that contained speech (so it
includes the VAD hangover and the turn-detection wait, as the user hears it).

One JSON line per turn is appended to logs/turns.jsonl (tag from $PIPE_TAG) and
the same dict is sent to the client as {"type": "trace", ...} before audio_end.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "logs" / "turns.jsonl"
TAG = os.environ.get("PIPE_TAG", "")
_lock = threading.Lock()


def now() -> float:
    return time.monotonic()


def _ms(a: float | None, b: float | None) -> float | None:
    """(b - a) in ms, or None when either mark is missing."""
    if a is None or b is None:
        return None
    return round((b - a) * 1000.0, 1)


@dataclass
class TurnTrace:
    turn_id: int
    tag: str = TAG
    # --- listening side (frame loop) ---
    speech_start_t: float | None = None
    speech_end_t: float | None = None     # last push that contained speech
    soft_t: float | None = None           # endpoint_soft event
    check_t: float | None = None          # endpoint (check) event
    smart_turn_ms: list[float] = field(default_factory=list)
    smart_turn_p: list[float] = field(default_factory=list)
    eager_start_t: float | None = None
    commit_t: float | None = None
    commit_trigger: str | None = None
    utter_ms: float | None = None
    # --- reply side (process) ---
    stt_eager_used: bool = False
    stt_ms: float | None = None           # STT call wall time (eager: since launch)
    transcript_t: float | None = None
    transcript: str = ""
    llm_start_t: float | None = None
    llm_first_token_t: float | None = None
    llm_first_sentence_t: float | None = None
    reply_first: str = ""
    tts_first_ms: float | None = None     # synth time of the first sentence
    first_audio_t: float | None = None    # first audio bytes handed to the socket
    audio_end_t: float | None = None
    n_sentences: int = 0
    cancelled: bool = False
    error: str | None = None
    # --- benchmark labels ---
    llm_backend: str | None = None        # "session" (LiteRT-LM raw session, CPU)
    clip_id: str | None = None            # client label of the played clip
    u_bucket: str | None = None           # U1..U4 (client label, else from utt_words)
    utt_words: int | None = None
    first_chunk_words: int | None = None  # words in the first spoken chunk
    r_first_sentence_words: int | None = None
    # --- memory (memory_store.py) ---
    memory_ms: float | None = None        # embed + exact scan, off the critical path
    memory_hits: int | None = None        # facts injected into the user chunk
    # --- monitoring ---
    profile: str | None = None            # whose memory (separate file per profile)
    voice: str | None = None              # TTS voice id
    reply: str = ""                       # Nova's full reply as spoken
    memory_facts: list[str] = field(default_factory=list)  # recalled fact texts
    # --- document library (rag_store.py) ---
    rag_ms: float | None = None           # library search (hybrid + compression)
    rag_gate: bool | None = None          # True = answered from the person's documents
    rag_sources: list[str] = field(default_factory=list)  # "Short title p. N"
    filler: str = ""                      # Nova's short lead-in while she looked it up
    safety: str | None = None             # child-safety guard category (safety.py), when it fired
    tool: str | None = None               # tool used this turn (tools.py), if any
    tool_via: str | None = None           # "rule" (fast router) or "model" (Gemma 4 native call)
    tool_online: bool = False             # the tool used the internet
    tool_ms: float | None = None          # tool run time

    def mark_speech(self) -> None:
        t = now()
        if self.speech_start_t is None:
            self.speech_start_t = t
        self.speech_end_t = t

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(
            hangover_ms=_ms(self.speech_end_t, self.commit_t),
            soft_after_speech_ms=_ms(self.speech_end_t, self.soft_t),
            stt_after_commit_ms=_ms(self.commit_t, self.transcript_t),
            llm_ttft_ms=_ms(self.llm_start_t, self.llm_first_token_t),
            llm_first_sentence_ms=_ms(self.llm_start_t, self.llm_first_sentence_t),
            commit_to_audio_ms=_ms(self.commit_t, self.first_audio_t),
            ttfa_ms=_ms(self.speech_end_t, self.first_audio_t),
            reply_total_ms=_ms(self.commit_t, self.audio_end_t),
            utt_audio_ms=self.utter_ms,
            chunks_n=self.n_sentences,
            bucket=self.u_bucket,
        )
        return d

    def finish(self) -> dict:
        """Append to logs/turns.jsonl and return the flat dict for the client."""
        d = self.as_dict()
        d["wall"] = time.time()
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with _lock, LOG_PATH.open("a") as f:
                f.write(json.dumps(d) + "\n")
        except OSError:
            pass
        return d
