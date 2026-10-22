"""Full-duplex voice WebSocket: /ws/voice (everything local, CPU only).

Client streams mic PCM16 16 kHz mono frames (binary) up; sends JSON control
(config/greet/stop) as text. Server runs Silero VAD (+ GTCRN denoiser on the VAD
branch) and Smart Turn endpointing, then per turn: Moonshine STT -> (echo | streaming
Gemma 4 E2B on LiteRT-LM CPU) -> per-sentence TTS (open_tts, sherpa-onnx), streaming
one PCM16 WAV per sentence back. Barge-in: user speech during playback cancels the reply.

Protocol (server -> client, JSON text):
  {type:state, value:listening|processing|speaking}
  {type:transcript, text}     {type:reply, text}      {type:sentence, text}
  {type:partial, text}        {type:stage, ...}       live progress for the page
  {type:barge_in}             {type:audio_end}        {type:error, message}
  {type:trace, ...}           per-turn latency marks (turn_trace.TurnTrace),
                              sent just before audio_end
Binary server -> client = one playable audio chunk (a spoken sentence).
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import struct
import threading
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger("nova.realtime")

import offline_engine
import open_tts
from stt_sherpa import looks_degenerate as _looks_degenerate
from turn_trace import TurnTrace, now
from vad import SR, SemanticEndpointer, TurnDetector

import embedder
import memory_store
from memory_store import compose_prompt, extract_facts, format_block
import personas
import conv_store
import safety
import tools

try:  # document library per profile (rag_store.py); the pipeline still runs without it
    import rag_store
    RAG_ON = os.environ.get("RAG", "1") != "0"
except Exception as _e:  # noqa: BLE001
    rag_store = None
    RAG_ON = False


def _rag_lookup(profile: str | None, text: str) -> dict | None:
    """Search this profile's document library; None when there is no library or no documents."""
    if not RAG_ON or not text.strip():
        return None
    try:
        lib = rag_store.get(profile or personas.ORDER[0])
        if not lib.docs():
            return None
        return lib.search(text, k=3, budget_tokens=RAG_BUDGET)
    except Exception as e:  # noqa: BLE001 — the library must never break a turn
        log.warning("library search failed: %s", e)
        return None


RAG_BUDGET = int(os.environ.get("RAG_BUDGET", "200"))  # tokens of passage; its prefill hides behind the lead-in
MEMORY_WINS_COS = float(os.environ.get("MEMORY_WINS_COS", "0.5"))


def _use_library(rag: dict | None, mem_facts: list[dict]) -> bool:
    """Library answer only when its gate opens AND no personal memory matches clearly better
    ("who is my best friend" must come from memory, not from a story that mentions friends)."""
    if not rag or not rag.get("gate") or _chat(rag.get("query_norm") or ""):
        return False
    best_mem = max((float(f.get("cos", 0)) for f in mem_facts), default=0.0)
    personal = _personal(rag.get("query_norm") or "")
    return not (personal and best_mem >= MEMORY_WINS_COS and best_mem >= float(rag.get("top_score", 0)) - 0.05)


def _rag_user_chunks(text: str, rag: dict, mem_facts: list[dict], profile: str | None) -> tuple[str, str]:
    """The two user turns of a library answer: a [LOOKUP] cue (Gemma says one short lead-in
    while the passage is read) and the [NOTES] turn with the compressed passage + memory facts."""
    p = personas.PERSONAS.get(profile or "", personas.PERSONAS[personas.ORDER[0]])
    name, lib = p["name"] or "the user", p["library"]
    ex = personas.leadin_examples(profile)[0]
    lookup = (f"{text}\n(First reply with ONE short sentence of at most ten words, speaking directly to {name}, "
              f"saying you are checking their {lib}, for example: \"{ex}\" Do not answer the question yet.)")
    block = format_block(mem_facts)
    if _STORY_RX.search(text):  # "tell me a story about a spider": retell it, don't describe it or refuse
        notes = (f"Notes from {name}'s {lib}:\n{rag['context']}\n" + (block + "\n" if block else "")
                 + f"{name} asked: {text}\nRetell the story in these notes for {name} in four or five short, simple "
                 f"sentences, as a storyteller would, starting directly with the story (you already said you are checking). "
                 f"Begin with a sentence like 'Here is the story of the clever crow.' using the real title. Never write labels like 'Title:'.")
    else:
        notes = (f"Notes from {name}'s {lib}:\n{rag['context']}\n" + (block + "\n" if block else "")
                 + f"Using only these notes, answer {name}'s question: {text}\n"
                 f"Start directly with the answer (you already said you are checking). Mention the source once by its short title. "
                 f"If the notes do not answer it, say it is not in their {lib}.")
    return lookup, notes


# Small talk and questions about the conversation itself: never a library lookup or a "not in your
# library" reply ("Do you like ice cream?", "What did you just say?", "How's your day going?").
_META_RX = re.compile(r"\b(you|you've|you're) (just )?(say|said|mean|meant)\b|\b(say|repeat) (that|it)( again)?\b|\bsay again\b", re.I)
_SMALL_RX = re.compile(
    r"^\s*(do|did|are|were|have|can|could|would|will|is) (you|your)\b(?!.*\b(tell|explain|find|check|look|show|read|give|help|summari[sz]e)\b)"
    r"|\bhow('s| is| are| was)? (you|your)\b|\bwhat('s| is) your (name|favou?rite)\b|\byour (day|weekend|favou?rite)\b"
    r"|^\s*(hi|hello|hey|thanks|thank you|good (morning|night|evening|afternoon))\b[\w ,!.]{0,20}$", re.I)
_LIBRARY_CUE = re.compile(r"\b(my|our|notes?|books?|papers?|orders?|stock|files?|diary|chapters?|sheets?|story|stories)\b", re.I)


_LEADIN_RX = re.compile(r"(on it|one sec|one moment|hold on|let me|let's|i'?m checking|i am checking|i'?m looking|i am looking|"
                        r"i will (check|look)|i see (the|your) book|checking|pulling up|opening|looking)\b", re.I)


def _chat(text: str) -> bool:
    """Small talk or a question about the conversation itself: never a library lookup or a
    "not in your library" reply ("Do you like ice cream?", "What did you just say?")."""
    t = text or ""
    return bool(_META_RX.search(t) or (_SMALL_RX.search(t) and not _LIBRARY_CUE.search(t)))


_STORY_RX = re.compile(r"\b(story|stories|tale|read me a|bedtime story)\b", re.I)


def _personal(text: str) -> bool:
    return bool(re.search(r"\b(my|me|i|i'm|mine|our)\b", text, re.I))


def _not_found_note(profile: str | None, text: str, rag: dict | None, mem_facts: list[dict]) -> str:
    """For a knowledge question the library could not answer, tell Gemma so (no guessing)."""
    if rag is None:  # no library for this person
        return ""
    question = text.strip().endswith("?") or re.match(
        r"^\s*(what|which|who|whom|where|when|why|how|is|are|was|were|does|do|did|can|could|explain|tell me|define)\b", text, re.I)
    best_mem = max((float(f.get("cos", 0)) for f in mem_facts), default=0.0)
    if not question or _chat(text) or (_personal(text) and best_mem >= 0.40):
        return ""
    p = dict(personas.PERSONAS.get(profile or "", personas.PERSONAS[personas.ORDER[0]]))
    p["name"] = p["name"] or "the user"
    if _personal(text) and re.search(r"\bmy\b", text, re.I) and not re.search(
            r"\b(orders?|stock|sales?|returns?|books?|notes?|papers?|chapters?|exams?|tests?|library|files?|documents?|diary|sheets?)\b", text, re.I):
        # a question about their own life that memory does not cover: not a library miss
        return (f"\n(Note: you do not know this about {p['name']} yet. Say so warmly in one sentence and invite "
                f"{p['name']} to tell you. Do not guess.)")
    return (f"\n(Note: {p['name']}'s {p['library']} were searched and nothing relevant was found. Tell {p['name']} "
            f"\"That isn't in your {p['library']}.\" and ask if they want a general answer. Do not answer it now.)")

# Memory layer (memory_store.py): off with MEMORY=0 or when the embedding model is missing.
MEMORY_ON = os.environ.get("MEMORY", "1") != "0" and embedder.available()
TOOLS_ON = os.environ.get("TOOLS", "1") != "0"  # tools.py: 15 tools, rules first, Gemma 4 native calls as fallback
_TAG_RX = re.compile(r"<\|?(?:tool_call|tool_response|tool|channel|turn|\")\|?>")  # special tokens never reach TTS
_CALL_PREFIXES = ("<|tool_call>", "call:")
_CALL_START_RX = re.compile(r"(?:<\|tool_call>|call:)")
_NO_TOOL_REPLY = "I'd need to search the web for that. Say 'search the web for' and your question if you want me to."

# Smart Turn v3 semantic endpointer, shared across connections (stateless
# inference). None when the model file is absent — the detector then commits
# on the plain silence "endpoint" (TURN_CHECK_MS) alone.
_semantic: SemanticEndpointer | None = None
_semantic_lock = threading.Lock()


def _get_semantic() -> SemanticEndpointer | None:
    global _semantic
    if _semantic is None and SemanticEndpointer.available():
        with _semantic_lock:
            if _semantic is None:
                _semantic = SemanticEndpointer()
    return _semantic

# Cap a single utterance so a client that never endpoints can't grow memory
# without bound. 16 kHz * 2 bytes * 60 s = ~1.9 MB.
_MAX_UTTERANCE_BYTES = 16000 * 2 * 60


KNOWN_VOICES = tuple(open_tts.VOICES)


def _tts_for(voice: str):
    return open_tts.get(voice)


_warm_tasks: set = set()


def _warm_done(t: "asyncio.Task") -> None:
    _warm_tasks.discard(t)
    if not t.cancelled() and t.exception() is not None:
        log.warning("TTS warm-up failed: %s", t.exception())


def _synth(text: str, voice: str, speed: float = 1.0) -> dict:
    eng = _tts_for(voice)
    if speed != 1.0:
        try:
            return eng.synthesize(text, voice, speed=speed)
        except TypeError:  # an engine without a speed control
            pass
    return eng.synthesize(text, voice)

SYSTEM = (
    "You are Nova, a warm voice assistant having a spoken "
    "conversation. Answer in one to three short sentences of at most ten "
    "words each, and start with a short sentence. No markdown, no lists. "
    "Sometimes you are given things you remember about this person; use them "
    "naturally when they matter and never read the list back."
)
_BOUNDARY = re.compile(r"[.!?]+[\s]")  # sentence end
_CLAUSE = re.compile(r"[,;:.!?]+[\s]")  # any clause break — safety fallback
_FIRST_MIN_WORDS = 3  # don't speak a 1-word fragment as the opener
_FIRST_MAX_WORDS = 14  # run-on opener: fall back to the last clause break


def _pop_clause(buf: str, first_done: bool) -> tuple[str, str] | None:
    """Return (sentence, remaining) if a speakable chunk is ready, else None.

    Chunks are whole sentences (the voice is synthesised sentence by sentence so
    prosody never breaks mid-clause; the system prompt keeps sentences short so
    the first one arrives fast). The opener must be at least `_FIRST_MIN_WORDS`
    words — a short "Hello!" is
    MERGED with the next sentence rather than blocking the flush. Safety valve:
    if the model ignores the prompt and runs past `_FIRST_MAX_WORDS` without a
    sentence end, the opener is cut at its last clause break instead.
    """
    for m in _BOUNDARY.finditer(buf):
        cut = m.end()
        sent = buf[:cut].strip()
        if not first_done and len(sent.split()) < _FIRST_MIN_WORDS:
            continue  # too short — keep the fragment, try a later boundary
        return sent, buf[cut:]
    if not first_done and len(buf.split()) >= _FIRST_MAX_WORDS:
        last = None
        for m in _CLAUSE.finditer(buf):
            if len(buf[:m.end()].split()) >= _FIRST_MIN_WORDS:
                last = m
        if last is not None:
            return buf[:last.end()].strip(), buf[last.end():]
    return None


def _u_bucket(words: int) -> str:
    """Utterance length bucket for the trace: U1 <= 2 words, U2 <= 8, U3 <= 20, U4 longer."""
    return "U1" if words <= 2 else "U2" if words <= 8 else "U3" if words <= 20 else "U4"


_ABBR = {"dr.", "mr.", "mrs.", "mt.", "st.", "vs.", "etc.", "e.g.", "i.e.", "u.s."}


def _first_sentence_words(text: str) -> int:
    """Words in the reply's first sentence (abbreviations don't end it)."""
    w = text.split()
    for i, x in enumerate(w):
        if (x[-1] in ".!?" and x.lower() not in _ABBR and i + 1 < len(w)
                and (w[i + 1][0].isupper() or w[i + 1][0].isdigit())):
            return i + 1
    return len(w)


# ---- conversational turn-taking helpers ----
# Words that acknowledge without taking the turn: said while Nova is talking
# they must NOT interrupt; said on their own afterwards, the
# pure fillers get no reply.
_BACKCHANNEL = {
    "mm", "mmm", "mhm", "mmhmm", "hmm", "hm", "uh", "uhhuh", "huh", "um", "umm", "erm",
    "ah", "aha", "ahh", "oh", "ooh", "yeah", "yep", "yup", "yes", "ya", "okay", "ok", "k",
    "right", "sure", "cool", "nice", "wow", "alright", "true", "exactly", "totally", "indeed",
    "got", "it", "i", "see", "gotcha", "fine", "great", "good", "and", "so", "well", "her",
}
_FILLER_ONLY = {"mm", "mmm", "mhm", "mmhmm", "hmm", "hm", "uh", "uhhuh", "huh", "um",
                "umm", "erm", "ah", "aha", "ahh", "oh", "ooh", "her"}
# One of these anywhere = a real interruption even if short ("wait", "stop it").
_INTERRUPT_WORDS = {"stop", "wait", "hold", "hang", "pause", "sorry", "excuse", "no", "nope",
                    "actually", "but", "hey", "nova", "listen", "enough", "quiet", "shut",
                    "what", "why", "how", "repeat", "again"}
BARGE_MIN_WORDS = int(os.environ.get("BARGE_MIN_WORDS", "2"))
BARGE_FIRST_EVAL_S = float(os.environ.get("BARGE_FIRST_EVAL_S", "0.15"))
BARGE_EVAL_EVERY_S = float(os.environ.get("BARGE_EVAL_EVERY_S", "0.10"))
ECHO_TAIL_S = float(os.environ.get("ECHO_TAIL_S", "0.35"))  # speaker→mic delay + room tail


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("-", ""))


def _echo_ratio(words: list[str], reference: str) -> float:
    """Share of `words` that also occur in what Nova was saying (her own echo)."""
    if not words:
        return 0.0
    ref = set(_words(reference))
    return sum(w in ref for w in words) / len(words)


def _is_filler_only(text: str) -> bool:
    w = _words(text)
    return bool(w) and all(x in _FILLER_ONLY for x in w)


def _barge_verdict(text: str, inflight: str, speech_ended: bool, elapsed_s: float) -> str:
    """'confirm' | 'dismiss' | 'wait' for a pending interruption."""
    w = _words(text)
    if not w:
        return "dismiss" if (speech_ended or elapsed_s > 2.0) else "wait"
    own = [x for x in w if x not in set(_words(inflight))]
    if ((_echo_ratio(w, inflight) >= 0.6 or _fuzzy_echo(text, inflight) >= 0.7)
            and not any(x in _INTERRUPT_WORDS for x in own)):
        return "dismiss"  # Nova hearing herself through the speaker
    if any(x in _INTERRUPT_WORDS for x in own):
        return "confirm"
    content = [x for x in own if x not in _BACKCHANNEL]
    if len(content) >= BARGE_MIN_WORDS:
        return "confirm"
    if speech_ended:
        return "confirm" if content and elapsed_s > 1.2 else "dismiss"
    return "confirm" if (content and elapsed_s > 1.5) else "wait"


ECHO_CORR = float(os.environ.get("ECHO_CORR", "0.45"))
# Echo canceller residual below this share of the mic RMS = nothing but echo.
ECHO_RESIDUAL = float(os.environ.get("ECHO_RESIDUAL", "0.35"))
# A content-word interruption needs this much voiced speech (short coughs and
# "uh-huh" make Moonshine invent words on 150-300 ms onsets).
BARGE_MIN_SPEECH_S = float(os.environ.get("BARGE_MIN_SPEECH_S", "0.35"))
# Onset echo correlation above this = Nova is loud in the mic while the user
# talks; content-word interruptions then need 3 words or 0.8 s of speech.
BARGE_LOUD_ECHO_CORR = float(os.environ.get("BARGE_LOUD_ECHO_CORR", "0.3"))
# One-word or two-word replies Smart Turn is unsure about but that are complete.
_Q_START = re.compile(r"^(what|what's|when|where|who|whom|whose|why|how|which|is|are|was|were|do|does|did|can|could|will|would|"
                      r"should|have|has|had|tell me|remind me)\b", re.I)
_DANGLING = {"and", "or", "but", "so", "because", "the", "a", "an", "to", "of", "with", "my", "our", "on", "in", "for",
             "about", "then", "um", "uh", "like", "if", "that", "this", "at", "from", "your"}


def _complete_question(t: str) -> bool:
    t = (t or "").strip()
    w = _words(t)
    return bool(t.endswith("?") and len(w) >= 4 and _Q_START.match(t) and w[-1] not in _DANGLING)


_SHORT_COMPLETE = {"yes", "no", "yeah", "yep", "nope", "okay", "ok", "sure", "thanks",
                   "thank", "you", "fine", "right", "correct", "exactly", "please", "bye",
                   "goodbye", "hello", "hi", "hey", "great", "cool", "nothing", "not", "really",
                   "absolutely", "definitely", "maybe", "true", "false", "done", "stop"}


def _fuzzy_echo(text: str, inflight: str) -> float:
    """Letter-level share of `text` found in what Nova said (catches garbled
    echo transcripts like 'Rototides' for 'prototypes')."""
    import difflib

    a = re.sub(r"[^a-z]", "", text.lower())
    b = re.sub(r"[^a-z]", "", inflight.lower())
    if len(a) < 4 or not b:
        return 0.0
    m = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(bl.size for bl in m.get_matching_blocks() if bl.size >= 3) / len(a)


def _echo_corr(mic: "np.ndarray", ref: "np.ndarray") -> float:
    """Max normalised cross-correlation of the mic segment against Nova's own
    playback over all lags (ref is longer than mic by the lag search range).
    High = the mic is just her voice coming back through the speaker; a user
    talking over her adds uncorrelated energy and pulls it down (double talk)."""
    import numpy as np
    from scipy.signal import fftconvolve

    if mic.size < 1600 or ref.size < mic.size or not np.any(ref):
        return 0.0
    mic = mic - mic.mean()
    num = fftconvolve(ref, mic[::-1], mode="valid")
    e = np.concatenate([[0.0], np.cumsum(ref.astype(np.float64) ** 2)])
    win = e[mic.size:] - e[:-mic.size]
    den = np.sqrt(np.maximum(win, 1e-9) * float(np.dot(mic, mic)) + 1e-12)
    return float(np.max(np.abs(num) / den))


def _subtract_echo(mic: "np.ndarray", ref: "np.ndarray") -> tuple["np.ndarray", float]:
    """Single-tap echo canceller: find the lag where Nova's playback best
    explains the mic, subtract the least-squares-scaled copy. Leaves the
    user's own voice when both talk at once (double talk). `ref` covers the
    mic span plus the lag search range before it. Returns (residual, corr)."""
    import numpy as np
    from scipy.signal import fftconvolve

    if mic.size < 1600 or ref.size < mic.size or not np.any(ref):
        return mic, 0.0
    num = fftconvolve(ref, mic[::-1], mode="valid")
    e = np.concatenate([[0.0], np.cumsum(ref.astype(np.float64) ** 2)])
    win = e[mic.size:] - e[:-mic.size]
    den = np.sqrt(np.maximum(win, 1e-9) * float(np.dot(mic, mic)) + 1e-12)
    corr = np.abs(num) / den
    lag = int(np.argmax(corr))
    seg = ref[lag:lag + mic.size]
    g = float(np.dot(mic, seg) / max(float(np.dot(seg, seg)), 1e-9))
    return (mic - g * seg).astype(np.float32), float(corr[lag])


def _pcm_from_wav(b: bytes) -> "np.ndarray":
    import io

    import numpy as np
    import soundfile as sf

    try:
        a, sr = sf.read(io.BytesIO(b), dtype="float32")
    except Exception:  # noqa: BLE001
        return np.zeros(0, dtype=np.float32)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        n = int(len(a) * SR / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
    return a


def _wav_seconds(b: bytes) -> float:
    """Duration of a PCM WAV blob (walks the RIFF chunks)."""
    try:
        sr = struct.unpack_from("<I", b, 24)[0]
        bps = struct.unpack_from("<H", b, 34)[0] // 8 or 2
        ch = struct.unpack_from("<H", b, 22)[0] or 1
        off = 12
        while off + 8 <= len(b):
            cid, size = b[off:off + 4], struct.unpack_from("<I", b, off + 4)[0]
            if cid == b"data":
                return min(size, len(b) - off - 8) / (sr * bps * ch)
            off += 8 + size + (size & 1)
    except Exception:  # noqa: BLE001
        pass
    return 0.0


async def _sentence_stream(make_iter, stop: asyncio.Event | None = None,
                           on_first_delta=None):
    """Async-yield sentences as an LLM streams deltas (first sentence ASAP).
    `make_iter` returns a blocking iterator of text deltas. [stop] (checked
    per delta — Event.is_set() is thread-safe to read) aborts the iterator on
    barge-in so the model stops decoding the abandoned reply instead of
    finishing it and queueing the next turn behind it."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def worker() -> None:
        buf = ""
        first_done = False  # first chunk flushes on any clause break (lower latency)
        seen_token = False
        try:
            for delta in make_iter():
                if stop is not None and stop.is_set():
                    break  # closes the generator -> cancels the decode
                if not seen_token and delta:
                    seen_token = True
                    if on_first_delta is not None:
                        on_first_delta()  # TTFT mark (runs on the worker thread)
                buf += delta
                while True:
                    popped = _pop_clause(buf, first_done)
                    if popped is None:
                        break
                    sent, buf = popped
                    if sent:
                        loop.call_soon_threadsafe(q.put_nowait, sent)
                        first_done = True
            if buf.strip():
                loop.call_soon_threadsafe(q.put_nowait, buf.strip())
        except Exception as e:  # noqa: BLE001
            loop.call_soon_threadsafe(q.put_nowait, _Err(str(e)))
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    loop.run_in_executor(None, worker)
    while True:
        item = await q.get()
        if item is None:
            return
        if isinstance(item, _Err):
            raise RuntimeError(item.msg)
        yield item


class _Err:
    def __init__(self, msg: str) -> None:
        self.msg = msg


def _sentence_stream_sync(make_iter):
    """Synchronous mirror of `_sentence_stream` for benchmarks/CLI.

    Yields ("first_token", delta) once, then ("clause", sentence) per flush using
    the SAME first-clause-on-any-break / then-sentence-boundary rule as the WS
    path, so measured latency matches production behaviour.
    """
    buf = ""
    first_token_sent = False
    first_done = False
    for delta in make_iter():
        if not first_token_sent and delta:
            first_token_sent = True
            yield ("first_token", delta)
        buf += delta
        while True:
            popped = _pop_clause(buf, first_done)
            if popped is None:
                break
            sent, buf = popped
            if sent:
                yield ("clause", sent)
                first_done = True
    if buf.strip():
        yield ("clause", buf.strip())


async def voice_ws(ws: WebSocket) -> None:
    await ws.accept()
    td = TurnDetector()
    cfg = {"echo": False, "voice": open_tts.DEFAULT_VOICE,
           "language": offline_engine.STT_LANGUAGE, "profile": memory_store.DEFAULT_PROFILE}
    state = "listening"
    collecting = False
    utterance = bytearray()
    cancel = asyncio.Event()
    proc: asyncio.Task | None = None
    # Eager STT: decode kicked off at the soft-silence mark so it overlaps the
    # semantic vote / remaining hangover. Staleness = any speech after launch,
    # handled by _drop_eager("speech resumed") in the frame loop.
    eager_stt: asyncio.Task | None = None
    import weakref as _wr
    dropped_eager: "_wr.WeakSet" = _wr.WeakSet()  # eager transcripts already thrown away (no speculation)
    # Per-turn latency trace (turn_trace.TurnTrace): created at speech_start,
    # handed to process() at commit, logged to logs/turns.jsonl.
    turn_seq = 0
    trace: TurnTrace | None = None
    # Pre-roll: the VAD needs ~200 ms of speech before speech_start fires, so
    # without this the first syllables never reached STT. Keep the last 400 ms
    # of raw audio while idle and prepend it to a new utterance.
    _PREROLL_BYTES = SR * 2 * 400 // 1000
    preroll = bytearray()
    # Rolling conversation memory for the LLM ([user, assistant, ...]).
    history: list[dict] = []
    # Client playback model: the server paces nothing, so it estimates when the
    # client finishes PLAYING what was sent (chunks play back to back). Nova is
    # "speaking" until then (+ECHO_TAIL_S), so late interruptions are barge-ins
    # and her own echo is never taken as a new user turn.
    audible_until = 0.0
    schedule: list[tuple[str, float, float]] = []   # (sentence, start, end) of this reply
    last_reply = {"text": "", "end": 0.0}           # for echo-of-last-reply filtering
    cur_user = {"text": ""}
    utter_origin = {"fresh": False}  # True while the current recording began with a fresh speech_start
    last_clip = {"pcm": b""}                        # for continuation merges
    pending: dict | None = None                     # pending barge-in under evaluation
    forced_events: list[str] = []                   # replayed on the next frame
    import collections as _c
    mic_ring: "_c.deque" = _c.deque(maxlen=200)      # (arrival_t, int16 frame), ~6 s
    pending_buf_snapshot: list[bytes] = [b""]
    ref_audio: list[tuple[float, object]] = []      # (play_start, 16 kHz float) of Nova's audio
    # LiteRT-LM v0.17 raw session on CPU + speculative prefill (llm_session.py). The history keeps the
    # exact user chunk (with memory) the model saw and is not trimmed (the session resets itself when full).
    import llm_session

    sess_llm = await asyncio.to_thread(llm_session.get, SYSTEM)
    SESSION_HIST = True
    spec_cache: dict = {}  # eager transcript -> library search result, reused at commit
    conv = {"profile": None, "sid": None}  # this connection's saved session (conv_store.py)

    def _conv_session() -> int | None:
        p = cfg.get("profile")
        if conv["profile"] != p:
            if conv["sid"] is not None:
                conv_store.end_session(conv["profile"], conv["sid"])
            conv.update(profile=p, sid=conv_store.start_session(p, cfg.get("voice", ""), "cpu"))
        return conv["sid"]

    def sysp() -> str:
        return personas.runtime_system(cfg.get("profile"))

    async def _predict(sem: SemanticEndpointer) -> float:
        """Smart Turn inference on the current utterance, timed into the trace."""
        t0 = now()
        p = await asyncio.to_thread(sem.predict_pcm16, bytes(utterance))
        if trace is not None:
            trace.smart_turn_ms.append(round((now() - t0) * 1000.0, 1))
            trace.smart_turn_p.append(round(p, 3))
        return p

    def _drop_eager(reason: str) -> None:
        nonlocal eager_stt
        if eager_stt is not None:
            log.info("eager STT wasted (%s)", reason)
            dropped_eager.add(eager_stt)
            eager_stt.add_done_callback(lambda t: t.exception())
        eager_stt = None

    closed = {"v": False}

    async def send(obj: dict) -> None:
        # The client may disconnect mid-reply (tab closed, page reload); a late
        # state/trace message must not raise "send after close" inside process().
        if closed["v"]:
            return
        try:
            await ws.send_text(json.dumps(obj))
        except (RuntimeError, WebSocketDisconnect):
            closed["v"] = True
            log.info("client gone; dropping %s message", obj.get("type"))

    async def send_bin(b: bytes) -> None:
        if closed["v"]:
            return
        try:
            await ws.send_bytes(b)
        except (RuntimeError, WebSocketDisconnect):
            closed["v"] = True
            log.info("client gone; dropping audio chunk")

    loop = asyncio.get_running_loop()

    async def stage(name: str, status: str, **kw) -> None:
        """Live progress for the web page: which step is running now, and how long it took."""
        await send({"type": "stage", "stage": name, "status": status, **kw})

    def stage_threadsafe(name: str, status: str, **kw) -> None:
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(stage(name, status, **kw)))

    partial_state = {"task": None, "last": 0.0, "seq": 0}

    async def _partial_tick(seq: int) -> None:
        try:
            txt = await asyncio.to_thread(offline_engine.stt_partial, bytes(utterance), SR)
            if txt and seq == partial_state["seq"] and collecting and eager_stt is None:
                await send({"type": "partial", "text": txt})
        except Exception:  # noqa: BLE001 — display only
            pass

    async def set_state(s: str) -> None:
        nonlocal state
        state = s
        await send({"type": "state", "value": s})

    async def speak(text: str, tr: TurnTrace | None = None) -> None:
        if cancel.is_set():
            return
        t0 = now()
        r = await asyncio.to_thread(_synth, text, cfg["voice"], cfg.get("speed", 1.0))
        audio = r["wav"]
        if cancel.is_set():
            return
        if tr is not None:
            if tr.tts_first_ms is None:
                tr.tts_first_ms = round((now() - t0) * 1000.0, 1)
            if tr.first_audio_t is None:
                tr.first_audio_t = now()
        await send_bin(audio)

    async def process(audio: bytes, pre_stt: asyncio.Task | None = None,
                      trace: TurnTrace | None = None) -> None:
        tr = trace if trace is not None else TurnTrace(-1)
        finished = False
        cur_user.clear()
        cur_user["text"] = ""  # the previous turn is settled; nothing of it may be re-logged by an interrupt
        try:
            await set_state("processing")
            t0 = now()
            if pre_stt is not None:
                # Eager STT started at the soft-silence mark — the decode has
                # been overlapping the semantic vote / remaining hangover.
                text = await pre_stt
                tr.stt_ms = round((now() - (tr.eager_start_t or t0)) * 1000.0, 1)
                if _looks_degenerate(text, len(audio) / 32000.0):
                    # Small decoders can loop on a buffer cut mid-word; the
                    # full clip at commit is authoritative (~30-90 ms more).
                    log.info("eager STT degenerate (%r) — re-decoding at commit", text[:60])
                    tr.stt_eager_used = False
                    text = await asyncio.to_thread(offline_engine.stt_pcm, audio, SR, cfg["language"])
            else:
                text = await asyncio.to_thread(offline_engine.stt_pcm, audio, SR, cfg["language"])
                tr.stt_ms = round((now() - t0) * 1000.0, 1)
            text = personas.apply_aliases(cfg.get("profile"), (text or "").strip())
            tr.transcript_t = now()
            tr.transcript = text
            # Benchmark labels first, so ignored turns (e.g. an empty STT) still
            # land in the right bucket in logs/turns.jsonl.
            tr.utt_words = len(text.split())
            tr.u_bucket = cfg.get("u_bucket")
            tr.clip_id = cfg.get("clip_id")
            if cancel.is_set():  # user kept talking while we transcribed
                tr.error = "superseded"
                tr.cancelled = True
                tr.finish()
                finished = True
                return
            reason = None
            if not text:
                reason = "empty_transcript"
            elif _is_filler_only(text):
                reason = "filler_only"
            elif (now() - last_reply["end"] < 2.0
                  and _echo_ratio(_words(text), last_reply["text"]) >= 0.6):
                reason = "echo_of_reply"
            if reason:
                log.info("turn ignored (%s): %r", reason, text[:60])
                tr.error = reason
                tr.finish()
                finished = True
                await set_state("listening")
                return
            cur_user.clear()
            cur_user["text"] = text  # prompt/pre/stage of the previous turn must not leak into this one
            tr.u_bucket = tr.u_bucket or _u_bucket(tr.utt_words)
            tr.llm_backend = offline_engine.LLM_BACKEND
            schedule.clear()
            await send({"type": "transcript", "text": text})
            # Child-safety guard (kids template): runs before the model; a hit speaks a fixed safe reply.
            child = personas.is_child(cfg.get("profile"))
            flag = safety.check_input(text) if child else None
            if not flag and not cfg["echo"] and safety.check_health(text):
                flag = "health"  # red-flag symptoms: a fixed "call someone now" reply, never the model's opinion
            if flag:
                tr.safety = flag
                if flag == "personal":
                    tr.transcript = "[private details redacted]"
                cur_user["text"] = ""  # a barge-in must not put the flagged words into history or the KV
                log.info("child-safety guard: %s", flag)
            elif child and safety.blocks_memory(text):
                tr.transcript = "[private details redacted]"  # e.g. where a child lives: not kept in the saved history
            # Memory: embed the transcript, exact scan of this profile's facts, inject
            # the top facts into the user chunk (≤ ~60 tokens, so the prefill block
            # stays flat). Runs in a thread; a few ms on a laptop CPU.
            mem_facts: list[dict] = []
            if MEMORY_ON and not cfg["echo"] and not flag:
                await stage("memory", "start", profile=cfg.get("profile"))
                try:
                    mem_facts, mem_ms = await asyncio.to_thread(
                        memory_store.get(cfg.get("profile")).retrieve, text)
                    tr.memory_ms, tr.memory_hits = round(mem_ms, 2), len(mem_facts)
                    tr.memory_facts = [f["text"] for f in mem_facts]
                    await send({"type": "memory", "profile": cfg.get("profile"),
                                "facts": mem_facts, "ms": round(mem_ms, 2)})
                    await stage("memory", "done", ms=round(mem_ms, 1), hits=len(mem_facts))
                except Exception as e:  # noqa: BLE001 — memory must never break a turn
                    log.warning("memory retrieve failed: %s", e)
            prompt = compose_prompt(text, mem_facts)
            tool_pick = None
            if TOOLS_ON and not cfg["echo"] and not flag:
                tool_pick = tools.route(cfg.get("profile"), text, child)
                if tool_pick:
                    log.info("tool (rule): %s %s", tool_pick[0], tool_pick[1])
            rag = None
            cur_user.pop("pre", None)
            if RAG_ON and not cfg["echo"] and not flag and not tool_pick:
                await stage("rag", "start", library=personas.PERSONAS.get(cfg.get("profile"), {}).get("library", "documents"))
                if spec_cache.get("text") == text and spec_cache.get("profile") == cfg.get("profile"):
                    rag = spec_cache.get("rag")
                else:
                    rag = await asyncio.to_thread(_rag_lookup, cfg.get("profile"), text)
                if rag is not None:
                    tr.rag_ms, tr.rag_gate = rag["ms"].get("total"), bool(rag.get("gate"))
                    srcs = [{"title": p.get("short_title") or p.get("title"), "page": p.get("page_label") or p.get("page"),
                             "section": p.get("section"), "text": p.get("text", "")[:600], "score": round(float(p.get("score", 0)), 3)}
                            for p in rag.get("passages", [])]
                    tr.rag_sources = [f"{x['title']} " + (f"p. {x['page']}" if str(x['page']).isdigit() else str(x['page'])) for x in srcs]
                    await stage("rag", "done", ms=round(rag["ms"].get("total", 0), 1), hits=len(srcs), gate=bool(rag.get("gate")),
                                top=round(float(rag.get("top_score", 0)), 3))
                    if _use_library(rag, mem_facts):
                        await send({"type": "sources", "items": srcs, "leadin": True})
                else:
                    await stage("rag", "done", ms=0, hits=0, gate=False)
            use_rag = bool(_use_library(rag, mem_facts))
            tool_turn: dict = {"name": None, "call": "", "lead": "", "answer": "", "user2": None}
            tctx = {"profile": cfg.get("profile"), "child": child}

            def send_threadsafe(obj: dict) -> None:
                loop.call_soon_threadsafe(lambda: asyncio.ensure_future(send(obj)))

            def _apply_device(res: dict) -> None:
                act = res.get("action")
                if act in ("slower", "faster", "normal_speed"):
                    cfg["speed"] = {"slower": 0.85, "faster": 1.15, "normal_speed": 1.0}[act]
                elif act == "voice":
                    v = res.get("voice")
                    if v not in KNOWN_VOICES or not open_tts.available(v):
                        log.warning("voice switch to %r ignored (not installed)", v)
                        return
                    cfg["voice"] = v
                send_threadsafe({"type": "device", "action": act, "voice": cfg.get("voice") if act == "voice" else None,
                                 "speed": cfg.get("speed", 1.0)})

            def _tool_answer(base_h, base_user, name, args, call_text, _sys, via):
                """Speak a lead-in while a slow (online) tool runs, run it, then answer from its result."""
                tool_turn.update(name=name, call=call_text)
                t = tools.TOOLS[name]
                tr.tool, tr.tool_via, tr.tool_online = name, via, t.online
                stage_threadsafe("tool", "start", tool=name, title=tools.card(name, args, {}).get("title"), online=t.online)
                if t.online and tools.online():
                    lead = tools.leadin(name, args, tctx)
                    if lead:
                        tool_turn["lead"] = lead
                        yield lead + " "
                if cancel.is_set():
                    return
                res = tools.run(name, args, tctx)
                tr.tool_ms = res.get("ms")
                if res.get("ok") and name == "device_control":
                    _apply_device(res)
                send_threadsafe({"type": "tool", "via": via, **tools.card(name, args, res)})
                stage_threadsafe("tool", "done", ms=res.get("ms"), tool=name, ok=bool(res.get("ok")), online=t.online)
                if call_text:
                    h2 = base_h + [{"role": "user", "content": base_user}, {"role": "assistant", "content": call_text}]
                    user2 = tools.note(name, res).strip()
                else:
                    h2, user2 = base_h, base_user + tools.note(name, res)
                tool_turn["user2"] = user2
                cur_user["prompt"] = user2  # an interruption logs what the model actually answered from
                if cancel.is_set():
                    return
                head, said_any = "", False
                for d in sess_llm.stream(h2, user2, offline_engine.LLM_MAX_TOKENS, system=_sys):
                    tool_turn["answer"] += d
                    if not said_any:
                        head += d
                        if len(head.strip()) < 6:
                            continue
                        if re.match(r"\s*(<\|tool_call>|call:)", head):  # the model tried another call: speak a plain confirmation
                            log.info("tool answer was another call; using a plain confirmation")
                            tool_turn["answer"] = tools.confirm(name, res)
                            yield tool_turn["answer"]
                            return
                        said_any, d = True, head
                    d = _TAG_RX.sub("", d)
                    if d:
                        yield d
                if not said_any and head.strip():
                    if re.match(r"\s*(<\|tool_call>|call:)", head):
                        tool_turn["answer"] = tools.confirm(name, res)
                        yield tool_turn["answer"]
                    else:
                        yield _TAG_RX.sub("", head)

            def _plain_or_tool(_h, _prompt, _sys):
                """A normal answer, unless Gemma 4 starts its reply with a native tool call."""
                buf, said = "", ""
                it = sess_llm.stream(_h, _prompt, offline_engine.LLM_MAX_TOKENS, system=_sys)
                mode = None
                try:
                    for d in it:
                        if mode == "text":
                            if "<|tool_call>" in d:  # a call after some words ("Let me check. <|tool_call>...")
                                pre, rest = d.split("<|tool_call>", 1)
                                if pre:
                                    said += pre
                                    yield pre
                                buf, mode = "<|tool_call>" + rest, "call"
                                if "<tool_call|>" in buf:
                                    break
                                continue
                            said += d
                            yield _TAG_RX.sub("", d)
                            continue
                        buf += d
                        if mode == "call":
                            if "<tool_call|>" in buf:
                                break
                            continue
                        s = buf.lstrip()
                        if s.startswith("<|tool_call>"):
                            mode = "call"
                            if "<tool_call|>" in s:
                                break
                        elif s and not "<|tool_call>".startswith(s):
                            mode, said = "text", buf
                            yield _TAG_RX.sub("", buf)
                finally:
                    if mode == "call":
                        it.close()  # stop decoding: the tool runs next
                if mode != "call":
                    if mode is None and buf:
                        yield _TAG_RX.sub("", buf)
                    return
                parsed = tools.parse_call(buf)
                call = (said + buf[: buf.find("<tool_call|>") + len("<tool_call|>")] if "<tool_call|>" in buf else said + buf).strip()
                if parsed and not tools.model_call_ok(parsed[0], text):
                    # nobody asked for that (e.g. a note for "my cat's name is Whiskers"): answer normally instead
                    log.info("model tool call ignored (not requested): %s %s", parsed[0], parsed[1])
                    tool_turn.update(name="ignored")
                    tool_turn["user2"] = _prompt + "\n(No tool is needed for this. Answer directly.)"
                    it2 = sess_llm.stream(_h, tool_turn["user2"], offline_engine.LLM_MAX_TOKENS, system=_sys)
                    head, live = "", False
                    try:
                        for d in it2:
                            tool_turn["answer"] += d
                            if not live:
                                head += d
                                s2 = head.lstrip()
                                if _CALL_START_RX.match(s2):
                                    # the model insists on the tool: never speak call markup
                                    tool_turn["answer"] = _NO_TOOL_REPLY
                                    yield _NO_TOOL_REPLY
                                    return
                                if len(s2) < 12 and any(p.startswith(s2) for p in _CALL_PREFIXES):
                                    continue  # might still become a call
                                live, d = True, head
                            d = _TAG_RX.sub("", d)
                            if d:
                                yield d
                        if not live and head.strip():
                            yield _TAG_RX.sub("", head)
                    finally:
                        it2.close()
                    return
                if not parsed or parsed[0] not in tools.allowed(tctx["profile"], child):
                    log.info("model tool call not allowed or unreadable: %r", buf[:120])
                    tool_turn.update(name="blocked", call=call, user2="(That tool is not available here. Say so in one short sentence.)",
                                     answer="Sorry, I can't do that one here.")
                    yield "Sorry, I can't do that one here."
                    return
                log.info("tool (model): %s %s", parsed[0], parsed[1])
                yield from _tool_answer(_h, _prompt, parsed[0], parsed[1], call, _sys, "model")
            if not use_rag and not flag and not cfg["echo"]:
                prompt = prompt + _not_found_note(cfg.get("profile"), text, rag, mem_facts)
            if rag and rag.get("gate") and not _use_library(rag, mem_facts):
                await stage("rag", "done", ms=round(rag["ms"].get("total", 0), 1), hits=0, gate=False, reason="memory matched better")
            if use_rag:
                lookup, prompt = _rag_user_chunks(text, rag, mem_facts, cfg.get("profile"))
                cur_user["pre"] = [{"role": "user", "content": lookup}]
            if not flag:
                cur_user["prompt"] = prompt
            tr.profile, tr.voice = cfg.get("profile"), cfg.get("voice")
            if not cancel.is_set():  # a superseded turn must not flip the state back from "listening"
                await set_state("speaking")
            if flag:
                if flag == "health":
                    pp = personas.PERSONAS.get(cfg.get("profile"), {})
                    full = safety.health_reply(pp.get("name", ""), pp.get("emergency_contacts", ""))
                else:
                    full = safety.REPLIES[flag]
                await send({"type": "safety", "category": flag})
                await send({"type": "sentence", "text": full})
                await speak(full, tr)
                await send({"type": "reply", "text": full})
            elif cfg["echo"]:
                await send({"type": "reply", "text": text})
                await speak(text, tr)
            else:
                if tool_pick:
                    _h0, _sys0 = list(history), sysp()
                    make_iter = lambda: _tool_answer(_h0, prompt, tool_pick[0], tool_pick[1], "", _sys0, "rule")  # noqa: E731
                elif use_rag:
                    _h, _sys = list(history), sysp()
                    rag_turn = {"filler": "", "answer": ""}
                    cur_user["rag_turn"] = rag_turn

                    def _chained():
                        # 1) short lead-in (already speculatively prefilled), spoken at once
                        for d in sess_llm.stream(_h, lookup, 28, system=_sys):
                            m = re.search(r"[.!?](\s|$)", rag_turn["filler"] + d)
                            if m:  # first sentence complete: keep it, cut the rest (cancels the decode)
                                keep = (rag_turn["filler"] + d)[: m.end()].rstrip()
                                piece = keep[len(rag_turn["filler"]):]
                                rag_turn["filler"] = keep
                                if piece:
                                    yield piece
                                break
                            rag_turn["filler"] += d
                            yield d
                        f = rag_turn["filler"].strip()
                        if f and f[-1] not in ".!?":
                            yield "."
                            rag_turn["filler"] = f + "."
                        yield " "
                        if cancel.is_set():  # interrupted during the lead-in: never prefill the notes
                            return
                        rag_turn["stage"] = 2
                        # 2) the answer from the notes; its prefill runs while the lead-in plays
                        h2 = _h + [{"role": "user", "content": lookup},
                                   {"role": "assistant", "content": rag_turn["filler"].strip()}]
                        head, passed = "", False
                        for d in sess_llm.stream(h2, prompt, offline_engine.LLM_MAX_TOKENS, system=_sys):
                            rag_turn["answer"] += d
                            if passed:
                                yield d
                                continue
                            head += d
                            m = re.search(r"[.!?](\s|$)", head)
                            if not m:
                                continue
                            first, rest = head[: m.end()], head[m.end():]
                            passed = True
                            if _LEADIN_RX.match(first.strip()) and not re.search(r"\d", first):
                                log.info("dropped a second lead-in: %r", first.strip())  # already said one
                                if rest:
                                    yield rest
                            else:
                                yield head
                        if not passed and head:
                            yield head
                    make_iter = _chained
                elif TOOLS_ON:
                    _h1, _sys1 = list(history), sysp()
                    make_iter = lambda: _plain_or_tool(_h1, prompt, _sys1)  # noqa: E731
                else:
                    _h, _sys = list(history), sysp()
                    make_iter = lambda: sess_llm.stream(_h, prompt, offline_engine.LLM_MAX_TOKENS, system=_sys)  # noqa: E731
                full = ""
                if cancel.is_set():  # superseded before the model started: no prefill, keep the speculation
                    make_iter = lambda: iter(())  # noqa: E731
                tr.llm_start_t = now()
                await stage("llm", "start")

                def _first_token() -> None:
                    if tr.llm_first_token_t is None:
                        tr.llm_first_token_t = now()
                        stage_threadsafe("llm", "first", ms=round((tr.llm_first_token_t - tr.llm_start_t) * 1000))
                # Synth-ahead pipeline: clauses flow through a synth worker into
                # an ordered sender, so clause n+1 is synthesizing while clause
                # n's audio is on the wire (a serial `await speak(sent)` loop
                # would stall synthesis behind every send). clause_q is
                # unbounded (tiny strings — never worth a producer deadlock);
                # the bounded audio_q is what stops a barge-in from wasting
                # more than ~2 clauses of synth work.
                clause_q: asyncio.Queue = asyncio.Queue()
                audio_q: asyncio.Queue = asyncio.Queue(maxsize=2)

                async def synth_worker() -> None:
                    try:
                        while True:
                            sent = await clause_q.get()
                            if sent is None:
                                return
                            if cancel.is_set():
                                continue  # drain remaining clauses cheaply
                            if not re.search(r"[A-Za-z0-9]", sent):
                                continue  # e.g. a lone "—" copied from the interrupted-reply marker
                            ts0 = now()
                            r = await asyncio.to_thread(
                                _synth, sent, cfg["voice"], cfg.get("speed", 1.0))
                            chunk = r["wav"]
                            if tr.tts_first_ms is None:
                                tr.tts_first_ms = round((now() - ts0) * 1000.0, 1)
                            await audio_q.put((sent, chunk))
                    finally:
                        # Safe: the sender never dies (it swallows send errors
                        # and keeps draining), so this put cannot block forever
                        # and a dead sender can never deadlock the synth worker.
                        await audio_q.put(None)

                async def sender() -> None:
                    nonlocal audible_until
                    while True:
                        item = await audio_q.get()
                        if item is None:
                            return
                        sent_text, chunk = item
                        if cancel.is_set():
                            continue  # drained, not sent
                        if tr.first_audio_t is None:
                            tr.first_audio_t = now()
                            await stage("tts", "done", ms=round(tr.tts_first_ms or 0))
                        start = max(now() + 0.03, audible_until)
                        end = start + _wav_seconds(chunk)
                        schedule.append((sent_text, start, end))
                        ref_audio.append((start, _pcm_from_wav(chunk)))
                        del ref_audio[:-12]
                        audible_until = end
                        try:
                            await send({"type": "sentence", "text": sent_text})
                            await send_bin(chunk)
                        except Exception:  # noqa: BLE001 — client gone
                            # Keep DRAINING so synth_worker can never block on
                            # a full audio_q (the deadlock that leaked a task
                            # tree per dropped connection); cancel stops
                            # further synthesis cheaply.
                            cancel.set()

                synth_task = asyncio.create_task(synth_worker())
                send_task = asyncio.create_task(sender())
                try:
                    async for sent in _sentence_stream(
                            make_iter, stop=cancel, on_first_delta=_first_token):
                        if cancel.is_set() or synth_task.done():
                            break
                        if tr.llm_first_sentence_t is None:
                            tr.llm_first_sentence_t = now()
                            tr.reply_first = sent
                            await stage("llm", "done", ms=round((tr.llm_first_sentence_t - tr.llm_start_t) * 1000))
                            await stage("tts", "start")
                        if child and safety.check_output(sent):
                            log.info("child-safety output guard replaced a sentence")
                            sent = safety.OUTPUT_REPLACEMENT
                            if use_rag:
                                rag_turn["guarded"] = True
                        tr.n_sentences += 1
                        full += sent + " "
                        clause_q.put_nowait(sent)
                finally:
                    clause_q.put_nowait(None)
                    # Wait for BOTH workers whatever happened — awaiting them
                    # sequentially skipped the sender when synth raised, which
                    # let audio frames race the error/state messages on the
                    # same websocket.
                    done = await asyncio.gather(
                        synth_task, send_task, return_exceptions=True)
                    for r in done:
                        if (isinstance(r, BaseException)
                                and not isinstance(r, asyncio.CancelledError)):
                            raise r
                tr.first_chunk_words = len(tr.reply_first.split()) if tr.reply_first else None
                if full and tr.r_first_sentence_words is None:
                    tr.r_first_sentence_words = _first_sentence_words(full)
                if full and not cancel.is_set():
                    await send({"type": "reply", "text": full.strip()})
                if full and not cancel.is_set():
                    if tool_turn.get("user2") and SESSION_HIST:
                        tr.filler = tool_turn["lead"]
                        if tool_turn["call"]:  # the model's own call stays in its context
                            history.append({"role": "user", "content": prompt})
                            history.append({"role": "assistant", "content": tool_turn["call"]})
                        history.append({"role": "user", "content": tool_turn["user2"]})
                        history.append({"role": "assistant", "content": tool_turn["answer"].strip()})
                    elif use_rag and SESSION_HIST:
                        tr.filler = rag_turn["filler"].strip()
                        answer = rag_turn["answer"].strip()
                        if rag_turn.get("guarded"):  # history holds what was spoken, not the unsafe text
                            answer = full.strip()[len(rag_turn["filler"].strip()):].strip()
                        history.append({"role": "user", "content": lookup})
                        history.append({"role": "assistant", "content": rag_turn["filler"].strip()})
                        history.append({"role": "user", "content": prompt})
                        history.append({"role": "assistant", "content": answer})
                    else:
                        history.append({"role": "user", "content": prompt if SESSION_HIST else text})
                        history.append({"role": "assistant", "content": full.strip()})
                    if not SESSION_HIST:
                        del history[:-2 * offline_engine.LLM_HISTORY_TURNS]
                    cur_user["logged"] = True  # a later barge-in only trims the reply to what was heard
                if full:
                    last_reply["text"] = full.strip()
                if full and not cancel.is_set() and MEMORY_ON and not cfg["echo"] and not flag \
                        and tool_turn.get("name") not in tools.TOOLS and not (child and safety.blocks_memory(text)):
                    # Rule-based fact extraction from what the user said (immediate path;
                    # LLM extraction at idle is a later flag). Never on the critical path:
                    # the reply audio is already sent.
                    try:
                        store = memory_store.get(cfg.get("profile"))
                        saved = [f for f in extract_facts(text) if store.add(f) is not None]
                        if saved:
                            await send({"type": "memory_saved", "profile": cfg.get("profile"), "facts": saved})
                    except Exception as e:  # noqa: BLE001
                        log.warning("memory save failed: %s", e)
            tr.audio_end_t = now()
            tr.cancelled = cancel.is_set()
            if not cfg["echo"]:
                tr.reply = (locals().get("full") or "").strip()
            d = tr.finish()
            try:
                conv_store.add_turn(cfg.get("profile"), _conv_session(), d)
            except Exception as e:  # noqa: BLE001 — history must never break a turn
                log.warning("history save failed: %s", e)
            finished = True
            if not cancel.is_set():
                await send({"type": "trace", **d})
                await send({"type": "audio_end"})
                # Stay "speaking" while the client is still playing (plus the
                # speaker→mic echo tail); a barge-in sets `cancel` and wakes us.
                rest = audible_until + ECHO_TAIL_S - now()
                if rest > 0:
                    try:
                        await asyncio.wait_for(cancel.wait(), timeout=rest)
                    except asyncio.TimeoutError:
                        pass
                last_reply["end"] = now()
        except Exception as e:  # noqa: BLE001
            if not finished:
                tr.error = str(e)
                tr.cancelled = cancel.is_set()
                tr.finish()
                finished = True
            log.warning("turn failed: %s", e, exc_info=True)
            try:
                await send({"type": "error", "message": str(e)})
            except Exception:  # noqa: BLE001 — client already gone
                log.info("error not delivered (socket closed): %s", e)
        finally:
            if not finished:  # e.g. task cancelled on disconnect
                tr.cancelled = True
                tr.finish()
            cancel.clear()
            # NOTE: no td.reset() here — the detector was already re-armed by
            # turn_committed() at commit (or by the barge-in path), and this
            # finally can land 1-2s later, mid-barge-in-utterance; resetting
            # then wiped in_speech and the new turn never endpointed.
            if state != "listening":
                await set_state("listening")

    async def greet() -> None:
        """Nova greets the person once when the conversation starts (an opening line is fine
        the first time, never as filler). Runs like a reply: echo-gated and interruptible."""
        nonlocal audible_until
        name = ""
        try:
            for f in memory_store.get(cfg.get("profile")).list():
                m = re.match(r"User's name is ([A-Z][a-z]+)\.", f["text"])
                if m:
                    name = m.group(1)
                    break
        except Exception:  # noqa: BLE001
            pass
        text = personas.greeting(cfg.get("profile"), name)
        try:
            await set_state("speaking")
            r = await asyncio.to_thread(_synth, text, cfg["voice"], cfg.get("speed", 1.0))
            if cancel.is_set():
                return
            chunk = r["wav"]
            start = max(now() + 0.03, audible_until)
            end = start + _wav_seconds(chunk)
            schedule.clear()
            schedule.append((text, start, end))
            ref_audio.append((start, _pcm_from_wav(chunk)))
            audible_until = end
            last_reply["text"] = text
            # Gemma must know she already greeted, or her first reply opens with "Hello!" again.
            if not history:
                history.append({"role": "user", "content": "(The conversation starts.)"})
                history.append({"role": "assistant", "content": text})
            await send({"type": "sentence", "text": text})
            await send_bin(chunk)
            await send({"type": "reply", "text": text})
            await send({"type": "audio_end"})
            rest = audible_until + ECHO_TAIL_S - now()
            if rest > 0:
                try:
                    await asyncio.wait_for(cancel.wait(), timeout=rest)
                except asyncio.TimeoutError:
                    pass
            last_reply["end"] = now()
        finally:
            cancel.clear()
            if state != "listening":
                await set_state("listening")

    def _inflight_text(t0: float) -> str:
        """What Nova was saying around the user's onset (for the echo check)."""
        return " ".join(txt for txt, a, b in schedule if b >= t0 - 1.0 and a <= now() + 0.3)

    def _heard_until(t: float) -> str:
        parts = []
        for txt, a, b in schedule:
            if t >= b:
                parts.append(txt)
            elif t > a:
                w = txt.split()
                parts.append(" ".join(w[:max(1, int(len(w) * (t - a) / max(b - a, 1e-3)))]))
                break
            else:
                break
        return " ".join(parts)

    def _mic_and_ref(dur_s: float, max_lag_s: float = 0.5):
        """Last `dur_s` of mic audio and the matching window of Nova's playback
        (extended `max_lag_s` into the past for the echo-delay search)."""
        import numpy as np

        frames = [(t, f) for t, f in mic_ring if t > now() - dur_s - 0.05]
        if not frames or not ref_audio:
            return None, None
        mic = np.concatenate([np.frombuffer(f, dtype=np.int16) for _, f in frames]).astype(np.float32) / 32768.0
        mic = mic[-int(dur_s * SR):]
        r0 = frames[-1][0] - mic.size / SR - max_lag_s
        ref = np.zeros(mic.size + int(max_lag_s * SR), dtype=np.float32)
        for st, seg in ref_audio:
            if seg is None or not len(seg):
                continue
            off = int(round((st - r0) * SR))
            a, b = max(0, off), min(ref.size, off + len(seg))
            if b > a:
                ref[a:b] += seg[a - off:b - off]
        return mic, ref

    def _acoustic_echo(t_end: float, dur_s: float = 0.8) -> float:
        mic, ref = _mic_and_ref(dur_s)
        return 0.0 if mic is None else _echo_corr(mic, ref)

    def _pending_stt(buf_len_s: float) -> str:
        t0 = now()
        try:
            return _pending_stt_inner(buf_len_s)
        finally:
            if now() - t0 > 0.3:
                log.info("slow barge-in STT: %.0f ms for %.2f s", (now() - t0) * 1000, buf_len_s)

    def _pending_stt_inner(buf_len_s: float) -> str:
        """Transcribe the pending onset with Nova's own echo subtracted."""
        import numpy as np

        mic, ref = _mic_and_ref(buf_len_s)
        if mic is not None:
            res, corr = _subtract_echo(mic, ref)
            if corr >= 0.3:
                # Residual energy vs mic energy: echo-only onsets cancel to a
                # faint remainder that STT turns into invented words.
                if float(np.sqrt(np.mean(res ** 2))) < ECHO_RESIDUAL * float(np.sqrt(np.mean(mic ** 2)) + 1e-9):
                    return ""
                pcm = (np.clip(res, -1, 1) * 32767).astype(np.int16).tobytes()
                return offline_engine.stt_pcm(pcm, SR, cfg["language"])
        return offline_engine.stt_pcm(bytes(pending_buf_snapshot[0]), SR, cfg["language"])

    async def _echo_check(pd: dict) -> bool:
        """Acoustic echo test on the onset window; True = Nova's own voice."""
        win = min(0.8, max(0.2, len(pd["buf"]) / (2 * SR)))
        pd["ae_t"] = now()
        _ta = now()
        corr = await asyncio.to_thread(_acoustic_echo, now(), win)
        pd["ae_max"] = max(pd.get("ae_max", 0.0), corr)
        if now() - _ta > 0.1:
            log.info("echo check took %.0f ms", (now() - _ta) * 1000)
        pd["ae_corr"] = corr
        if corr >= ECHO_CORR:
            log.info("pending onset is Nova's own echo (corr %.2f)", corr)
            pd["text"] = f"<echo corr={corr:.2f}>"
            return True
        return False

    async def _check_pending() -> str:
        """Transcribe the pending speech (non-blocking) and classify it."""
        pd = pending
        el = now() - pd["t0"]
        if el >= BARGE_FIRST_EVAL_S and now() - pd.get("ae_t", 0.0) >= 0.15:
            if await _echo_check(pd):
                return "dismiss"
        task = pd["task"]
        if task is not None:
            if not task.done():
                return "wait"
            pd["task"] = None
            try:
                pd["text"] = (task.result() or "").strip()
            except Exception:  # noqa: BLE001
                pd["text"] = ""
            v = _barge_verdict(pd["text"], _inflight_text(pd["t0"]), pd["ended"], el)
            log.info("barge eval %.0f ms speech %.2f s: %r -> %s", el * 1000, pd.get("speech_s", 0), pd["text"][:40], v)
            if v == "confirm":
                own = [x for x in _words(pd["text"]) if x not in set(_words(_inflight_text(pd["t0"])))]
                urgent = any(x in _INTERRUPT_WORDS for x in own)
                content = {x for x in own if x not in _BACKCHANNEL}
                prev = pd.get("prev_content")
                pd["prev_content"] = content
                allw = _words(pd["text"])
                pw = pd.get("prev_words") or []
                pd["prev_words"] = allw
                # Same words twice (shared content word, or the new decode
                # starts like the previous one: "It's not if" -> "It's not even").
                stable = bool(prev and prev & content) or (
                    len(pw) >= 2 and allw[:2] == pw[:2])
                # Loud echo under the onset (double talk): STT on the residue is
                # unreliable, so ask for more before stopping Nova.
                loud_echo = pd.get("ae_max", 0.0) >= BARGE_LOUD_ECHO_CORR
                strong = (len(content) >= 2 and stable) or len(content) >= 3 \
                    or pd.get("speech_s", 0.0) >= 0.7
                if not urgent and pd.get("speech_s", 0.0) < BARGE_MIN_SPEECH_S:
                    # Too little voiced audio to trust the words yet.
                    v = "dismiss" if pd["ended"] else "wait"
                elif not urgent and _looks_degenerate(pd["text"], len(pd["buf"]) / (2 * SR)):
                    v = "dismiss" if pd["ended"] else "wait"  # "you, you, you" = STT looping on a hum
                elif not urgent and loud_echo and not strong:
                    v = "dismiss" if pd["ended"] else "wait"
                elif not urgent and not pd["ended"] and not stable:
                    # Words invented from a hum or echo residue change from one
                    # decode to the next; a real interruption keeps its words.
                    v = "wait"
                elif not urgent and pd["ended"] and not stable \
                        and pd.get("speech_s", 0.0) < 0.6:
                    v = "dismiss"
                elif await _echo_check(pd):
                    # Re-test on the window STT just heard before stopping Nova.
                    v = "dismiss"
            if v != "wait":
                log.info("barge-in %s after %.0f ms: %r", v, el * 1000, pd["text"][:60])
                return v
        if (el >= BARGE_FIRST_EVAL_S and now() - pd["last"] >= BARGE_EVAL_EVERY_S) or pd["ended"]:
            if pd["task"] is None:
                pd["last"] = now()
                pending_buf_snapshot[0] = bytes(pd["buf"])
                _tq = now()
                def _timed(n=len(pd["buf"]) / (2 * SR), _tq=_tq):
                    t1 = now()
                    r = _pending_stt(n)
                    log.info("barge STT queued %.0f ms, ran %.0f ms", (t1 - _tq) * 1000, (now() - t1) * 1000)
                    return r
                pd["task"] = asyncio.create_task(asyncio.to_thread(_timed))
        if el > 4.0:  # never hold an evaluation open forever
            return "confirm" if _words(pd["text"]) else "dismiss"
        return "wait"

    def _history_heard(heard: str) -> None:
        """History after an interruption keeps only what the user actually heard."""
        cut = (heard + " —") if heard else "—"
        rt = cur_user.get("rag_turn") if SESSION_HIST else None
        lead = (rt or {}).get("filler", "").strip()
        if cur_user.get("logged"):  # process() already wrote the full turn: trim it in place
            n = 4 if rt else 2
            del history[-n:]
        if rt is not None and cur_user.get("pre"):
            if rt.get("stage") != 2 or not heard or len(heard) <= len(lead):
                # cut during the lead-in: keep the plain question and what was heard, not a dangling
                # "checking your papers" (the cache rewinds the short lookup turn cheaply)
                history.append({"role": "user", "content": cur_user["text"]})
                history.append({"role": "assistant", "content": (heard + " —") if heard else "—"})
                return
            history.extend(cur_user["pre"])
            history.append({"role": "assistant", "content": lead})
            rest = heard[len(lead):].strip() if heard.startswith(lead) else heard
            history.append({"role": "user", "content": cur_user["prompt"]})
            history.append({"role": "assistant", "content": (rest + " —") if rest else "—"})
            return
        history.append({"role": "user", "content": cur_user["prompt"] if SESSION_HIST else cur_user["text"]})
        history.append({"role": "assistant", "content": cut})

    async def _interrupt(buf: bytes, heard: str, ended: bool = False, continuing: bool = False) -> None:
        """Confirmed interruption: stop Nova, keep only what was heard, listen."""
        nonlocal collecting, utterance, trace, turn_seq, audible_until
        cancel.set()
        sess_llm.cancel()
        if continuing:
            pass  # the clip is re-transcribed together with what follows: logging it now would duplicate it
        elif cur_user["text"] and "prompt" in cur_user:
            _history_heard(heard)
            if not SESSION_HIST:
                del history[:-2 * offline_engine.LLM_HISTORY_TURNS]
        cur_user["text"] = ""  # this turn is settled; a second interrupt must not add it again
        audible_until = now()
        schedule.clear()
        await send({"type": "barge_in"})
        _drop_eager("barge-in")
        collecting, utterance = True, bytearray(buf)
        utter_origin["fresh"] = False  # this recording already holds the merged turn
        preroll.clear()
        turn_seq += 1
        trace = TurnTrace(turn_seq)
        trace.mark_speech()
        if ended:
            # The user already stopped talking while we were deciding: run the
            # end-of-turn check on the next frame instead of waiting for the
            # next VAD mark (which may have already fired during the decision).
            forced_events[:] = ["endpoint_soft", "endpoint"]
        await set_state("listening")

    async def _timer_watch() -> None:
        """Speak finished timers (tools.py) when Nova is idle on this connection."""
        nonlocal audible_until
        while not closed["v"]:
            await asyncio.sleep(1.0)
            p = cfg.get("profile")
            if not (TOOLS_ON and p) or state != "listening" or collecting or pending is not None or td.in_speech:
                continue
            try:
                due = await asyncio.to_thread(tools.due_timers, p)
            except Exception as e:  # noqa: BLE001
                log.warning("timer check failed: %s", e)
                continue
            for t in due:
                who = personas.PERSONAS.get(p, {}).get("name", "")
                what = f" for the {t['label']}" if t.get("label") else ""
                msg_t = (f"{who}, your" if who else "Your") + f" {t['length']} timer{what} is done."
                log.info("timer done: %s", msg_t)
                await send({"type": "timer_done", **t})
                await set_state("speaking")
                await send({"type": "sentence", "text": msg_t})
                r = await asyncio.to_thread(_synth, msg_t, cfg["voice"], cfg.get("speed", 1.0))
                dur = _wav_seconds(r["wav"])
                audible_until = now() + dur
                last_reply.update(text=msg_t, end=now() + dur)
                await send_bin(r["wav"])
                await asyncio.sleep(dur + 0.3)
                if state == "speaking":
                    await set_state("listening")

    timer_task = asyncio.create_task(_timer_watch())
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                closed["v"] = True
                break
            data = msg.get("bytes")
            if data is not None:
                mic_ring.append((now(), data))
                events, any_speech = td.push(data)
                if forced_events:
                    events = forced_events + [e for e in events if e not in forced_events]
                    forced_events = []
                if not collecting:
                    preroll += data
                    if len(preroll) > _PREROLL_BYTES:
                        del preroll[: len(preroll) - _PREROLL_BYTES]
                if state in ("processing", "speaking"):
                    if pending is None and collecting and utter_origin["fresh"]:
                        # Speech that resumed in the same frames as the commit (a pause mid-question:
                        # "...how transformers work, | step by step"). The VAD is already in speech, so
                        # no new speech_start will come: hand it to the same paths as a fresh onset,
                        # or it would sit unread until the reply ends and swallow the next barge-in.
                        buf = bytes(utterance)
                        collecting, utterance = False, bytearray()
                        if not schedule:
                            log.info("speech resumed at the commit — continuing the user's turn")
                            await _interrupt(bytes(last_clip["pcm"]) + buf, heard="", continuing=True)
                            continue
                        log.info("pending onset (resumed at commit) while %s", state)
                        pending = {"t0": now(), "buf": bytearray(buf), "last": 0.0,
                                   "task": None, "ended": False, "text": "", "speech_s": len(buf) / (2 * SR)}
                        continue
                    if pending is None and "speech_start" in events:
                        if not schedule:
                            # Nothing played yet ("thinking"): the user is
                            # continuing their turn — drop the pending answer
                            # and keep listening to BOTH parts.
                            log.info("speech during processing — continuing the user's turn")
                            await _interrupt(bytes(last_clip["pcm"]) + bytes(preroll), heard="", continuing=True)
                            continue
                        log.info("pending onset while %s", state)
                        pending = {"t0": now(), "buf": bytearray(preroll), "last": 0.0,
                                   "task": None, "ended": False, "text": "",
                                   "speech_s": 0.2}  # speech_start needs ~192 ms of speech
                        preroll.clear()
                        continue
                    if pending is not None:
                        pending["buf"] += data
                        if any_speech:
                            pending["ended"] = False
                            pending["speech_s"] += len(data) / (2 * SR)
                        if "endpoint_soft" in events or "endpoint" in events or "endpoint_hard" in events:
                            pending["ended"] = True
                        verdict = await _check_pending()
                        if verdict == "confirm":
                            buf, ended = bytes(pending["buf"]), pending["ended"]
                            pending = None
                            await _interrupt(buf, heard=_heard_until(now()), ended=ended)
                        elif verdict == "dismiss":
                            log.info("not an interruption (%r) — Nova keeps talking",
                                     pending["text"][:40])
                            pending = None
                            td.turn_committed()  # re-arm the VAD for the next onset
                    continue
                if state == "listening" and pending is not None:
                    # Nova finished playing while an onset was still being
                    # judged: those words are the user's next turn, not lost.
                    pd, pending = pending, None
                    if pd["task"] is not None and pd["task"].done() and not pd["text"]:
                        try:
                            pd["text"] = (pd["task"].result() or "").strip()
                        except Exception:  # noqa: BLE001
                            pd["text"] = ""
                    verdict = _barge_verdict(pd["text"], last_reply["text"], True, now() - pd["t0"]) \
                        if pd["text"] else "keep"
                    if verdict != "dismiss" and td.in_speech:
                        collecting, utterance = True, bytearray(pd["buf"])
                        turn_seq += 1
                        trace = TurnTrace(turn_seq)
                        trace.mark_speech()
                    elif verdict == "dismiss":
                        td.turn_committed()
                if state == "listening":
                    # speech_start is handled AFTER the commit block: a single
                    # batched push can carry endpoint_hard AND speech_start,
                    # and wiping the utterance first committed a fragment.
                    fresh_start = "speech_start" in events
                    if "noise_reset" in events and collecting:
                        # The "speech" that opened this turn was background
                        # talk / TV heard before the room level was learned.
                        log.info("dropping utterance: background speech, not the user")
                        collecting, utterance = False, bytearray()
                        _drop_eager("noise reset")
                        trace = None
                    if collecting:
                        utterance += data
                        if any_speech and trace is not None:
                            trace.mark_speech()  # speech_end_t = last speechy push
                        if eager_stt is not None and any_speech:
                            _drop_eager("speech resumed")
                    if collecting and trace is not None:
                        if "endpoint_soft" in events:
                            trace.soft_t = now()
                        if "endpoint" in events:
                            trace.check_t = now()
                    # Force a turn if the speaker never pauses, so memory stays bounded.
                    over_cap = len(utterance) >= _MAX_UTTERANCE_BYTES
                    commit = False
                    semantic = _get_semantic()
                    if collecting and "endpoint_soft" in events and eager_stt is None:
                        if trace is not None:
                            trace.eager_start_t = now()
                        eager_stt = asyncio.create_task(asyncio.to_thread(
                            offline_engine.stt_pcm, bytes(utterance), SR,
                            cfg["language"]))
                        await stage("turn", "start", detail="pause heard")
                        await stage("stt", "start")
                        _t_stt = now()

                        def _stt_done(t: asyncio.Task, _t0=_t_stt) -> None:
                            if t.cancelled() or t.exception() is not None:
                                return
                            asyncio.ensure_future(stage("stt", "done", ms=round((now() - _t0) * 1000), text=(t.result() or "").strip()))
                        eager_stt.add_done_callback(_stt_done)
                        # Speculative prefill: as soon as the eager transcript is in, fetch its
                        # memory facts and prefill the exact user chunk while Smart Turn decides.
                        # A commit with the same transcript decodes at once; otherwise the
                        # session rewinds (free on v0.17).
                        _epoch = sess_llm.epoch

                        def _spec_local(t: asyncio.Task, _h=list(history), _epoch=_epoch) -> None:
                            if t.cancelled() or t.exception() is not None or t in dropped_eager:
                                return
                            txt = (t.result() or "").strip()
                            if not txt or _is_filler_only(txt):
                                return
                            prof, echo_on = cfg.get("profile"), cfg["echo"]
                            txt = personas.apply_aliases(prof, txt)
                            if personas.is_child(prof) and safety.check_input(txt):
                                return  # never prefill a flagged child utterance into the KV

                            def job() -> None:
                                facts = []
                                if MEMORY_ON and not echo_on:
                                    try:
                                        facts, _ = memory_store.get(prof).retrieve(txt, count=False)
                                    except Exception:  # noqa: BLE001
                                        facts = []
                                rag = _rag_lookup(prof, txt) if not echo_on else None
                                spec_cache.clear()
                                spec_cache.update(text=txt, rag=rag, profile=prof)
                                user = (_rag_user_chunks(txt, rag, facts, prof)[0] if _use_library(rag, facts)
                                        else compose_prompt(txt, facts) + _not_found_note(prof, txt, rag, facts))
                                sess_llm.speculate(_h, user, system=personas.runtime_system(prof), epoch=_epoch)
                            threading.Thread(target=job, name="spec-prefill", daemon=True).start()
                        eager_stt.add_done_callback(_spec_local)
                    trigger = None
                    if collecting and semantic is not None:
                        # Two-stage: at the soft mark (TURN_SOFT_MS) commit only on a
                        # very confident "complete" (>=THRESHOLD_SOFT); at the
                        # check mark (TURN_CHECK_MS) commit unless clearly
                        # incomplete (<THRESHOLD_CHECK). Inference (~60-90 ms on a laptop CPU)
                        # runs in a thread; frames just queue briefly.
                        if "endpoint_soft" in events:
                            prob = await _predict(semantic)
                            if prob >= semantic.THRESHOLD_SOFT:
                                commit, trigger = True, f"soft p={prob:.2f}"
                            else:
                                log.info("turn wait at soft: p=%.2f", prob)
                                await stage("turn", "wait", p=round(prob, 2))
                        if not commit and "endpoint" in events:
                            prob = await _predict(semantic)
                            if prob >= semantic.THRESHOLD_CHECK:
                                commit, trigger = True, f"check p={prob:.2f}"
                            else:
                                # Smart Turn is unsure on one-word replies ("Yes.");
                                # if the eager transcript is a complete short answer,
                                # don't make the user wait for the hard cap.
                                et = eager_stt
                                short = ""
                                if et is not None and et.done() and not et.cancelled():
                                    try:
                                        short = (et.result() or "").strip()
                                    except Exception:  # noqa: BLE001
                                        short = ""
                                sw = _words(short)
                                if sw and len(sw) <= 3 and all(x in _SHORT_COMPLETE for x in sw):
                                    commit, trigger = True, f"short answer p={prob:.2f}"
                                elif _complete_question(short):
                                    # Smart Turn is often unsure on flat, read-out questions ("What have I got on
                                    # this week?"); a full, punctuated question after 672 ms of silence is done.
                                    commit, trigger = True, f"complete question p={prob:.2f}"
                                else:
                                    log.info("turn wait at check: p=%.2f", prob)
                        if not commit and "endpoint_hard" in events:
                            commit, trigger = True, "hard cap"
                    elif collecting:
                        if "endpoint" in events or "endpoint_hard" in events:
                            commit, trigger = True, "fallback silence"
                    if collecting and (commit or over_cap):
                        log.info("turn commit: %s (%d bytes)",
                                 trigger or "over cap", len(utterance))
                        td.turn_committed()
                        collecting = False
                        schedule.clear()  # the previous reply finished playing (we were listening): nothing of it is "playing" now
                        clip = bytes(utterance)
                        last_clip["pcm"] = clip
                        utterance = bytearray()
                        pre = None
                        if eager_stt is not None:
                            pre, eager_stt = eager_stt, None
                            log.info("eager STT used")
                        tr_commit, trace = trace, None
                        if tr_commit is None:
                            turn_seq += 1
                            tr_commit = TurnTrace(turn_seq)
                        tr_commit.commit_t = now()
                        tr_commit.commit_trigger = trigger or "over cap"
                        tr_commit.utter_ms = round(len(clip) / 32.0, 1)
                        tr_commit.stt_eager_used = pre is not None
                        await stage("turn", "done", detail=trigger or "over cap",
                                    ms=round(((tr_commit.commit_t - tr_commit.speech_end_t) * 1000) if tr_commit.speech_end_t else 0))
                        if proc is not None and not proc.done():
                            # Previous turn still winding down (barge-in set
                            # cancel; it drains in well under a second). The
                            # NEW turn wins: wait briefly instead of dropping
                            # the user's clip.
                            cancel.set()
                            try:
                                await asyncio.wait_for(asyncio.shield(proc), 1.5)
                            except (asyncio.TimeoutError, Exception):  # noqa: BLE001
                                log.warning("previous turn did not wind down in 1.5 s; cancelling it")
                                proc.cancel()  # it must never resume and write into the new turn's state
                                await asyncio.wait([proc], timeout=2.0)
                            cancel.clear()
                        proc = asyncio.create_task(process(clip, pre, tr_commit))
                    if fresh_start and not collecting:
                        utter_origin["fresh"] = True
                        collecting, utterance = True, bytearray(preroll)
                        preroll.clear()
                        turn_seq += 1
                        trace = TurnTrace(turn_seq)
                        trace.mark_speech()
                        _drop_eager("new utterance")
                        partial_state["seq"] += 1
                        partial_state["last"] = now()
                        await stage("hear", "start")
                    # Live partial transcript every ~0.6 s while the person is talking.
                    if (collecting and any_speech and eager_stt is None
                            and now() - partial_state["last"] >= 0.6
                            and (partial_state["task"] is None or partial_state["task"].done())):
                        partial_state["last"] = now()
                        partial_state["task"] = asyncio.create_task(_partial_tick(partial_state["seq"]))
                continue
            text = msg.get("text")
            if text:
                try:
                    ctrl = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if ctrl.get("type") == "config":
                    cfg["echo"] = bool(ctrl.get("echo", cfg["echo"]))
                    v = str(ctrl.get("voice") or cfg["voice"])
                    if v not in KNOWN_VOICES or not open_tts.available(v):
                        log.warning("unknown/unavailable voice %r; keeping %s", v, cfg["voice"])
                        v = cfg["voice"]
                    cfg["voice"] = v
                    log.info("TTS voice: %s", v)
                    # Build/warm the engine now (in a thread) so the first turn
                    # never pays the load; _tts_for is lock-guarded, so a turn
                    # racing this just waits on the same build.
                    t = asyncio.create_task(asyncio.to_thread(_tts_for, v))
                    _warm_tasks.add(t)
                    t.add_done_callback(_warm_done)
                    # Benchmark labels (bench/e2e_bench.py), copied into the per-turn trace.
                    for k in ("u_bucket", "clip_id"):
                        if k in ctrl:
                            cfg[k] = ctrl[k] or None
                    if "language" in ctrl:
                        cfg["language"] = ctrl["language"] or None
                    if "profile" in ctrl:
                        p = str(ctrl["profile"] or memory_store.DEFAULT_PROFILE)
                        cfg["profile"] = p if p in memory_store.PROFILES else memory_store.DEFAULT_PROFILE
                        import templates as _tpl
                        cfg["speed"] = _tpl.speed(_tpl.current(cfg["profile"]))  # Care template: a slower voice
                        if proc is None or proc.done():
                            # One-time prefill of this profile's system prompt while the greeting plays.
                            asyncio.ensure_future(asyncio.to_thread(sess_llm.warm_system, sysp()))
                elif ctrl.get("type") == "greet":
                    if (proc is None or proc.done()) and not collecting:
                        cancel.clear()
                        proc = asyncio.create_task(greet())
                elif ctrl.get("type") == "stop":
                    # Only meaningful with a live turn — a stray stop used to
                    # leave `cancel` set with no process() to clear it, muting
                    # the NEXT turn's entire reply.
                    if proc is not None and not proc.done():
                        cancel.set()
                        sess_llm.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        cancel.set()
        sess_llm.cancel()
        if conv["sid"] is not None:
            with contextlib.suppress(Exception):
                conv_store.end_session(conv["profile"], conv["sid"])
        timer_task.cancel()
        if proc and not proc.done():
            proc.cancel()
