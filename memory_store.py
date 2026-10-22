"""On-device memory for Nova: one SQLite file per profile, exact in-memory vector scan.

- Store of record: SQLite (facts table: text, embedding BLOB, embed_model, dim, timestamps, source).
- Search: the profile's fact vectors are held as one float32 matrix and scanned exactly
  (dot product; vectors are unit-norm); 10k facts scan in well under a millisecond on a laptop CPU.
- Score = 0.8·cosine + 0.2·recency (30-day half-life); facts below MIN_COS are never returned
  so an unrelated question gets no memory block at all.
- Writes: rule-based extraction from the user's sentence (immediate, < 1 ms); near-duplicates
  (cosine ≥ DEDUP_COS) update the existing row instead of inserting.
- Profiles: separate files under MEMORY_DIR (default data/); a profile's optional seed facts come from personas.py.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from pathlib import Path

import logging

import numpy as np

import embedder

log = logging.getLogger("nova.memory")

HERE = Path(__file__).resolve().parent
MEMORY_DIR = Path(os.environ.get("MEMORY_DIR", HERE / "data"))
TOP_K = int(os.environ.get("MEMORY_TOP_K", "5"))
MIN_COS = float(os.environ.get("MEMORY_MIN_COS", "0.25"))
# Schedule complement: generic "what's on this week?" questions score only ~0.2 cosine
# against specific dated facts, so facts carrying a time expression are tagged at insert
# time and a schedule-type question pulls them explicitly (still ranked, still budgeted).
_TIME_RX = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow|tonight|today|next (?:week|month|year)|this (?:week|month|weekend)|every (?:morning|evening|night|day|sunday|monday|tuesday|wednesday|thursday|friday|saturday)|\d{1,2}(?::\d{2})? ?(?:am|pm)|at \d{1,2}\b|on the \d{1,2}(?:st|nd|rd|th)|\d{1,2} (?:january|february|march|april|may|june|july|august|september|october|november|december)|(?:january|february|march|april|may|june|july|august|september|october|november|december)\b|appointment|exam|curfew|anniversary|birthday|play on|deadline|meeting|mock test|test on|reading group)", re.I)
_SCHEDULE_Q_RX = re.compile(r"\b(this week|this weekend|today|tomorrow|tonight|coming up|upcoming|schedule|calendar|agenda|plans?|got on|on this week|what'?s on|anything on|appointments?|reminders?)\b", re.I)
DEDUP_COS = 0.90
RECENCY_HALF_LIFE_DAYS = 30.0
BUDGET_WORDS = int(os.environ.get("MEMORY_BUDGET_WORDS", "45"))  # ≈ 60 tokens; keeps the user chunk ≤ 128 tokens

# Profiles (personas.py), each with its own SQLite memory file.
from personas import ORDER as _ORDER, PERSONAS as _PERSONAS  # noqa: E402

PROFILES: dict[str, dict] = {k: {"title": _PERSONAS[k]["title"],
                                 "facts": _PERSONAS[k]["facts"]} for k in _ORDER}
DEFAULT_PROFILE = _ORDER[0]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts(
  id INTEGER PRIMARY KEY,
  text TEXT NOT NULL,
  embedding BLOB NOT NULL,
  embed_model TEXT NOT NULL,
  dim INTEGER NOT NULL,
  created REAL NOT NULL,
  updated REAL NOT NULL,
  source TEXT NOT NULL,
  hits INTEGER NOT NULL DEFAULT 0,
  deleted INTEGER NOT NULL DEFAULT 0,
  tags TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


class MemoryStore:
    def __init__(self, profile: str, path: Path | None = None) -> None:
        if profile not in PROFILES and not re.fullmatch(r"[a-z0-9_-]{1,32}", profile):
            raise ValueError(f"bad profile name {profile!r}")
        self.profile = profile
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self.path = path or (MEMORY_DIR / f"{profile}.sqlite")
        self._lock = threading.RLock()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.executescript(_SCHEMA)
        self._ids: list[int] = []
        self._X = np.zeros((0, embedder.DIM), np.float32)
        self._updated: list[float] = []
        self._texts: list[str] = []
        self._timed: list[bool] = []
        self._load()
        if profile in PROFILES and self.db.execute("SELECT value FROM meta WHERE key='seeded'").fetchone() is None:
            self.seed()

    # ---------- persistence ----------
    def _load(self) -> None:
        rows = self.db.execute("SELECT id, embedding, updated, embed_model, dim, text, tags FROM facts WHERE deleted=0 ORDER BY id").fetchall()
        stale = [r[0] for r in rows if r[3] != embedder.MODEL_NAME or r[4] != embedder.DIM]
        if stale:  # embedding model changed → re-embed those rows
            texts = {r[0]: r[1] for r in self.db.execute("SELECT id, text FROM facts WHERE id IN (%s)" % ",".join("?" * len(stale)), stale)}
            V = embedder.get().encode([texts[i] for i in stale])
            with self.db:
                for i, v in zip(stale, V):
                    self.db.execute("UPDATE facts SET embedding=?, embed_model=?, dim=? WHERE id=?", (v.tobytes(), embedder.MODEL_NAME, embedder.DIM, i))
            rows = self.db.execute("SELECT id, embedding, updated, embed_model, dim, text, tags FROM facts WHERE deleted=0 ORDER BY id").fetchall()
        self._ids = [r[0] for r in rows]
        self._updated = [r[2] for r in rows]
        self._texts = [r[5] for r in rows]
        self._timed = ["time" in r[6] for r in rows]
        self._X = (np.frombuffer(b"".join(r[1] for r in rows), np.float32).reshape(len(rows), embedder.DIM).copy()
                   if rows else np.zeros((0, embedder.DIM), np.float32))

    def seed(self) -> int:
        n = 0
        for f in PROFILES.get(self.profile, {}).get("facts", []):
            if self.add(f, source="seed") is not None:
                n += 1
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('seeded', ?)", (str(time.time()),))
        return n

    def reset(self) -> int:
        with self._lock, self.db:
            self.db.execute("DELETE FROM facts")
            self.db.execute("DELETE FROM meta")
        self._load()
        return self.seed()

    # ---------- write ----------
    def add(self, text: str, source: str = "user") -> int | None:
        """Insert a fact (or refresh a near-duplicate). Returns the row id, None if empty."""
        text = " ".join(text.split()).strip()
        if not text:
            return None
        v = embedder.get().encode([text])[0]
        with self._lock:
            if len(self._ids):
                sims = self._X @ v
                j = int(np.argmax(sims))
                if sims[j] >= DEDUP_COS:
                    fid = self._ids[j]
                    tags = "time" if _TIME_RX.search(text) else ""
                    with self.db:
                        self.db.execute("UPDATE facts SET text=?, embedding=?, updated=?, tags=? WHERE id=?", (text, v.tobytes(), time.time(), tags, fid))
                    self._X[j] = v
                    self._updated[j] = time.time()
                    self._texts[j] = text
                    self._timed[j] = bool(tags)
                    return fid
            t = time.time()
            tags = "time" if _TIME_RX.search(text) else ""
            with self.db:
                cur = self.db.execute(
                    "INSERT INTO facts(text, embedding, embed_model, dim, created, updated, source, tags) VALUES (?,?,?,?,?,?,?,?)",
                    (text, v.tobytes(), embedder.MODEL_NAME, embedder.DIM, t, t, source, tags))
            self._ids.append(int(cur.lastrowid))
            self._updated.append(t)
            self._texts.append(text)
            self._timed.append(bool(tags))
            self._X = np.vstack([self._X, v[None, :]]) if len(self._X) else v[None, :].copy()
            return int(cur.lastrowid)

    def delete(self, fid: int) -> bool:
        with self._lock:
            with self.db:
                n = self.db.execute("UPDATE facts SET deleted=1 WHERE id=? AND deleted=0", (fid,)).rowcount
            if n:
                self._load()
            return bool(n)

    # ---------- read ----------
    def retrieve(self, query: str, k: int = TOP_K, count: bool = True) -> tuple[list[dict], float]:
        """Top-k facts for a query → ([{id, text, cos, score}], ms). Empty if nothing passes MIN_COS."""
        t0 = time.perf_counter()
        if not query.strip() or not len(self._ids):
            return [], (time.perf_counter() - t0) * 1000.0
        q = embedder.get().encode([query])[0]
        t_embed = (time.perf_counter() - t0) * 1000.0
        with self._lock:
            cos = self._X @ q
            age_days = (time.time() - np.asarray(self._updated)) / 86400.0
            score = 0.8 * cos + 0.2 * np.exp(-age_days / RECENCY_HALF_LIFE_DAYS)
            schedule_q = bool(_SCHEDULE_Q_RX.search(query))
            timed = np.asarray(self._timed, bool) if self._timed else np.zeros(len(cos), bool)
            keep = cos >= MIN_COS
            if schedule_q:  # schedule complement: dated facts qualify even at low cosine
                keep |= timed & (cos >= 0.05)  # dated facts are few; the budget still caps them
                score = score + 0.15 * timed
            order = np.argsort(-score)[: max(k * 3, k)]
            out, words = [], 0
            for j in order:
                if not keep[j] or len(out) >= k:
                    continue
                w = len(self._texts[j].split())
                if words + w > BUDGET_WORDS:
                    continue
                words += w
                out.append({"id": self._ids[j], "text": self._texts[j], "cos": round(float(cos[j]), 3),
                            "score": round(float(score[j]), 3), "why": "schedule" if (schedule_q and timed[j] and cos[j] < MIN_COS) else "similar"})
            if out and count:  # speculative lookups (count=False) do not bump recall counters
                with self.db:
                    self.db.executemany("UPDATE facts SET hits=hits+1 WHERE id=?", [(f["id"],) for f in out])
        total = (time.perf_counter() - t0) * 1000.0
        log.info("memory[%s]: %d/%d facts, embed %.1f ms, scan+db %.1f ms, schedule_q=%s",
                 self.profile, len(out), len(self._ids), t_embed, total - t_embed, schedule_q)
        return out, total

    def list(self) -> list[dict]:
        rows = self.db.execute("SELECT id, text, source, created, updated, hits FROM facts WHERE deleted=0 ORDER BY updated DESC").fetchall()
        return [dict(id=r[0], text=r[1], source=r[2], created=r[3], updated=r[4], hits=r[5]) for r in rows]

    def count(self) -> int:
        return len(self._ids)

    def close(self) -> None:
        self.db.close()


# ---------- rule-based fact extraction (immediate path; LLM extraction is a later flag) ----------
_RULES = [
    (re.compile(r"\bmy ([a-z][a-z ]{1,30}?)'s name is ([A-Za-z][\w -]{0,40})", re.I), "User's {0} is called {1}."),
    (re.compile(r"\bmy ([a-z][a-z ]{1,30}?)(?:'s| is) (?:called|named) ([A-Za-z][\w -]{0,40})", re.I), "User's {0} is called {1}."),
    (re.compile(r"\bmy ([a-z][a-z' ]{1,30}?) (?:is|are) ([^.,;!?]{2,60})", re.I), "User's {0} is {1}."),
    (re.compile(r"\bremind me (?:to |about |that )([^.,;!?]{3,80})", re.I), "User asked to be reminded to {0}."),
    (re.compile(r"\bI (?:really )?(love|like|hate|prefer|enjoy|dislike) ([^.,;!?]{2,60})", re.I), "User {0}s {1}."),
    (re.compile(r"\bI(?:'m| am) allergic to ([^.,;!?]{2,60})", re.I), "User is allergic to {0}."),
    (re.compile(r"\bI(?:'m| am) (?:afraid|scared) of ([^.,;!?]{2,60})", re.I), "User is afraid of {0}."),
    (re.compile(r"\bI (?:have|got|own) (?:a |an |two |three )?([^.,;!?]{2,60})", re.I), "User has {0}."),
    (re.compile(r"\bI(?:'m| am) (?:going|travelling|traveling|flying) to ([^.,;!?]{2,60})", re.I), "User is going to {0}."),
    (re.compile(r"\bI(?:'m| am) (?:learning|studying) ([^.,;!?]{2,60})", re.I), "User is learning {0}."),
    (re.compile(r"\bI (?:work|study) (?:at|as|in) ([^.,;!?]{2,60})", re.I), "User works at {0}."),
    (re.compile(r"\bmy (?:name is|name's) ([A-Za-z][a-z]+)", re.I), "User's name is {0}."),
    # reversed and looser phrasings ("Leela is my sister", "his name is Bruno", "I moved to Denver")
    (re.compile(r"\b(?!(?:This|That|She|He|It|Here|There|Who|What|Which|Mine)\b)([A-Z][a-z]+)(?:'s| is) my ((?:best |little |big |elder |younger )?(?:sister|brother|mom|mother|dad|father|wife|husband|"
                r"partner|son|daughter|friend|boss|manager|cousin|aunt|uncle|grandma|grandmother|grandpa|grandfather|teacher|cat|dog|"
                r"co-?founder|business partner|girlfriend|boyfriend|roommate|colleague))\b"), "User's {1} is called {0}."),
    (re.compile(r"\b(?:new |our |a )?(puppy|kitten|dog|cat|fish|parrot|rabbit|baby)\b[^.!?]{0,30}?\b(?:his|her|its|their) name is ([A-Z][a-z]+)", re.I),
     "User has a {0} called {1}."),
    (re.compile(r"\bI(?: just| recently)? (?:moved|shifted|relocated) to ([A-Z][^.,;!?]{1,40})"), "User moved to {0}."),
    (re.compile(r"\bI(?: just| recently)? (?:started|joined|began) (?:working |a job )?(?:at|with) ([A-Z][^.,;!?]{1,40})"), "User works at {0}."),
    (re.compile(r"\bI can'?t (?:eat|have|drink) ([^.,;!?]{2,40})[,;]? (?:I'?m|as I'?m|because I'?m) allergic", re.I), "User is allergic to {0}."),
    (re.compile(r"\bI usually ([^.,;!?]{4,60})", re.I), "User said they usually {0}."),
    (re.compile(r"\bour ((?:new |next )?(?:store|shop|office|launch|collection|warehouse|show|event) (?:opens|launches|is|starts) [^.,;!?]{2,60})", re.I),
     "User's {0}."),
]
_PRON = [(re.compile(r"\bmy\b", re.I), "their"), (re.compile(r"\bme\b", re.I), "them"), (re.compile(r"\bI'm\b", re.I), "they are"), (re.compile(r"\bI\b"), "they")]


_QUESTION_RX = re.compile(r"^\s*(what|when|where|who|why|how|which|do|does|did|is|are|am|can|could|will|would|should|have|has|tell me|remind me what)\b", re.I)
# Objects that are pronouns or fillers are fragments ("I have it", "my thing is that"); questions
# are already excluded by _QUESTION_RX, so prepositional objects ("on 3 May") are kept.
_BAD_OBJ_RX = re.compile(r"^(any|anything|something|nothing|no|not|a lot|got|this|that|it|them|him|her|me|you)\b", re.I)


def extract_facts(user_text: str) -> list[str]:
    """Candidate memory facts from one user sentence (third person, short). Deterministic, < 1 ms.
    Questions ("what have I got on this week?") are never facts; objects that start with a
    preposition or a pronoun are fragments, not facts."""
    out: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", user_text.strip()):
        if not sent or _QUESTION_RX.match(sent):
            continue
        taken: list[tuple[int, int]] = []  # earlier, more specific rules win over overlapping generic ones
        for rx, tmpl in _RULES:
            for m in rx.finditer(sent):
                if any(m.start() < b and a < m.end() for a, b in taken):
                    continue
                taken.append((m.start(), m.end()))
                groups = [g.strip() for g in m.groups()]
                if any(_BAD_OBJ_RX.match(g) for g in groups[-1:]):
                    continue
                fact = tmpl.format(*groups)
                for rx2, rep in _PRON:
                    fact = rx2.sub(rep, fact)
                fact = fact[0].upper() + fact[1:]
                if len(fact.split()) >= 3 and fact not in out:
                    out.append(fact)
    return out[:3]


def format_block(facts: list[dict]) -> str:
    if not facts:
        return ""
    return "Things you remember about this person:\n" + "\n".join(f"- {f['text']}" for f in facts)


def compose_prompt(user_text: str, facts: list[dict]) -> str:
    """The user chunk sent to the LLM: memory block (if any) + the user's words. Stays ≤ ~128 tokens."""
    block = format_block(facts)
    return f"{block}\n\nThe person says: {user_text}" if block else user_text


_stores: dict[str, MemoryStore] = {}
_stores_lock = threading.Lock()


def get(profile: str | None) -> MemoryStore:
    p = profile or DEFAULT_PROFILE
    with _stores_lock:
        s = _stores.get(p)
        if s is None:
            s = _stores[p] = MemoryStore(p)
        return s


def profiles() -> list[dict]:
    return [{"id": k, "title": v["title"], "facts": get(k).count(), "voice": _PERSONAS[k].get("voice"),
             "child": _is_child(k), "template": _tmpl(k)} for k, v in PROFILES.items()]


def _is_child(k: str) -> bool:
    import personas
    return personas.is_child(k)


def _tmpl(k: str) -> str:
    import templates
    return templates.current(k)
