"""Per-profile conversation history: sessions and turns in data/<profile>_conversations.sqlite.

One websocket connection = one session. Each answered turn stores what the person said, Nova's reply
(and her lead-in when she looked something up), the memories and documents used, and the measured
time to first audio. Everything stays on this computer.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIR = HERE / "data"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY, started REAL NOT NULL, ended REAL, voice TEXT, device TEXT);
CREATE TABLE IF NOT EXISTS turns(id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, ts REAL NOT NULL,
  user TEXT, reply TEXT, filler TEXT, sources TEXT, memories TEXT, ttfa_ms REAL, interrupted INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id);
"""
_lock = threading.Lock()
_dbs: dict[str, sqlite3.Connection] = {}


def _db(profile: str) -> sqlite3.Connection:
    with _lock:
        if profile not in _dbs:
            DIR.mkdir(parents=True, exist_ok=True)
            c = sqlite3.connect(str(DIR / f"{profile}_conversations.sqlite"), check_same_thread=False)
            c.executescript(_SCHEMA)
            _dbs[profile] = c
        return _dbs[profile]


def start_session(profile: str, voice: str = "", device: str = "") -> int:
    c = _db(profile)
    with _lock, c:
        return int(c.execute("INSERT INTO sessions(started, voice, device) VALUES (?,?,?)", (time.time(), voice, device)).lastrowid)


def end_session(profile: str, sid: int) -> None:
    c = _db(profile)
    with _lock, c:
        c.execute("UPDATE sessions SET ended=? WHERE id=?", (time.time(), sid))
        # sessions where nothing was said are noise in the history
        if not c.execute("SELECT 1 FROM turns WHERE session_id=?", (sid,)).fetchone():
            c.execute("DELETE FROM sessions WHERE id=?", (sid,))


def add_turn(profile: str, sid: int, d: dict) -> None:
    if not (d.get("transcript") or "").strip():
        return
    c = _db(profile)
    with _lock, c:
        # the session row may have been deleted from the History drawer during this call: recreate it
        c.execute("INSERT OR IGNORE INTO sessions(id, started) VALUES (?, ?)", (sid, time.time()))
        c.execute("INSERT INTO turns(session_id, ts, user, reply, filler, sources, memories, ttfa_ms, interrupted) VALUES (?,?,?,?,?,?,?,?,?)",
                  (sid, d.get("wall") or time.time(), d.get("transcript"), d.get("reply") or d.get("reply_first") or "",
                   d.get("filler") or "", json.dumps((d.get("rag_sources") or []) + ([f"{d['tool'].replace('_', ' ')} tool" + (" (internet)" if d.get("tool_online") else "")] if d.get("tool") else [])), json.dumps(d.get("memory_facts") or []),
                   d.get("ttfa_ms"), int(bool(d.get("cancelled")))))


def sessions(profile: str, limit: int = 30) -> list[dict]:
    c = _db(profile)
    with _lock:
        rows = c.execute("""SELECT s.id, s.started, s.ended, s.voice, s.device, COUNT(t.id),
                        (SELECT user FROM turns WHERE session_id=s.id ORDER BY ts LIMIT 1), AVG(t.ttfa_ms)
                        FROM sessions s LEFT JOIN turns t ON t.session_id=s.id
                            GROUP BY s.id HAVING COUNT(t.id) > 0 ORDER BY s.started DESC LIMIT ?""", (limit,)).fetchall()
    return [dict(id=r[0], started=r[1], ended=r[2], voice=r[3], device=r[4], turns=r[5], first=r[6],
                 avg_ttfa_ms=round(r[7]) if r[7] else None) for r in rows]


def turns(profile: str, sid: int) -> list[dict]:
    c = _db(profile)
    with _lock:
        rows = c.execute("SELECT ts, user, reply, filler, sources, memories, ttfa_ms, interrupted FROM turns WHERE session_id=? ORDER BY ts", (sid,)).fetchall()
    return [dict(ts=r[0], user=r[1], reply=r[2], filler=r[3], sources=json.loads(r[4] or "[]"), memories=json.loads(r[5] or "[]"),
                 ttfa_ms=r[6], interrupted=bool(r[7])) for r in rows]


def delete(profile: str, sid: int | None = None) -> int:
    """Delete one session (sid) or all of this person's sessions; returns the number of sessions removed."""
    c = _db(profile)
    with _lock, c:
        if sid is None:
            n = c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            c.execute("DELETE FROM turns"); c.execute("DELETE FROM sessions")
        else:
            n = c.execute("SELECT COUNT(*) FROM sessions WHERE id=?", (sid,)).fetchone()[0]
            c.execute("DELETE FROM turns WHERE session_id=?", (sid,)); c.execute("DELETE FROM sessions WHERE id=?", (sid,))
    return n
