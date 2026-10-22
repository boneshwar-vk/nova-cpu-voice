"""Nova's tools: 15 actions the local model can use (10 offline, 5 online when the computer has internet).

Two ways a tool is chosen, fastest first:
  1. route(profile, text): deterministic rules for the common requests ("set a timer for 5 minutes",
     "what's 18% of 2499", "weather in Paris tomorrow") -> (name, args) in < 1 ms, no model call.
  2. Gemma 4's native function calling: declarations() goes into the system turn as
     <|tool>declaration:...<tool|>; a reply that starts with <|tool_call>call:name{...}<tool_call|>
     is parsed by parse_call().
run(name, args, ctx) executes a tool and returns a small dict; note(name, result) turns it into the line
the model answers from. Online tools need the internet (checked by online()); without it they return
{"error": "offline"} and Nova says so. Online tools use keyless public APIs only (Open-Meteo, Wikipedia,
DuckDuckGo instant answers, Google News RSS, Frankfurter exchange rates). A child-safe profile (the "kids"
template) gets only the child_ok tools: no web search, news or Wikipedia.
Per-profile data (reminders, calendar, notes, timers) lives in data/<profile>_tools.sqlite on this computer.
Environment: NOVA_TZ (IANA time zone, default: the system's), NOVA_HOME_CITY (default city for weather),
NOVA_NEWS_LOCALE (default en-US), NOVA_CURRENCY (default USD).
"""
from __future__ import annotations

import ast
import datetime as dt
import json
import math
import operator
import os
import re
import socket
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
DIR = Path(os.environ.get("MEMORY_DIR", HERE / "data"))


def _local_tz() -> dt.tzinfo:
    name = os.environ.get("NOVA_TZ")
    if name:
        try:
            return ZoneInfo(name)
        except Exception:  # noqa: BLE001
            pass
    return dt.datetime.now().astimezone().tzinfo or dt.timezone.utc


LOCAL_TZ = _local_tz()
HOME_CITY = os.environ.get("NOVA_HOME_CITY", "")
NEWS_LOCALE = os.environ.get("NOVA_NEWS_LOCALE", "en-US")  # Google News language-region, e.g. en-GB, en-IN, de-DE
HOME_CURRENCY = os.environ.get("NOVA_CURRENCY", "USD").upper()  # default target of "what's the exchange rate"
UA = "nova-cpu-voice/1.0 (local voice assistant)"
HTTP_TIMEOUT = 4.0

# ───────────────────────────── per-person store ─────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders(id INTEGER PRIMARY KEY, text TEXT NOT NULL, due REAL, due_text TEXT, done INTEGER DEFAULT 0, created REAL);
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, title TEXT NOT NULL, start REAL, when_text TEXT, created REAL);
CREATE TABLE IF NOT EXISTS notes(id INTEGER PRIMARY KEY, text TEXT NOT NULL, created REAL);
CREATE TABLE IF NOT EXISTS timers(id INTEGER PRIMARY KEY, label TEXT, seconds REAL, ends REAL, done INTEGER DEFAULT 0, created REAL);
"""
_dbs: dict[str, sqlite3.Connection] = {}
_lock = threading.Lock()


def db(profile: str) -> sqlite3.Connection:
    with _lock:
        c = _dbs.get(profile)
        if c is None:
            DIR.mkdir(parents=True, exist_ok=True)
            c = sqlite3.connect(str(DIR / f"{profile}_tools.sqlite"), check_same_thread=False)
            c.executescript(_SCHEMA)
            _dbs[profile] = c
        return c


def _q(profile: str, sql: str, args: tuple = ()) -> list[tuple]:
    c = db(profile)
    with _lock, c:
        return c.execute(sql, args).fetchall()


def _x(profile: str, sql: str, args: tuple = ()) -> int:
    c = db(profile)
    with _lock, c:
        return c.execute(sql, args).lastrowid


# ───────────────────────────── time phrases ─────────────────────────────

_WD = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_MON = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
_NUMW = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
         "forty five": 45, "forty-five": 45, "sixty": 60, "ninety": 90, "half an": 0.5, "half a": 0.5}
_NUM = r"(\d+(?:\.\d+)?|half an|half a|forty[- ]five|" + "|".join(sorted((k for k in _NUMW if " " not in k and "-" not in k), key=len, reverse=True)) + ")"


def _num(s: str) -> float:
    s = s.lower().strip()
    return float(s) if re.match(r"^\d", s) else float(_NUMW.get(s, _NUMW.get(s.replace("-", " "), 0)))


def now() -> dt.datetime:
    return dt.datetime.now(LOCAL_TZ)


def parse_when(text: str, base: dt.datetime | None = None) -> tuple[dt.datetime | None, str]:
    """'tomorrow at 5 pm', 'in 20 minutes', 'on Friday', 'on 3 May at 9', 'tonight' -> (datetime, phrase)."""
    t = words_to_digits((text or "").lower())  # "at eight" -> "at 8"
    b = base or now()
    m = re.search(_NUM + r"\s*(minute|min|hour|hr|day|week)s?\b", t)
    if m and re.search(r"\b(in|after)\b", t[: m.start() + 1] + " in"):
        n, unit = _num(m.group(1)), m.group(2)
        delta = {"minute": dt.timedelta(minutes=n), "min": dt.timedelta(minutes=n), "hour": dt.timedelta(hours=n),
                 "hr": dt.timedelta(hours=n), "day": dt.timedelta(days=n), "week": dt.timedelta(weeks=n)}[unit]
        return b + delta, m.group(0)
    day = None
    phrase = []
    if "day after tomorrow" in t:
        day = b.date() + dt.timedelta(days=2); phrase.append("day after tomorrow")
    elif "tomorrow" in t:
        day = b.date() + dt.timedelta(days=1); phrase.append("tomorrow")
    elif re.search(r"\b(today|tonight|this evening|this afternoon|this morning)\b", t):
        day = b.date(); phrase.append(re.search(r"\b(today|tonight|this evening|this afternoon|this morning)\b", t).group(1))
    else:
        m = re.search(r"\b(next )?(" + "|".join(_WD) + r")\b", t)
        if m:
            k = (_WD.index(m.group(2)) - b.weekday()) % 7 or 7
            if m.group(1):
                k = k if k > 0 else 7
            day = b.date() + dt.timedelta(days=k); phrase.append(m.group(0))
        m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + "|".join(_MON) + r")[a-z]*\b", t) or \
            re.search(r"\b(" + "|".join(_MON) + r")[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?\b", t)
        if m:
            g = m.groups()
            d_, mon = (int(g[0]), g[1]) if g[0].isdigit() else (int(g[1]), g[0])
            mi = _MON.index(mon[:3]) + 1
            try:
                cand = dt.date(b.year, mi, d_)
                if cand < b.date():
                    cand = dt.date(b.year + 1, mi, d_)
                day = cand; phrase.append(m.group(0))
            except ValueError:
                pass
    hh = mm = None
    m = re.search(r"\b(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)", t) or re.search(r"\bat\s+(\d{1,2})(?::(\d{2}))?\b", t)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2) or 0)
        ap = (m.group(3) or "").replace(".", "") if m.lastindex and m.lastindex >= 3 else ""
        if ap == "pm" and hh < 12:
            hh += 12
        elif ap == "am" and hh == 12:
            hh = 0
        elif not ap and hh < 12 and day is None:  # "at 8" with no am/pm: the next 8 o'clock (morning or evening)
            cand = [b.replace(hour=h, minute=mm, second=0, microsecond=0) for h in (hh, hh + 12)]
            hh = next((c.hour for c in cand if c > b), hh)
        elif not ap and hh < 8:  # "tomorrow at 5" means the afternoon/evening, not 5 am
            hh += 12
        phrase.append(m.group(0).strip())
    elif re.search(r"\btonight\b|\bthis evening\b", t):
        hh, mm = 20, 0
    elif "morning" in t:
        hh, mm = 9, 0
    elif "afternoon" in t:
        hh, mm = 15, 0
    elif "evening" in t:
        hh, mm = 18, 0
    if day is None and hh is None:
        return None, ""
    day = day or b.date()
    if hh is None:
        hh, mm = 9, 0
    out = dt.datetime(day.year, day.month, day.day, hh, mm, tzinfo=LOCAL_TZ)
    if out < b and "today" not in phrase and len(phrase) == 1 and phrase[0].startswith("at"):
        out += dt.timedelta(days=1)  # "at 7" said after 7 means tomorrow
    return out, " ".join(phrase)


def say_when(d: dt.datetime) -> str:
    today = now().date()
    if d.date() == today:
        day = "today"
    elif d.date() == today + dt.timedelta(days=1):
        day = "tomorrow"
    elif 0 < (d.date() - today).days < 7:
        day = d.strftime("%A")
    else:
        day = d.strftime("%A, %-d %B")
    return f"{day} at {d.strftime('%-I:%M %p').replace(':00', '')}"


# ───────────────────────────── tool definitions ─────────────────────────────

@dataclass
class Tool:
    name: str
    description: str
    params: dict  # name -> {"type": "STRING|NUMBER", "description": str, "enum": [...]?}
    required: list
    run: Callable[[dict, dict], dict]
    online: bool = False
    child_ok: bool = True
    leadin: Callable[[dict, dict], str] | None = None  # spoken while a slow (online) tool runs
    examples: list = field(default_factory=list)


def _ok(**kw) -> dict:
    return {"ok": True, **kw}


def _err(msg: str, **kw) -> dict:
    return {"ok": False, "error": msg, **kw}


# 1. date and time
_CITY_TZ = {"london": "Europe/London", "uk": "Europe/London", "new york": "America/New_York", "usa": "America/New_York",
            "san francisco": "America/Los_Angeles", "california": "America/Los_Angeles", "seattle": "America/Los_Angeles",
            "dubai": "Asia/Dubai", "singapore": "Asia/Singapore", "tokyo": "Asia/Tokyo", "japan": "Asia/Tokyo",
            "sydney": "Australia/Sydney", "berlin": "Europe/Berlin", "germany": "Europe/Berlin", "paris": "Europe/Paris",
            "france": "Europe/Paris", "toronto": "America/Toronto", "hong kong": "Asia/Hong_Kong", "beijing": "Asia/Shanghai",
            "china": "Asia/Shanghai", "moscow": "Europe/Moscow", "nairobi": "Africa/Nairobi", "kathmandu": "Asia/Kathmandu",
            "dhaka": "Asia/Dhaka", "colombo": "Asia/Colombo", "india": "Asia/Kolkata"}


def t_datetime(a: dict, ctx: dict) -> dict:
    place = (a.get("place") or "").strip().lower()
    tz = ZoneInfo(_CITY_TZ[place]) if place in _CITY_TZ else LOCAL_TZ
    d = dt.datetime.now(tz)
    where = place.title() if place in _CITY_TZ else "here"
    return _ok(place=where, time=d.strftime("%-I:%M %p"), date=d.strftime("%A, %-d %B %Y"), timezone=str(tz))


# 2. timers
def t_timer(a: dict, ctx: dict) -> dict:
    p = ctx["profile"]
    act = (a.get("action") or "set").lower()
    if act == "list":
        rows = _q(p, "SELECT label, ends FROM timers WHERE done=0 AND ends > ? ORDER BY ends", (time.time(),))
        return _ok(timers=[{"label": l or "timer", "minutes_left": round((e - time.time()) / 60, 1)} for l, e in rows])
    if act == "cancel":
        n = len(_q(p, "SELECT id FROM timers WHERE done=0 AND ends > ?", (time.time(),)))
        _x(p, "UPDATE timers SET done=1 WHERE done=0")
        return _ok(cancelled=n)
    secs = float(a.get("minutes") or 0) * 60 + float(a.get("seconds") or 0)
    if secs <= 0 or secs > 24 * 3600:
        return _err("the timer needs a length between a few seconds and a day")
    label = (a.get("label") or "").strip()[:40]
    tid = _x(p, "INSERT INTO timers(label, seconds, ends, created) VALUES (?,?,?,?)", (label, secs, time.time() + secs, time.time()))
    return _ok(id=tid, label=label or "timer", length=_say_dur(secs), ends_at=(now() + dt.timedelta(seconds=secs)).strftime("%-I:%M %p"))


def _say_dur(secs: float) -> str:
    secs = int(round(secs))
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    parts = ([f"{h} hour{'s' * (h != 1)}"] if h else []) + ([f"{m} minute{'s' * (m != 1)}"] if m else []) + ([f"{s} second{'s' * (s != 1)}"] if s else [])
    return " and ".join(parts) or "0 seconds"


def due_timers(profile: str) -> list[dict]:
    """Timers that just finished (marked done here, so each is announced once)."""
    t = time.time()
    rows = _q(profile, "SELECT id, label, seconds, ends FROM timers WHERE done=0 AND ends <= ?", (t,))
    for r in rows:
        _x(profile, "UPDATE timers SET done=1 WHERE id=?", (r[0],))
    # a timer that ran out while nobody was connected is not announced minutes later
    return [{"id": i, "label": l or "", "length": _say_dur(s)} for i, l, s, e in rows if t - e <= 120]


# 3. reminders
def t_reminders(a: dict, ctx: dict) -> dict:
    p = ctx["profile"]
    act = (a.get("action") or "add").lower()
    if act == "list":
        rows = _q(p, "SELECT text, due FROM reminders WHERE done=0 ORDER BY COALESCE(due, 9e18) LIMIT 8")
        return _ok(reminders=[{"text": t, "when": say_when(dt.datetime.fromtimestamp(d, LOCAL_TZ)) if d else "no time set"} for t, d in rows])
    if act == "done":
        key = (a.get("text") or "").lower()
        rows = _q(p, "SELECT id, text FROM reminders WHERE done=0")
        hit = [r for r in rows if key and any(w in r[1].lower() for w in key.split() if len(w) > 3)]
        for r in hit[:1]:
            _x(p, "UPDATE reminders SET done=1 WHERE id=?", (r[0],))
        return _ok(done=[r[1] for r in hit[:1]]) if hit else _err("no matching reminder")
    text = (a.get("text") or "").strip()
    if not text:
        return _err("what should I remind you about?")
    when, phrase = parse_when(a.get("when") or text)
    _x(p, "INSERT INTO reminders(text, due, due_text, created) VALUES (?,?,?,?)", (text, when.timestamp() if when else None, phrase, time.time()))
    return _ok(reminder=text, when=say_when(when) if when else "no time set")


# 4. calendar
def t_calendar(a: dict, ctx: dict) -> dict:
    p = ctx["profile"]
    act = (a.get("action") or "list").lower()
    if act == "add":
        title = (a.get("title") or "").strip()
        when, phrase = parse_when(a.get("when") or title)
        if not title or not when:
            return _err("I need what the event is and when it is")
        _x(p, "INSERT INTO events(title, start, when_text, created) VALUES (?,?,?,?)", (title, when.timestamp(), phrase, time.time()))
        return _ok(added=title, when=say_when(when))
    rng = (a.get("range") or "this week").lower()
    t0 = now().replace(hour=0, minute=0, second=0, microsecond=0)
    span = {"today": 1, "tomorrow": 2, "this week": 7, "next week": 14, "this month": 31}.get(rng, 7)
    start = t0 + (dt.timedelta(days=1) if rng == "tomorrow" else dt.timedelta(days=7) if rng == "next week" else dt.timedelta(0))
    end = t0 + dt.timedelta(days=span)
    ev = _q(p, "SELECT title, start FROM events WHERE start >= ? AND start < ? ORDER BY start", (start.timestamp(), end.timestamp()))
    rem = _q(p, "SELECT text, due FROM reminders WHERE done=0 AND due >= ? AND due < ? ORDER BY due", (start.timestamp(), end.timestamp()))
    items = [{"what": t, "when": say_when(dt.datetime.fromtimestamp(s, LOCAL_TZ))} for t, s in ev] + \
            [{"what": "reminder: " + t, "when": say_when(dt.datetime.fromtimestamp(s, LOCAL_TZ))} for t, s in rem]
    return _ok(range=rng, items=items, note="memory facts about plans are given separately")


# 5. notes / to-do list
def t_notes(a: dict, ctx: dict) -> dict:
    p = ctx["profile"]
    act = (a.get("action") or "add").lower()
    if act == "list":
        rows = _q(p, "SELECT text FROM notes ORDER BY id DESC LIMIT 8")
        return _ok(notes=[r[0] for r in rows])
    if act == "clear":
        n = len(_q(p, "SELECT id FROM notes"))
        _x(p, "DELETE FROM notes")
        return _ok(cleared=n)
    text = (a.get("text") or "").strip()
    if not text:
        return _err("what should I write down?")
    _x(p, "INSERT INTO notes(text, created) VALUES (?,?)", (text[:300], time.time()))
    return _ok(saved=text[:300], total=len(_q(p, "SELECT id FROM notes")))


# 6. calculator
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos, ast.FloorDiv: operator.floordiv}
_FUN = {"sqrt": math.sqrt, "log": math.log10, "ln": math.log, "sin": lambda x: math.sin(math.radians(x)),
        "cos": lambda x: math.cos(math.radians(x)), "tan": lambda x: math.tan(math.radians(x)), "abs": abs, "round": round}


def _safe_eval(node):
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        l, r = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(r) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](l, r)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUN and len(node.args) == 1:
        return _FUN[node.func.id](_safe_eval(node.args[0]))
    if isinstance(node, ast.Name) and node.id in ("pi", "e"):
        return getattr(math, node.id)
    raise ValueError("not a sum I can do")


def normalize_expr(s: str) -> str:
    s = s.lower().replace(",", "").replace("×", "*").replace("÷", "/").replace("^", "**")
    s = re.sub(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s*(\d+(?:\.\d+)?)", r"(\1/100*\2)", s)
    for w, o in ((r"\bplus\b", "+"), (r"\bminus\b", "-"), (r"\b(times|multiplied by|into)\b", "*"), (r"\b(divided by|by|over)\b", "/"),
                 (r"\bsquared\b", "**2"), (r"\bcubed\b", "**3"), (r"\bsquare root of\b", "sqrt"), (r"\bto the power of\b", "**")):
        s = re.sub(w, o, s)
    s = re.sub(r"sqrt\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", s)
    s = re.sub(r"[^0-9+\-*/().%a-z ]", " ", s)
    s = re.sub(r"\b(what|is|whats|calculate|compute|equals|equal|the|of|answer|please|nova|how much)\b", " ", s)
    return " ".join(s.split())


def t_calculate(a: dict, ctx: dict) -> dict:
    expr = normalize_expr(str(a.get("expression") or ""))
    try:
        v = _safe_eval(ast.parse(expr, mode="eval"))
    except Exception:  # noqa: BLE001
        return _err("I couldn't read that as a calculation", expression=expr)
    if isinstance(v, float):
        v = round(v, 6)
        if v.is_integer():
            v = int(v)
    return _ok(expression=expr, result=v)


# 7. unit conversion
_UNITS = {  # to a base unit per dimension
    "length": {"mm": .001, "cm": .01, "m": 1, "metre": 1, "meter": 1, "km": 1000, "kilometre": 1000, "kilometer": 1000,
               "inch": .0254, "in": .0254, "foot": .3048, "feet": .3048, "ft": .3048, "yard": .9144, "mile": 1609.344},
    "mass": {"mg": 1e-6, "g": .001, "gram": .001, "kg": 1, "kilogram": 1, "tonne": 1000, "lb": .45359237, "pound": .45359237,
             "oz": .0283495, "ounce": .0283495},
    "volume": {"ml": .001, "millilitre": .001, "milliliter": .001, "l": 1, "litre": 1, "liter": 1, "cup": .24,
               "gallon": 3.78541, "tablespoon": .015, "teaspoon": .005},
    "speed": {"kmh": 1, "km/h": 1, "kph": 1, "mph": 1.609344, "m/s": 3.6},
    "time": {"second": 1, "minute": 60, "hour": 3600, "day": 86400, "week": 604800},
}


def _unit(u: str) -> tuple[str, float] | None:
    u = u.lower().strip().rstrip(".")
    u = {"kilometres": "km", "kilometers": "km", "metres": "m", "meters": "m", "centimetres": "cm", "centimeters": "cm",
         "inches": "inch", "miles": "mile", "yards": "yard", "grams": "g", "kilograms": "kg", "kilos": "kg", "kgs": "kg",
         "pounds": "lb", "lbs": "lb", "ounces": "oz", "litres": "l", "liters": "l", "cups": "cup", "gallons": "gallon",
         "seconds": "second", "minutes": "minute", "hours": "hour", "days": "day", "weeks": "week",
         "millilitres": "ml", "milliliters": "ml", "tablespoons": "tablespoon", "teaspoons": "teaspoon"}.get(u, u)
    for dim, tab in _UNITS.items():
        if u in tab:
            return dim, tab[u]
    return None


def t_convert(a: dict, ctx: dict) -> dict:
    v, f, t = a.get("value"), str(a.get("from_unit") or ""), str(a.get("to_unit") or "")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return _err("I need a number to convert")
    temps = {"c": "c", "celsius": "c", "centigrade": "c", "f": "f", "fahrenheit": "f", "k": "k", "kelvin": "k"}
    if f.lower() in temps and t.lower() in temps:
        fu, tu = temps[f.lower()], temps[t.lower()]
        c = v if fu == "c" else (v - 32) * 5 / 9 if fu == "f" else v - 273.15
        out = c if tu == "c" else c * 9 / 5 + 32 if tu == "f" else c + 273.15
        return _ok(value=v, from_unit=f, to_unit=t, result=round(out, 2))
    a1, a2 = _unit(f), _unit(t)
    if not a1 or not a2 or a1[0] != a2[0]:
        return _err(f"I can't convert {f} to {t}")
    return _ok(value=v, from_unit=f, to_unit=t, result=round(v * a1[1] / a2[1], 4))


# 8. forget a memory
def t_forget(a: dict, ctx: dict) -> dict:
    import memory_store
    what = (a.get("fact") or "").strip()
    if not what:
        return _err("what should I forget?")
    store = memory_store.get(ctx["profile"])
    facts, _ = store.retrieve(what, count=False)
    if not facts or float(facts[0].get("cos", 0)) < 0.45:
        return _err("I couldn't find that in what I remember")
    f = facts[0]
    store.delete(f["id"])
    return _ok(forgot=f["text"])


# 9. device control (mute, volume, speed, voice) -> applied by the server/page
# Spoken names for the voices in open_tts.VOICES (only installed voices can be chosen).
VOICE_ALIASES = {"ljspeech": "ljspeech", "lj speech": "ljspeech", "lj": "ljspeech", "us": "ljspeech", "american": "ljspeech",
                 "sef": "sef", "southern english": "sef", "british": "sef", "uk": "sef", "english": "sef",
                 "matcha": "matcha"}


def voices_available() -> list[str]:
    """Voice ids whose model files are installed (open_tts.available)."""
    try:
        import open_tts
        return [v for v in open_tts.VOICES if open_tts.available(v)]
    except Exception:  # noqa: BLE001
        return []


def _voice_names(ids: list[str]) -> str:
    try:
        import open_tts
        names = [open_tts.VOICES[v]["display"].split(" (")[0] for v in ids]
    except Exception:  # noqa: BLE001
        names = list(ids)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1] if names else "none"


def t_device(a: dict, ctx: dict) -> dict:
    act = (a.get("action") or "").lower()
    allowed = {"mute", "unmute", "louder", "quieter", "slower", "faster", "normal_speed", "voice"}
    if act not in allowed:
        return _err("I can mute, change the volume, speak slower or faster, or change my voice")
    out = {"action": act}
    if act == "voice":
        raw = re.sub(r"\s+voice$", "", (a.get("voice") or "").lower().strip())
        if raw not in VOICE_ALIASES:  # speech recognition near-misses ("matcher" for "matcha")
            import difflib
            close = difflib.get_close_matches(raw, list(VOICE_ALIASES), n=1, cutoff=0.75)
            raw = close[0] if close else raw
        v = VOICE_ALIASES.get(raw, raw)
        have = voices_available()
        if v not in have:
            what = "that voice isn't installed" if v in VOICE_ALIASES.values() else f"I don't have a voice called {raw}"
            return _err(f"{what}; the voices I have are {_voice_names(have)}", voices=have)
        out["voice"] = v
    return _ok(**out)


# 10. library search (explicit: "search my papers for discrete flow matching")
def t_library(a: dict, ctx: dict) -> dict:
    import rag_store
    q = (a.get("query") or "").strip()
    if not q:
        return _err("what should I look for?")
    r = rag_store.get(ctx["profile"]).search(q, budget_tokens=200)
    if not r.get("passages"):
        return _err("nothing found in the library")
    return _ok(query=q, notes=r["context"][:1200], sources=[f"{p.get('short_title') or p.get('title')} p. {p.get('page_label') or p.get('page')}"
                                                          for p in r["passages"][:3]])


# ── online tools ──
_online_cache = {"t": 0.0, "v": False}


def online() -> bool:
    """Is the internet reachable right now? (cached 15 s; a 0.8 s TCP connect to a public DNS resolver)."""
    if time.time() - _online_cache["t"] < 15:
        return _online_cache["v"]
    ok = False
    for host in ("1.1.1.1", "8.8.8.8"):
        try:
            with socket.create_connection((host, 53), timeout=0.8):
                ok = True
                break
        except OSError:
            continue
    _online_cache.update(t=time.time(), v=ok)
    return ok


def _get_json(url: str, timeout: float = HTTP_TIMEOUT) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _get_text(url: str, timeout: float = HTTP_TIMEOUT) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


_WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog", 51: "light drizzle",
        53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
        71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers", 81: "rain showers", 82: "heavy rain showers",
        95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail"}


# 11. weather (Open-Meteo: free, no key)
def t_weather(a: dict, ctx: dict) -> dict:
    city = (a.get("city") or "").strip() or HOME_CITY
    if not city:
        return _err("which city? (set NOVA_HOME_CITY for a default)")
    day = (a.get("day") or "today").lower()
    g = _get_json("https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode({"name": city, "count": 1, "language": "en"}))
    if not g.get("results"):
        return _err(f"I couldn't find a place called {city}")
    r = g["results"][0]
    f = _get_json("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
        "latitude": r["latitude"], "longitude": r["longitude"], "timezone": "auto", "forecast_days": 3,
        "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code"}))
    i = 1 if day == "tomorrow" else 0
    d = f["daily"]
    out = dict(place=r["name"], day=day, high_c=round(d["temperature_2m_max"][i]), low_c=round(d["temperature_2m_min"][i]),
               rain_chance_pct=d["precipitation_probability_max"][i], sky=_WMO.get(d["weather_code"][i], "mixed"))
    if i == 0:
        c = f["current"]
        out.update(now_c=round(c["temperature_2m"]), feels_like_c=round(c["apparent_temperature"]),
                   now_sky=_WMO.get(c["weather_code"], "mixed"), humidity_pct=c["relative_humidity_2m"], wind_kmh=round(c["wind_speed_10m"]))
    return _ok(**out, source="Open-Meteo")


# 12. web search (DuckDuckGo Instant Answer API + Wikipedia search; free, no key)
def t_search(a: dict, ctx: dict) -> dict:
    q = (a.get("query") or "").strip()
    if not q:
        return _err("what should I search for?")
    hits = []
    try:
        j = _get_json("https://api.duckduckgo.com/?" + urllib.parse.urlencode({"q": q, "format": "json", "no_html": 1, "skip_disambig": 1}))
        if j.get("Answer"):
            hits.append({"title": "Instant answer", "text": str(j["Answer"])[:300], "source": "DuckDuckGo"})
        if j.get("AbstractText"):
            hits.append({"title": j.get("Heading") or q, "text": j["AbstractText"][:500], "source": j.get("AbstractSource") or "DuckDuckGo"})
        for t in (j.get("RelatedTopics") or [])[:3]:
            if isinstance(t, dict) and t.get("Text"):
                hits.append({"title": t.get("Text", "")[:60], "text": t["Text"][:240], "source": "DuckDuckGo"})
    except Exception:  # noqa: BLE001
        pass
    if re.search(r"\b(won|win|score|result|latest|current|today|yesterday|last night|this week|price|news)\b", q, re.I):
        try:  # current events: matching news headlines are the freshest free source
            n = t_news({"topic": re.sub(r"^(who|what|when|where)( is| was| are)?( the)? ", "", q, flags=re.I)}, ctx)
            hits = [{"title": h["headline"], "text": h["headline"], "source": h.get("source") or "Google News"}
                    for h in n.get("headlines", [])[:3]] + hits
        except Exception:  # noqa: BLE001
            pass
    if len(hits) < 2:
        try:
            s = _get_json("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
                {"action": "query", "list": "search", "srsearch": q, "format": "json", "srlimit": 3}))
            for h in s.get("query", {}).get("search", [])[:3]:
                hits.append({"title": h["title"], "text": re.sub(r"<[^>]+>", "", h.get("snippet", ""))[:240], "source": "Wikipedia"})
        except Exception:  # noqa: BLE001
            pass
    return _ok(query=q, results=hits[:4]) if hits else _err("the search found nothing useful")


# 13. Wikipedia summary
def t_wikipedia(a: dict, ctx: dict) -> dict:
    topic = (a.get("topic") or "").strip()
    if not topic:
        return _err("which topic?")
    s = _get_json("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "opensearch", "search": topic, "limit": 1, "namespace": 0, "format": "json"}))
    title = (s[1] or [None])[0] if isinstance(s, list) and len(s) > 1 else None
    if not title:
        return _err(f"Wikipedia has no page on {topic}")
    j = _get_json("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_")))
    return _ok(title=j.get("title", title), summary=(j.get("extract") or "")[:700], source="Wikipedia")


# 14. news headlines (Google News RSS)
def t_news(a: dict, ctx: dict) -> dict:
    topic = (a.get("topic") or "").strip()
    base = "https://news.google.com/rss"
    hl, _, gl = NEWS_LOCALE.partition("-")
    loc = {"hl": NEWS_LOCALE, "gl": gl or "US", "ceid": f"{gl or 'US'}:{hl or 'en'}"}
    url = base + ("/search?" + urllib.parse.urlencode({"q": topic, **loc}) if topic else "?" + urllib.parse.urlencode(loc))
    import xml.etree.ElementTree as ET
    root = ET.fromstring(_get_text(url))
    items = []
    for it in root.iter("item"):
        t = (it.findtext("title") or "").strip()
        src = (it.findtext("source") or "").strip()
        if src and t.endswith(" - " + src):
            t = t[: -len(" - " + src)]
        items.append({"headline": t, "source": src})
        if len(items) >= 4:
            break
    return _ok(topic=topic or "top stories", headlines=items, source="Google News") if items else _err("no headlines found")


# 15. currency (Frankfurter, ECB rates: free, no key)
_CUR = {"rupee": "INR", "rupees": "INR", "inr": "INR", "dollar": "USD", "dollars": "USD", "usd": "USD", "euro": "EUR", "euros": "EUR",
        "eur": "EUR", "pound": "GBP", "pounds": "GBP", "gbp": "GBP", "yen": "JPY", "jpy": "JPY", "dirham": "AED", "aed": "AED",
        "singapore dollar": "SGD", "sgd": "SGD", "australian dollar": "AUD", "aud": "AUD", "yuan": "CNY", "cny": "CNY",
        "canadian dollar": "CAD", "cad": "CAD", "swiss franc": "CHF", "chf": "CHF"}


def t_currency(a: dict, ctx: dict) -> dict:
    try:
        amt = float(a.get("amount") or 1)
    except (TypeError, ValueError):
        amt = 1.0
    f = _CUR.get(str(a.get("from_currency") or "").lower(), str(a.get("from_currency") or "").upper())
    t = _CUR.get(str(a.get("to_currency") or "USD").lower(), str(a.get("to_currency") or "USD").upper())
    if f == "AED" or t == "AED":
        return _err("I don't have dirham rates from my rate source")
    j = _get_json("https://api.frankfurter.app/latest?" + urllib.parse.urlencode({"amount": amt, "from": f, "to": t}))
    v = (j.get("rates") or {}).get(t)
    if v is None:
        return _err(f"I couldn't convert {f} to {t}")
    return _ok(amount=amt, from_currency=f, to_currency=t, result=round(v, 2), rate_date=j.get("date"), source="Frankfurter (ECB rates)")


def _lead_weather(a, ctx):
    return f"Checking the weather in {a.get('city') or HOME_CITY or 'your city'}."


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("get_datetime", "Current time and date, here or in a city abroad.",
         {"place": {"type": "STRING", "description": "city or country, empty for here"}}, [], t_datetime),
    Tool("timer", "Set, list or cancel a countdown timer; Nova speaks when it ends.",
         {"action": {"type": "STRING", "description": "set, list or cancel", "enum": ["set", "list", "cancel"]},
          "minutes": {"type": "NUMBER", "description": "length in minutes"}, "seconds": {"type": "NUMBER", "description": "extra seconds"},
          "label": {"type": "STRING", "description": "what the timer is for"}}, ["action"], t_timer),
    Tool("reminders", "Add, list or complete the person's reminders.",
         {"action": {"type": "STRING", "description": "add, list or done", "enum": ["add", "list", "done"]},
          "text": {"type": "STRING", "description": "what to remember to do"}, "when": {"type": "STRING", "description": "e.g. tomorrow at 5 pm"}},
         ["action"], t_reminders),
    Tool("calendar", "Add an event, or list events for today, tomorrow, this week or next week.",
         {"action": {"type": "STRING", "description": "add or list", "enum": ["add", "list"]},
          "title": {"type": "STRING", "description": "event name"}, "when": {"type": "STRING", "description": "date and time"},
          "range": {"type": "STRING", "description": "today, tomorrow, this week or next week"}}, ["action"], t_calendar),
    Tool("notes", "Write down, read back or clear the person's notes and to-do list.",
         {"action": {"type": "STRING", "description": "add, list or clear", "enum": ["add", "list", "clear"]},
          "text": {"type": "STRING", "description": "the note"}}, ["action"], t_notes),
    Tool("calculate", "Exact arithmetic and percentages.",
         {"expression": {"type": "STRING", "description": "e.g. 18% of 2499 or 12*7+3"}}, ["expression"], t_calculate),
    Tool("convert_units", "Convert length, weight, volume, speed, time or temperature.",
         {"value": {"type": "NUMBER", "description": "amount"}, "from_unit": {"type": "STRING", "description": "e.g. km"},
          "to_unit": {"type": "STRING", "description": "e.g. miles"}}, ["value", "from_unit", "to_unit"], t_convert),
    Tool("forget_memory", "Forget one thing Nova remembers about the person.",
         {"fact": {"type": "STRING", "description": "what to forget"}}, ["fact"], t_forget, child_ok=False),
    Tool("device_control", "Mute, volume, speaking speed or Nova's voice (voices: " + ", ".join(voices_available() or ["ljspeech"]) + ").",
         {"action": {"type": "STRING", "description": "mute, unmute, louder, quieter, slower, faster, normal_speed or voice",
                     "enum": ["mute", "unmute", "louder", "quieter", "slower", "faster", "normal_speed", "voice"]},
          "voice": {"type": "STRING", "description": " or ".join(voices_available() or ["ljspeech"])}}, ["action"], t_device),
    Tool("search_library", "Search the person's own documents on this computer.",
         {"query": {"type": "STRING", "description": "what to look for"}}, ["query"], t_library),
    Tool("get_weather", "Weather now, today or tomorrow for a city (needs internet).",
         {"city": {"type": "STRING", "description": "city, empty for the person's home city"},
          "day": {"type": "STRING", "description": "today or tomorrow", "enum": ["today", "tomorrow"]}}, [], t_weather,
         online=True, leadin=_lead_weather),
    Tool("web_search", "Search the web for current facts (needs internet).",
         {"query": {"type": "STRING", "description": "search words"}}, ["query"], t_search, online=True, child_ok=False,
         leadin=lambda a, c: "Let me look that up online."),
    Tool("wikipedia", "Short Wikipedia summary of a person, place or topic (needs internet).",
         {"topic": {"type": "STRING", "description": "the topic"}}, ["topic"], t_wikipedia, online=True, child_ok=False,
         leadin=lambda a, c: f"Checking Wikipedia for {a.get('topic') or 'that'}."),
    Tool("get_news", "Latest news headlines, optionally on a topic (needs internet).",
         {"topic": {"type": "STRING", "description": "topic, empty for top stories"}}, [], t_news, online=True, child_ok=False,
         leadin=lambda a, c: f"Getting the latest headlines{(' on ' + a['topic']) if a.get('topic') else ''}."),
    Tool("currency_convert", "Convert money between currencies at today's rate (needs internet).",
         {"amount": {"type": "NUMBER", "description": "amount"}, "from_currency": {"type": "STRING", "description": "e.g. USD"},
          "to_currency": {"type": "STRING", "description": "e.g. EUR"}}, ["from_currency", "to_currency"], t_currency,
         online=True, child_ok=False, leadin=lambda a, c: "Checking today's exchange rate."),
]}


def is_child(profile: str | None) -> bool:
    try:
        import personas
        return personas.is_child(profile)
    except Exception:  # noqa: BLE001
        return False


def allowed(profile: str | None, child: bool = False) -> list[str]:
    """Tools this profile may use. A child-safe profile never gets web search, news, Wikipedia, currency or forget."""
    child = child or is_child(profile)
    return [n for n, t in TOOLS.items() if t.child_ok or not child]


# ───────────────────────────── running a tool ─────────────────────────────

def run(name: str, args: dict, ctx: dict) -> dict:
    """Execute a tool. Never raises: errors come back as {"ok": False, "error": ...}."""
    t = TOOLS.get(name)
    if t is None:
        return _err(f"unknown tool {name}")
    if name not in allowed(ctx.get("profile"), bool(ctx.get("child"))):
        return _err("that is not available in this profile")
    if t.online and not online():
        return _err("offline", detail="This needs the internet, and this computer is offline right now.")
    t0 = time.perf_counter()
    try:
        r = t.run(dict(args or {}), ctx)
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        r = _err("the online service did not answer in time", detail=str(e)[:120])
    except Exception as e:  # noqa: BLE001 — a tool must never break a turn
        r = _err("that did not work", detail=str(e)[:160])
    r["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    r["online"] = t.online
    return r


def confirm(name: str, result: dict) -> str:
    """A plain spoken confirmation from a tool's result, used if the model's answer is unusable."""
    r = result
    if not r.get("ok"):
        return "Sorry, I couldn't do that offline." if r.get("error") == "offline" else "Sorry, that didn't work."
    if name == "reminders" and r.get("reminder"):
        return f"Okay. I'll remind you to {r['reminder']}, {r.get('when', '')}.".replace(", no time set", "")
    if name == "timer" and r.get("length"):
        return f"Timer set for {r['length']}."
    if name == "calendar" and r.get("added"):
        return f"Added {r['added']}, {r.get('when', '')}."
    if name == "notes" and r.get("saved"):
        return "Noted."
    if name == "calculate" and "result" in r:
        return f"That's {r['result']}."
    if name == "convert_units" and "result" in r:
        return f"That's {r['result']} {r.get('to_unit', '')}."
    if name == "get_datetime":
        return f"It's {r.get('time')}, {r.get('date')}."
    return "Done."


def note(name: str, result: dict) -> str:
    """The line appended to the user's turn so the model answers from the tool's result."""
    r = {k: v for k, v in result.items() if k not in ("ms", "online", "ok")}
    if not result.get("ok"):
        if result.get("error") == "offline":
            return f"\n(Tool {name}: the computer is offline. Say you can't check that without internet, in one short sentence.)"
        return f"\n(Tool {name} failed: {result.get('error')}. Say so briefly and offer what you can do.)"
    return (f"\n(Tool {name} result: {json.dumps(r, ensure_ascii=False)[:1400]}\n"
            "Answer in one or two short spoken sentences using only this result. Say numbers naturally. "
            "Do not call any tool now, and do not mention JSON, tools or fields.)")


def leadin(name: str, args: dict, ctx: dict) -> str:
    t = TOOLS.get(name)
    return t.leadin(args, ctx) if t and t.leadin else ""


def card(name: str, args: dict, result: dict) -> dict:
    """A compact summary for the page's tool card."""
    t = TOOLS.get(name)
    title = {"get_datetime": "Clock", "timer": "Timer", "reminders": "Reminders", "calendar": "Calendar", "notes": "Notes",
             "calculate": "Calculator", "convert_units": "Unit converter", "forget_memory": "Memory", "device_control": "Device",
             "search_library": "Library", "get_weather": "Weather", "web_search": "Web search", "wikipedia": "Wikipedia",
             "get_news": "News", "currency_convert": "Currency"}.get(name, name)
    def short(v) -> str:
        if isinstance(v, list):
            parts = []
            for x in v[:3]:
                if isinstance(x, dict):
                    x = (x.get("headline") or x.get("what") or x.get("text") or x.get("title") or x.get("label") or "") + \
                        (f" · {x['when']}" if x.get("when") else "") + (f" ({x['source']})" if x.get("source") and x.get("headline") else "")
                parts.append(str(x)[:140])
            return " · ".join(parts) or "none"
        return str(v)[:240]
    skip = {"ms", "online", "ok", "notes", "id", "timezone", "note"}
    fields = {k.replace("_", " "): short(v) for k, v in result.items() if k not in skip}
    return {"name": name, "title": title, "args": args, "ok": bool(result.get("ok")), "online": bool(t and t.online),
            "ms": result.get("ms"), "fields": dict(list(fields.items())[:6]), "source": result.get("source", "")}


# ───────────────────────────── Gemma 4 native function calling ─────────────────────────────

Q = '<|"|>'  # Gemma 4's string quote token


def declarations(names: list[str], compact: bool = True) -> str:
    """Tool declarations in Gemma 4's chat-template format, appended to the system turn. Compact form (short
    descriptions, enums kept, parameter descriptions dropped) keeps 15 tools to a few hundred tokens of the KV cache."""
    out = []
    for n in names:
        t = TOOLS[n]
        props = ",".join(
            f"{k}:{{" + ("" if compact else f"description:{Q}{v['description']}{Q},")
            + (f"enum:[{','.join(Q + e + Q for e in v['enum'])}]," if v.get("enum") else "")
            + f"type:{Q}{v['type']}{Q}}}" for k, v in sorted(t.params.items()))
        req = ("required:[" + ",".join(Q + r + Q for r in t.required) + "],") if t.required else ""
        out.append(f"<|tool>declaration:{n}{{description:{Q}{t.description}{Q},parameters:{{properties:{{{props}}},{req}type:{Q}OBJECT{Q}}}}}<tool|>")
    return "".join(out)


_ASKED = {  # a model-chosen call only runs when the person actually asked for that kind of thing
    "timer": r"\b(timer|countdown|count down|alarm|wake me)\b",
    "reminders": r"\b(remind|reminder|reminders|wake me|don'?t let me forget)\b",
    "calendar": r"\b(calendar|schedule|appointment|meeting|event|plans?)\b",
    "notes": r"\b(note|notes|jot|write (it |this |that )?down|to-?do|list)\b",
    "forget_memory": r"\b(forget|delete|remove|erase)\b",
    "device_control": r"\b(slow|slower|fast|faster|speed|louder|quieter|volume|mute|unmute|voice|lower|softer)\b",
    "web_search": r"\b(latest|current|currently|today|tonight|yesterday|now|news|price|prices|score|won|win|recent|population|this (week|month|year)|"
                  r"search|online|google|internet|look (it |this )?up|web)\b",
    "wikipedia": r"\b(wikipedia|wiki|who is|who was|biography)\b",
    "get_news": r"\b(news|headlines?|latest|happening)\b",
    "currency_convert": r"\b(dollars?|euros?|pounds?|yen|rupees?|currency|exchange|rate|usd|inr|eur|gbp|\$|€|£)",
    "get_weather": r"\b(weather|rain|raining|sunny|hot|cold|temperature|forecast|umbrella|humid|wind)\b",
    "search_library": r"\b(search|look|find|check)\b",
}


def model_call_ok(name: str, user_text: str) -> bool:
    """Gemma sometimes calls a tool nobody asked for (a note for "my cat's name is Whiskers", a web search for a
    textbook question). Side effects and internet use need the request to mention them; read-only offline tools don't."""
    rx = _ASKED.get(name)
    return True if rx is None else bool(re.search(rx, user_text or "", re.I))


_CALL_RX = re.compile(r"<\|tool_call>\s*call:([a-z_]+)\{(.*?)\}\s*<tool_call\|>", re.S)


def parse_call(text: str) -> tuple[str, dict] | None:
    """'<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,day:<|"|>today<|"|>}<tool_call|>' -> ('get_weather', {...})."""
    m = _CALL_RX.search(text or "")
    if not m:
        m2 = re.search(r"call:([a-z_]+)\{(.*)\}", text or "", re.S)
        if not m2:
            return None
        m = m2
    name, body = m.group(1), m.group(2)
    if name not in TOOLS:
        return None
    args: dict = {}
    for k, v in re.findall(r"([a-z_]+):(" + re.escape(Q) + r".*?" + re.escape(Q) + r"|[-\d.]+|true|false)", body):
        v = v[len(Q):-len(Q)] if v.startswith(Q) else (float(v) if re.match(r"^-?\d", v) else v == "true")
        args[k] = v
    return name, args


# ───────────────────────────── fast rules ─────────────────────────────

_R = lambda p: re.compile(p, re.I)  # noqa: E731
_CITY = r"([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)?)"


_W1 = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                                   "sixteen seventeen eighteen nineteen".split())}
_W10 = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}


def words_to_digits(t: str) -> str:
    """'seven times eight' -> '7 times 8'; 'twenty five' -> '25'; 'three hundred' -> '300' (spoken maths)."""
    def num(m: re.Match) -> str:
        ws, total, cur = m.group(0).lower().replace("-", " ").split(), 0, 0
        for w in ws:
            if w in _W1:
                cur += _W1[w]
            elif w in _W10:
                cur += _W10[w]
            elif w == "hundred":
                cur = (cur or 1) * 100
            elif w == "thousand":
                total += (cur or 1) * 1000
                cur = 0
        return str(total + cur)
    words = "|".join(sorted(list(_W1) + list(_W10) + ["hundred", "thousand"], key=len, reverse=True))
    return re.sub(rf"\b(?:{words})(?:[\s-]+(?:and[\s-]+)?(?:{words}))*\b", num, t, flags=re.I)


def route(profile: str | None, text: str, child: bool = False) -> tuple[str, dict] | None:
    """Deterministic tool choice for common requests; None = no tool (or let the model decide)."""
    t = re.sub(r"\bto-morrow\b", "tomorrow", (text or "").strip(), flags=re.I)
    if re.search(r"\b(times|plus|minus|divided by|multiplied by|percent of|squared)\b", t, re.I):
        t = words_to_digits(t)
    tl = t.lower()
    ok = set(allowed(profile, child))

    def pick(name, args):
        return (name, args) if name in ok else None

    # timers
    tl = re.sub(r"\bcount ?down\b", "timer", tl)  # "put a 3 minute count down on"
    m = re.search(r"\b(?:set|start|put)\b.*?\btimer\b.*?\bfor\s+" + _NUM + r"\s*(minute|min|second|sec|hour)s?", tl) or \
        re.search(r"\b" + _NUM + r"[- ]?(minute|min|second|sec|hour)s?\s+timer\b", tl) or \
        re.search(r"\btimer\b.*?\b" + _NUM + r"\s*(minute|min|second|sec|hour)s?", tl)
    if m and not re.search(r"\b(cancel|stop|delete)\b", tl):
        n, u = _num(m.group(1)), m.group(2)
        lab = re.search(r"\bfor (?:the |my )?([a-z ]{3,30}?)(?:$|[.?!]| for \d)", tl.split("timer", 1)[-1])
        label = lab.group(1).strip() if lab and not re.search(r"\d|minute|second|hour", lab.group(1)) else ""
        return pick("timer", {"action": "set", "minutes": n * (60 if u.startswith("hour") else 1) if not u.startswith("s") else 0,
                              "seconds": n if u.startswith("s") else 0, "label": label})
    if re.search(r"\b(cancel|stop|delete)\b.*\btimers?\b", tl):
        return pick("timer", {"action": "cancel"})
    if re.search(r"\bhow (much time|long)\b.*\b(left|remaining)\b.*\btimer|\btimers?\b.*\b(left|running)\b", tl):
        return pick("timer", {"action": "list"})
    # reminders
    m = re.search(r"\bremind me (?:to |about |that )(.+)", t, re.I)
    if m:
        body = words_to_digits(m.group(1).strip().rstrip(".?!"))
        when, phrase = parse_when(body)
        text_ = re.sub(r"\s*\b(" + re.escape(phrase) + r")\b\s*", " ", body, flags=re.I).strip() if phrase else body
        text_ = re.sub(r"\s+(on|at|in|by)$", "", text_).strip()
        return pick("reminders", {"action": "add", "text": text_, "when": phrase})
    if re.search(r"\b(what are|list|read|show)\b.*\bmy reminders\b|\bany reminders\b|\bmy reminders\b", tl):
        return pick("reminders", {"action": "list"})
    # calendar
    m = re.search(r"\b(?:add|put|schedule)\b (.+?) (?:to|in|on) my (?:calendar|schedule)(.*)", t, re.I)
    if m:
        full = (m.group(1) + " " + m.group(2)).strip()
        _, phrase = parse_when(full)
        title = m.group(1)
        for part in (phrase.split(" ") if phrase else []):
            pass
        title = re.sub(r"\s+(on|at|this|next|tomorrow|today|tonight)\b.*$", "", title, flags=re.I).strip() or m.group(1).strip()
        return pick("calendar", {"action": "add", "title": title, "when": full})
    m = re.search(r"\b(?:what'?s|what is|what do i have|anything)\b.*\bon my (?:calendar|schedule)\b(.*)", tl) or \
        re.search(r"\bmy (?:calendar|schedule) (?:for )?(today|tomorrow|this week|next week)", tl)
    if m:
        rng = next((r for r in ("today", "tomorrow", "next week", "this week") if r in tl), "this week")
        return pick("calendar", {"action": "list", "range": rng})
    # notes / to-do
    m = re.search(r"\b(?:make a note|note down|write down|take a note|jot (?:this |that |it )?down|add to my (?:to-?do|notes?|list))\b(?: that| to)?[:,]?\s*(.+)", t, re.I)
    if m and len(m.group(1).split()) >= 2:
        return pick("notes", {"action": "add", "text": m.group(1).strip().rstrip(".")})
    if re.search(r"\b(read|what are|show|list)\b.*\bmy (notes|to-?do|list)\b", tl):
        return pick("notes", {"action": "list"})
    # time / date
    m = re.search(r"\bwhat(?:'s| is)? the time\b(?: (?:now )?in ([a-z ]+?))?[?.]?$|\bwhat time is it\b(?: (?:now )?in ([a-z ]+?))?[?.]?$|"
                  r"\btime in ([a-z ]+?) (?:right )?now\b|\bwhat(?:'s| is) (?:the )?date\b|\bwhat day is (?:it|today)\b|\btoday'?s date\b", tl)
    if m:
        place = next((g for g in m.groups() if g), "") if m.groups() else ""
        return pick("get_datetime", {"place": (place or "").strip()})
    # weather
    if re.search(r"\b(weather|temperature outside|going to rain|will it rain|is it raining|forecast|hot outside|cold outside|umbrella)\b", tl):
        cm = re.search(r"\b(?:in|at|for) " + _CITY + r"\b", t)
        city = cm.group(1) if cm and cm.group(1).lower() not in ("the", "my", "today", "tomorrow") else ""
        return pick("get_weather", {"city": city, "day": "tomorrow" if "tomorrow" in tl else "today"})
    # currency
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)?\s*(dollars?|usd|euros?|eur|pounds?|gbp|yen|jpy|rupees?|inr|sgd|aud|cad|yuan|cny|chf|"
                  r"singapore dollars?|australian dollars?|canadian dollars?|swiss francs?)\s+(?:to|in|into)\s+(rupees?|inr|dollars?|usd|euros?|eur|"
                  r"pounds?|gbp|yen|jpy|sgd|aud|cad|yuan|cny|chf)\b", tl)
    if m and not re.search(r"\b(km|kg|miles?|pounds of)\b", tl):
        return pick("currency_convert", {"amount": float(m.group(1).replace(",", "")) if m.group(1) else 1.0,
                                         "from_currency": m.group(2).rstrip("s") if m.group(2) not in ("usd", "eur", "gbp", "jpy", "inr", "sgd", "aud", "cad", "cny", "chf") else m.group(2),
                                         "to_currency": m.group(3).rstrip("s") if m.group(3) not in ("usd", "eur", "gbp", "jpy", "inr", "sgd", "aud", "cad", "cny", "chf") else m.group(3)})
    if re.search(r"\b(exchange rate|dollar rate)\b", tl):
        return pick("currency_convert", {"amount": 1.0, "from_currency": "usd",
                                         "to_currency": HOME_CURRENCY.lower() if HOME_CURRENCY != "USD" else "eur"})
    # unit conversion
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(degrees? )?([a-z/]+(?: [a-z]+)?)\s+(?:to|in|into)\s+(degrees? )?([a-z/]+)\b", tl)
    if m and (_unit(m.group(3)) or m.group(3) in ("celsius", "fahrenheit", "kelvin", "c", "f")) and \
            (_unit(m.group(5)) or m.group(5) in ("celsius", "fahrenheit", "kelvin", "c", "f")):
        return pick("convert_units", {"value": float(m.group(1)), "from_unit": m.group(3), "to_unit": m.group(5)})
    # calculator
    if re.search(r"\d", tl) and (re.search(r"\b(percent|%)\s*of\b", tl) or re.search(r"\d\s*(?:[+*/^x×÷]|plus|minus|times|multiplied by|divided by|into)\s*\d", tl)
                                 or re.search(r"\b(square root|squared|cubed)\b", tl)) and \
            re.search(r"\b(what|calculate|how much|compute|is)\b|^[\d(]", tl):
        expr = re.sub(r"^.*?\b(?:what(?:'s| is)|calculate|compute|how much is)\b", "", t, flags=re.I).strip(" ?.")
        return pick("calculate", {"expression": expr.replace(" x ", " * ")})
    # forget
    m = re.search(r"\b(?:forget|delete|remove) (?:that |about )?(?:my |the fact that )?(.+?)(?: from (?:your )?memory)?[.?!]?$", t, re.I)
    if m and re.search(r"\b(forget|from (your )?memory)\b", tl) and len(m.group(1).split()) >= 2:
        return pick("forget_memory", {"fact": m.group(1).strip()})
    # device
    for rx, act in ((r"\b(speak|talk) (more )?slow(er|ly)\b|\bslow down\b", "slower"), (r"\b(speak|talk) (more )?fast(er)?\b|\bspeed up\b", "faster"),
                    (r"\bnormal speed\b", "normal_speed"), (r"\b(louder|volume up|turn (it )?up)\b", "louder"),
                    (r"\b(quieter|softer|volume down|turn (it )?down)\b", "quieter"), (r"\bmute (the )?(mic|microphone|yourself)\b", "mute")):
        if re.search(rx, tl):
            return pick("device_control", {"action": act})
    m = re.search(r"\b(?:switch|change|use) (?:to |your voice to )?(?:the )?(lj ?speech|us voice|american|sef|southern english|british|matcha)(?: voice)?\b", tl)
    if m:
        v = re.sub(r" voice$", "", m.group(1))
        return pick("device_control", {"action": "voice", "voice": VOICE_ALIASES.get(v, v)})
    m = re.search(r"\b(?:switch|change) (?:to |your voice to )(?:the |a )?([a-z][a-z ]{0,20}?) voice\b", tl)
    if m:  # an unknown voice name: the tool answers with the voices that exist
        return pick("device_control", {"action": "voice", "voice": m.group(1)})
    # explicit library search
    m = re.search(r"\b(?:search|look (?:up|in)|check) my (?:papers|books|notes|files|library|documents)\b (?:for |about )?(.+)", t, re.I)
    if m:
        return pick("search_library", {"query": m.group(1).strip(" ?.")})
    # online look-ups
    m = re.search(r"\b(?:news|headlines)\b(?: (?:about|on) (.+?))?[?.]?$", tl)
    if m and re.search(r"\b(latest|today'?s|what'?s the|any|top|tell me the)\b", tl):
        return pick("get_news", {"topic": (m.group(1) or "").strip()})
    m = re.search(r"\b(?:search (?:the web|online|the internet|google) for|look (?:it )?up online|google)\s+(.+)", t, re.I)
    if m:
        return pick("web_search", {"query": m.group(1).strip(" ?.")})
    # recent events are never in the person's documents or the model's training data
    if re.search(r"\b(who won|final score|scores?\b.*\b(match|game)|results? of (the )?(match|election|game)|"
                 r"latest (?!on my|in my)|current (price|prime minister|president|ceo|champion)|right now in the world|"
                 r"(yesterday|last night|this morning)'?s? (match|game|news|result))", tl) and not re.search(r"\b(my|our)\b", tl):
        return pick("web_search", {"query": t.strip(" ?.")})
    m = re.search(r"\b(?:wikipedia|wiki)\b(?: (?:about|on|for))? (.+)", t, re.I) or \
        re.search(r"\b[Ww]ho (?:is|was) ([A-Z][\w.]+(?: [A-Z][\w.]+){1,3})\b", t)  # a named person, not "who is my best friend"
    if m:
        topic = (m.group(1) or "").strip(" ?.")
        if topic:
            return pick("wikipedia", {"topic": topic})
    return None
