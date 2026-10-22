"""Unit tests for tools.py (offline parts only: no network calls)."""
import datetime as dt
import time

import pytest

import tools as T


@pytest.fixture(autouse=True)
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "DIR", tmp_path)
    T._dbs.clear()
    yield
    T._dbs.clear()


@pytest.mark.parametrize("text,name,key,val", [
    ("Set a timer for 5 minutes", "timer", "minutes", 5.0),
    ("Set a 10 minute timer for the pasta", "timer", "label", "pasta"),
    ("Start a timer for 30 seconds", "timer", "seconds", 30.0),
    ("Remind me to call Chris tomorrow at 5 pm", "reminders", "text", "call Chris"),
    ("What's on my calendar this week?", "calendar", "range", "this week"),
    ("Make a note that the supplier wants payment by Monday", "notes", "action", "add"),
    ("What time is it in London?", "get_datetime", "place", "london"),
    ("What's the weather in Paris?", "get_weather", "city", "Paris"),
    ("Will it rain tomorrow?", "get_weather", "day", "tomorrow"),
    ("Convert 100 dollars to euros", "currency_convert", "amount", 100.0),
    ("Convert 10 km to miles", "convert_units", "to_unit", "miles"),
    ("What's 18 percent of 2499?", "calculate", "expression", "18 percent of 2499"),
    ("Speak slower please", "device_control", "action", "slower"),
    ("Search my notes for tax receipts", "search_library", "query", "tax receipts"),
    ("What's the latest news on AI?", "get_news", "topic", "ai"),
    ("Who is Sundar Pichai?", "wikipedia", "topic", "Sundar Pichai"),
])
def test_route(text, name, key, val):
    r = T.route("nova", text)
    assert r and r[0] == name and r[1].get(key) == val, r


@pytest.mark.parametrize("text", ["What's the name of my cat?", "Which nerve supplies the deltoid?", "Who is my best friend?",
                                  "What happened with the return on order 1105?", "Tell me a story about a spider", "I'm scared of the thunder"])
def test_no_tool_for_normal_questions(text):
    assert T.route("nova", text) is None


def test_child_gets_safe_subset_only():
    allowed = set(T.allowed("nova", child=True))
    assert {"web_search", "get_news", "wikipedia", "currency_convert", "forget_memory"}.isdisjoint(allowed)
    assert {"timer", "calculate", "get_weather", "reminders"} <= allowed
    assert T.route("nova", "Search the web for dinosaurs", child=True) is None
    assert T.run("web_search", {"query": "x"}, {"profile": "nova", "child": True})["ok"] is False


def test_parse_native_call():
    assert T.parse_call('<|tool_call>call:get_weather{city:<|"|>Oslo<|"|>,day:<|"|>tomorrow<|"|>}<tool_call|>') == \
        ("get_weather", {"city": "Oslo", "day": "tomorrow"})
    assert T.parse_call('<|tool_call>call:timer{action:<|"|>set<|"|>,minutes:3}<tool_call|>') == ("timer", {"action": "set", "minutes": 3.0})
    assert T.parse_call("<|tool_call>call:rm_rf{}<tool_call|>") is None


def test_declarations_format():
    d = T.declarations(["get_weather"])
    assert d.startswith("<|tool>declaration:get_weather{description:") and d.endswith("<tool|>") and 'type:<|"|>OBJECT<|"|>' in d


@pytest.mark.parametrize("expr,val", [("18 percent of 2499", 449.82), ("12 times 7 plus 3", 87), ("the square root of 144", 12),
                                      ("2 to the power of 10", 1024), ("100 divided by 8", 12.5)])
def test_calculate(expr, val):
    assert T.run("calculate", {"expression": expr}, {"profile": "nova"})["result"] == val


def test_calculate_refuses_code():
    assert T.run("calculate", {"expression": "__import__('os').system('ls')"}, {"profile": "nova"})["ok"] is False


def test_convert_units():
    c = lambda v, f, t: T.run("convert_units", {"value": v, "from_unit": f, "to_unit": t}, {"profile": "nova"})  # noqa: E731
    assert c(37, "celsius", "fahrenheit")["result"] == 98.6
    assert c(10, "km", "miles")["result"] == pytest.approx(6.2137, abs=1e-3)
    assert c(1, "kg", "pounds")["result"] == pytest.approx(2.2046, abs=1e-3)
    assert c(5, "km", "kg")["ok"] is False


def test_parse_when():
    base = dt.datetime(2026, 10, 12, 14, 0, tzinfo=T.LOCAL_TZ)  # a Monday
    assert T.parse_when("tomorrow at 5 pm", base)[0] == dt.datetime(2026, 10, 13, 17, 0, tzinfo=T.LOCAL_TZ)
    assert T.parse_when("on Friday", base)[0].date() == dt.date(2026, 10, 16)
    assert T.parse_when("in 20 minutes", base)[0] == base + dt.timedelta(minutes=20)
    assert T.parse_when("on 3 May at 9", base)[0] == dt.datetime(2027, 5, 3, 9, 0, tzinfo=T.LOCAL_TZ)
    assert T.parse_when("no time here", base)[0] is None


def test_reminders_calendar_notes_roundtrip():
    ctx = {"profile": "nova"}
    assert T.run("reminders", {"action": "add", "text": "call Chris", "when": "tomorrow at 5 pm"}, ctx)["ok"]
    assert T.run("reminders", {"action": "list"}, ctx)["reminders"][0]["text"] == "call Chris"
    assert T.run("calendar", {"action": "add", "title": "dentist", "when": "tomorrow at 4 pm"}, ctx)["ok"]
    items = T.run("calendar", {"action": "list", "range": "tomorrow"}, ctx)["items"]
    assert any(i["what"] == "dentist" for i in items) and any("call Chris" in i["what"] for i in items)
    assert T.run("notes", {"action": "add", "text": "buy printer ink"}, ctx)["total"] == 1
    assert T.run("notes", {"action": "list"}, ctx)["notes"] == ["buy printer ink"]
    assert T.run("reminders", {"action": "list"}, {"profile": "other"})["reminders"] == []  # per profile


def test_timer_fires_once():
    ctx = {"profile": "nova"}
    r = T.run("timer", {"action": "set", "seconds": 1, "label": "tea"}, ctx)
    assert r["ok"] and r["length"] == "1 second"
    time.sleep(1.1)
    due = T.due_timers("nova")
    assert len(due) == 1 and due[0]["label"] == "tea"
    assert T.due_timers("nova") == []


def test_online_tool_offline_message(monkeypatch):
    monkeypatch.setattr(T, "online", lambda: False)
    r = T.run("get_weather", {"city": "Paris"}, {"profile": "nova"})
    assert r["ok"] is False and r["error"] == "offline"
    assert "offline" in T.note("get_weather", r)


def test_spoken_numbers_and_tomorrow_spelling():
    assert T.route("nova", "What is seven times eight?") == ("calculate", {"expression": "7 times 8"})
    assert T.route("nova", "Will it rain to-morrow?")[1]["day"] == "tomorrow"
    assert T.words_to_digits("three hundred divided by twelve") == "300 divided by 12"



def test_model_calls_need_a_request():
    assert not T.model_call_ok("notes", "My cat's name is Whiskers.")
    assert not T.model_call_ok("web_search", "What is a black hole in simple words?")
    assert T.model_call_ok("web_search", "Who won the cricket match yesterday?")
    assert T.model_call_ok("timer", "Put a 3 minute countdown on")
    assert T.model_call_ok("get_datetime", "anything")  # read-only, offline
    assert T.route("nova", "Jot this down: buy printer ink") == ("notes", {"action": "add", "text": "buy printer ink"})


def test_voice_switch_only_to_installed_voices(monkeypatch):
    monkeypatch.setattr(T, "voices_available", lambda: ["ljspeech", "matcha"])
    ctx = {"profile": "nova"}
    assert T.route("nova", "Switch to the matcha voice") == ("device_control", {"action": "voice", "voice": "matcha"})
    assert T.route("nova", "Change your voice to the LJ Speech voice")[1]["voice"] == "ljspeech"
    assert T.run("device_control", {"action": "voice", "voice": "matcha"}, ctx)["voice"] == "matcha"
    assert T.run("device_control", {"action": "voice", "voice": "lj speech"}, ctx)["voice"] == "ljspeech"
    r = T.run("device_control", {"action": "voice", "voice": "sef"}, ctx)  # known voice, not installed
    assert r["ok"] is False and r["voices"] == ["ljspeech", "matcha"] and "isn't installed" in r["error"]
    r = T.run("device_control", {"action": "voice", "voice": "robot"}, ctx)
    assert r["ok"] is False and "don't have a voice called robot" in r["error"]
    assert T.route("nova", "Switch to the robot voice") == ("device_control", {"action": "voice", "voice": "robot"})
    assert T.run("device_control", {"action": "voice", "voice": "matcher"}, ctx)["voice"] == "matcha"  # STT near-miss
