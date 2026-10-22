"""Use-case templates: ready-made system-prompt presets a person picks for how Nova should help them.

Each template adds a short instruction block after the profile's own style (personas.runtime_system).
The choice is saved per profile in data/<profile>_settings.json; the default is "everyday".
The "kids" template also turns on the child-safety guard (safety.py).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

DIR = Path(os.environ.get("MEMORY_DIR", Path(__file__).resolve().parent / "data"))

TEMPLATES: dict[str, dict] = {
    "study": {
        "name": "Study from my books",
        "blurb": "Answers from your own textbooks with the page, then asks you a quick recall question.",
        "prompt": ("Template: study partner. Answer from their own books and notes with the page, explain simply, "
                   "then sometimes ask one short recall question to check they remember. Help with oral-exam "
                   "practice when asked. Be calm and encouraging."),
    },
    "kids": {
        "name": "Kids: homework and stories",
        "blurb": "Simple words, hints instead of answers, gentle stories, child-safety guard on.",
        "prompt": ("Template: child companion. Use very simple words and short, cheerful sentences. For homework, "
                   "give a hint and let them try before giving the answer. Tell gentle stories with happy endings. "
                   "Never ask for personal details. For anything unsafe or grown-up, say kindly to ask a grown-up."),
        "child_safe": True,
    },
    "care": {
        "name": "Care for a parent",
        "blurb": "Speaks slowly, reminds about medicines and appointments from their own notes, happy to chat.",
        "prompt": ("Template: companion for an older person. Speak slowly and clearly in short, warm sentences. Help "
                   "them remember medicine times and doctor visits from their own notes and reminders. Never give "
                   "medical advice beyond what their own doctor's notes say; for anything about health, suggest "
                   "calling their doctor or family. Be patient, and happy to simply chat."),
        "speed": 0.9,
    },
    "shop": {
        "name": "Shop and orders",
        "blurb": "Numbers first from your order sheets, order IDs read clearly, one next step.",
        "prompt": ("Template: small-business assistant. Give the number first, from their own order and stock "
                   "sheets, read order IDs clearly, and suggest one practical next step."),
    },
    "research": {
        "name": "Research assistant",
        "blurb": "Precise answers from your papers and drafts; unpublished work never leaves this device.",
        "prompt": ("Template: research assistant. Be precise and technical, answer from their papers and drafts with "
                   "the source, say when evidence is early or weak, and never guess numbers. Their drafts and data "
                   "stay on this device."),
    },
    "everyday": {
        "name": "Everyday assistant",
        "blurb": "Friendly and brief: plans, timers, reminders, maths and quick questions.",
        "prompt": ("Template: everyday assistant. Be friendly and brief. Help with plans, timers, reminders, notes, "
                   "maths and quick questions."),
    },
}
DEFAULTS: dict[str, str] = {}  # profile -> default template id
_lock = threading.Lock()


def _file(profile: str) -> Path:
    return DIR / f"{profile}_settings.json"


def current(profile: str | None) -> str:
    p = profile or ""
    try:
        t = json.loads(_file(p).read_text()).get("template")
        if t in TEMPLATES:
            return t
    except (OSError, ValueError):
        pass
    return DEFAULTS.get(p, "everyday")


def choose(profile: str, template: str) -> str:
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}")
    with _lock:
        DIR.mkdir(parents=True, exist_ok=True)
        f = _file(profile)
        try:
            d = json.loads(f.read_text())
        except (OSError, ValueError):
            d = {}
        d["template"] = template
        f.write_text(json.dumps(d))
    return template


def prompt(template: str | None) -> str:
    return TEMPLATES.get(template or "", {}).get("prompt", "")


def child_safe(template: str | None) -> bool:
    return bool(TEMPLATES.get(template or "", {}).get("child_safe"))


def speed(template: str | None) -> float:
    return float(TEMPLATES.get(template or "", {}).get("speed", 1.0))


def listing() -> list[dict]:
    return [{"id": k, "name": v["name"], "blurb": v["blurb"], "child_safe": bool(v.get("child_safe"))} for k, v in TEMPLATES.items()]
