"""The assistant's identity and system prompt: one generic assistant, "Nova".

The pipeline keeps a per-profile structure (separate memory, library and conversation files per
profile id) so more profiles can be added here, but only the generic "nova" profile ships, with no
personal facts. Facts are learned from the conversation and stored locally in data/.
"""
from __future__ import annotations

import os
import re

BASE_RULES = (
    "You are Nova, a warm voice assistant running privately on this computer. This is a spoken "
    "conversation: answer in one to three short sentences, start with a short sentence, no markdown, "
    "no lists, no emojis. Sometimes you are given things you remember about this person; use them "
    "naturally and never read the list back. Sometimes you are given notes from their {library}; then "
    "answer only from those notes and mention the source once by its short title. Never pretend to "
    "look something up, and never invent facts, numbers or page numbers."
)

PERSONAS: dict[str, dict] = {
    "nova": {
        "name": "", "title": "Nova", "library": "documents",
        "style": " Be friendly, clear and brief.",
        "facts": [],
    },
}
ORDER = ["nova"]


def is_child(profile: str | None) -> bool:
    """Child-safety guard on: a child profile, or anyone using the Kids template."""
    if PERSONAS.get(profile or "", {}).get("child"):
        return True
    try:
        import templates
        return templates.child_safe(templates.current(profile))
    except Exception:  # noqa: BLE001
        return False


def greeting(profile: str | None, name: str = "") -> str:
    """Opening line; `name` is the user's name when memory knows it."""
    p = PERSONAS.get(profile or "") or PERSONAS[ORDER[0]]
    if p.get("greeting"):
        return p["greeting"]
    name = name or p.get("name", "")
    return f"Hi {name}, it's good to hear from you." if name else "Hi, it's good to hear from you."


_LEADIN_RX = re.compile(r"\s*Example lead-ins?:\s*(.*)$", re.S)


def _split_style(style: str) -> tuple[str, list[str]]:
    """Profile style without its example lead-ins (they made the model 'pretend to check' in ordinary
    turns), plus the lead-in examples for the lookup turn."""
    m = _LEADIN_RX.search(style or "")
    if not m:
        return style, []
    return style[: m.start()].rstrip(), [x for x in re.findall(r"'(.+?)'(?=\s|$)", m.group(1)) if 4 <= len(x) <= 80]


def system_prompt(profile: str | None) -> str:
    p = PERSONAS.get(profile or "") or PERSONAS[ORDER[0]]
    return (BASE_RULES + " " + _split_style(p["style"])[0].strip()).format(name=p["name"], library=p["library"])


TOOL_RULE = ("You can use tools. When a request needs one (time, timers, reminders, calendar, notes, maths, units, "
             "weather, web search, news, Wikipedia, currency, settings), call the tool instead of guessing. Never say you set, "
             "added, saved, checked or looked something up unless a tool result in this turn says so, and never promise "
             "to check or search later: either call the tool now or answer.")
_runtime_cache: dict[str, str] = {}


def runtime_system(profile: str | None) -> str:
    """The system turn the model actually gets: profile prompt + tool rule + Gemma 4 tool declarations.
    TOOLS=0 turns tools off."""
    p = profile if profile in PERSONAS else ORDER[0]
    import templates
    t = templates.current(p)
    base = system_prompt(p) + (" " + templates.prompt(t) if templates.prompt(t) else "")
    if os.environ.get("TOOLS", "1") == "0":
        return base
    key = f"{p}|{t}"
    if key not in _runtime_cache:
        import tools
        # Shared prefix first (identical for every profile), then the profile part: switching profile only
        # re-reads the profile part (llm_cpu_server.LiveGemma._new_session reuses the prefix checkpoint).
        # Every profile declares the same tools so the prefix is shared; a child template's limits are
        # enforced when a tool runs (tools.run / tools.allowed), never by hiding declarations.
        _runtime_cache[key] = (TOOL_RULE + "\n\n" + tools.declarations(list(tools.TOOLS))
                               + "\n\nAbout this person:\n" + base)
    return _runtime_cache[key]


def leadin_examples(profile: str | None) -> list[str]:
    p = PERSONAS.get(profile or "") or PERSONAS[ORDER[0]]
    return [x.format(name=p["name"]) for x in _split_style(p["style"])[1]] or [f"One moment, checking your {p['library']}."]


# Per-profile vocabulary fixes: {profile: {regex: replacement}} for words speech recognition reliably
# mishears for that profile's topics. Empty by default.
ALIASES: dict[str, dict[str, str]] = {}


def apply_aliases(profile: str | None, text: str) -> str:
    for rx, rep in ALIASES.get(profile or "", {}).items():
        text = re.sub(rx, rep, text, flags=re.I)
    return text
