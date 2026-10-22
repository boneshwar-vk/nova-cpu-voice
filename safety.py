"""Child-safety guard (used when the "kids" template is selected) and a health red-flag guard for everyone.

Gemma 4 E2B follows a system prompt only loosely, so these keyword rules are the real guard:
- check_input(text): runs on what the child said BEFORE the language model; a hit skips the model
  and speaks a fixed safe reply (no [LOOKUP] lead-in, no memory write);
- check_output(sentence): runs on each sentence before it is spoken; a hit replaces the sentence;
- blocks_memory(text): personal data (address, phone, school name, PIN-like numbers) is never saved.
Deliberately conservative; false positives fall back to a gentle redirect.
"""
from __future__ import annotations

import re

_R = lambda *ps: re.compile("|".join(ps), re.I)  # noqa: E731

DISTRESS = _R(r"\b(hurt|hit|touch(ed)?)\s+me\b", r"\bkeep (it|this) (a )?secret\b", r"\b(i'?m|i am) lost\b",
              r"\b(want(s)? to die|kill (my|him|her)self|hurt (my|him|her)self)\b", r"\bcome with (him|her|me)\b.*\b(puppies|sweets|chocolate)\b",
              r"\b(uncle|stranger|man|aunty)\b.*\b(come with|follow|car)\b", r"\b(took|ate|drank)\b.*\bmedicine\b", r"\bbleeding\b")
PERSONAL = _R(r"\bmy (home )?address\b", r"\bi live (at|in flat|on)\b", r"\bwhat'?s your address\b",
              r"\bmy (house|home|flat) is (in|at|near|on)\b", r"\bi stay (in|at|near)\b", r"\bwhere do you live\b", r"\bflat\s*(no\.?\s*)?\d", r"\bphone number\b",
              r"\b\d{10}\b", r"\b\d{6}\b", r"\bmy school('s)? name\b", r"\bpassword\b", r"\botp\b", r"\bwhat is your address\b")
DANGER = _R(r"\b(light|turn on|switch on)\b.*\b(gas|stove|fire|matches|candle)\b", r"\bmatches\b", r"\bknife\b",
            r"\b(socket|plug)\b.*\b(finger|touch|wire)\b", r"\bmedicine(s)?\b.*\b(how many|more|take)\b",
            r"\bgo out alone\b", r"\bopen the door\b.*\bstranger", r"\bdownload\b.*\b(app|game|free fire|pubg)\b")
MATURE = _R(r"\b(kiss|sex|babies are made|naked|boyfriend|girlfriend)\b", r"\b(drugs?|alcohol|beer|cigarette|smok(e|ing))\b",
            r"\b(gun|bomb|murder|blood|kill(ing|ed)?|dead body|ghost.*die|die.*ghost|horror)\b",
            r"\bignore your rules\b", r"\bpretend (you are|to be) a grown[- ]?up\b", r"\b(swear|bad|rude|naughty|dirty|gaali|cuss) words?\b", r"\bgaali\b")
SECRET = _R(r"\bsecret friend\b", r"\bdon'?t tell (mummy|mommy|papa|mom|dad|my parents)\b", r"\bnot tell (mummy|mommy|papa)\b")

REPLIES = {
    "distress": "Thank you for telling me. You did the right thing. Please tell your mum, dad or teacher right now. "
                "A grown-up can also call a children's helpline.",
    "personal": "That's private, so keep it safe with your mum and dad. Shall we read a story instead?",
    "danger": "That could hurt you, so it's only for grown-ups. Please ask your mum or dad to help you.",
    "mature": "That's a good question for your mum, dad or teacher. Shall we read a fun story instead?",
    "secret": "I'm Nova, a computer helper. It's always okay to tell your mum and dad what we talk about.",
}
OUTPUT_REPLACEMENT = "Let's ask a grown-up about that."


def check_input(text: str) -> str | None:
    """Category of an unsafe child input (most serious first), else None."""
    for cat, rx in (("distress", DISTRESS), ("personal", PERSONAL), ("secret", SECRET), ("danger", DANGER), ("mature", MATURE)):
        if rx.search(text or ""):
            return cat
    return None


# Health red flags, for EVERY profile: never answered by the model, always "call someone now".
HEALTH = re.compile(
    r"\b(chest|heart)\b.{0,25}\b(pain|tight|tightness|hurts?|pressure|heavy)\b|\b(pain|tight|tightness|pressure)\b.{0,15}\bchest\b"
    r"|\b(can'?t|cannot|hard to|trouble|difficult(y)? to) breathe?\b|\bbreathless|short(ness)? of breath"
    r"|\b(dizzy|dizziness|faint(ed|ing)?|passed out|black(ed)? out)\b|\bi (fell|have fallen|had a fall)\b"
    r"|\b(numb(ness)?|weak(ness)?) (in|on) (my )?(arm|face|leg|side)\b|\bslurr(ed|ing)\b|\bbleeding (a lot|heavily|won'?t stop)\b"
    r"|\b(took|taken|swallowed) (too many|an overdose|extra) (tablets?|pills?|medicines?)\b"
    r"|\b(want to|going to) (kill|hurt) myself\b|\bsuicid", re.I)


_FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|my|me|mine)\b", re.I)
_STUDY_Q = re.compile(r"^\s*(what|which|explain|define|describe|why|how (does|do|is|are)|list|name the|causes? of)\b", re.I)


def check_health(text: str) -> bool:
    """A red-flag symptom said about oneself ("my chest feels tight", "I feel dizzy"); not a textbook question
    ("what is chest pain in anatomy?") and not furniture ("my chest of drawers")."""
    t = re.sub(r"\bchest of drawers\b", "", text or "", flags=re.I)
    return bool(HEALTH.search(t) and _FIRST_PERSON.search(t) and not _STUDY_Q.search(t))


def health_reply(name: str = "", contacts: str = "") -> str:
    who = contacts or "your doctor or your family"
    you = f", {name}" if name else ""
    return (f"That needs a doctor, not me{you}. Please call {who} right now. "
            "If it feels severe, call your local emergency number.")


def check_output(sentence: str) -> bool:
    """True when a sentence Nova is about to speak must be replaced."""
    s = sentence or ""
    return bool(DANGER.search(s) or MATURE.search(s) or re.search(r"\b\d{10}\b", s))


def blocks_memory(text: str) -> bool:
    t = text or ""
    return bool(PERSONAL.search(t) or re.search(r"\bschool\b.*\b(is|called|named)\b", t, re.I)
                or LOCATION.search(t))


# Where a child lives or goes is never saved, even when it is not a full address.
LOCATION = _R(r"\b(i|we) (live|stay)\b", r"\b(my|our) (house|home|flat|building|society|colony|area)\b", r"\bnear (my|our)\b",
              r"\bi (study|go|went) (at|in|to)\b", r"\bschool\b",
              r"\b(bus|van) (stop|number)\b", r"\bmy (road|street|lane)\b")
