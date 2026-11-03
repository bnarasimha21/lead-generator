"""Deterministic post-extraction verification (issue 2).

The model's structured output is not trusted on its own: every field with a
non-null value must carry a quote, and that quote must actually appear in
the transcript. Email and phone additionally get a syntax check. Anything
that fails is nulled out and listed in `rejected` so the UI can say
"could not confirm" instead of silently keeping a hallucinated value.
"""

import re
from typing import List, Tuple

from app.schema import FIELD_NAMES, Lead

_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def quote_in_transcript(quote: str, transcript: str) -> bool:
    if not quote:
        return False
    return _normalise(quote) in _normalise(transcript)


# --- Spoken email normalisation -------------------------------------------

_SPOKEN_EMAIL_WORDS = {
    " at ": "@",
    " dot ": ".",
    " underscore ": "_",
    " dash ": "-",
    " hyphen ": "-",
    " plus ": "+",
}


def normalise_spoken_email(raw: str) -> str:
    """Turn a spoken-out email ("john at example dot com") into
    "john@example.com". Returns the input unchanged if it already looks
    like an email, and returns "" if it cannot be confidently normalised.
    """
    if not raw:
        return ""
    candidate = raw.strip()
    if EMAIL_RE.match(candidate):
        return candidate
    lowered = " " + _normalise(candidate) + " "
    for spoken, symbol in _SPOKEN_EMAIL_WORDS.items():
        lowered = lowered.replace(spoken, symbol)
    lowered = lowered.strip().replace(" ", "")
    if EMAIL_RE.match(lowered):
        return lowered
    return ""


EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PHONE_RE = re.compile(r"^\+?[0-9][0-9 ().-]{6,18}[0-9]$")


def _check_email(value: str) -> bool:
    normalised = normalise_spoken_email(value)
    return bool(normalised) and bool(EMAIL_RE.match(normalised))


def _check_phone(value: str) -> bool:
    return bool(PHONE_RE.match(value.strip()))


def verify_lead(lead: Lead, transcript: str) -> Tuple[Lead, List[str]]:
    """Return a (possibly modified) Lead with every unverifiable field
    nulled out, and the list of field names that were rejected.
    """
    rejected: List[str] = []
    data = lead.model_dump()

    for name in FIELD_NAMES:
        field = data[name]
        value = field.get("value")
        evidence = field.get("evidence")
        if value is None:
            continue
        if not evidence or not quote_in_transcript(evidence, transcript):
            rejected.append(name)
            field["value"] = None
            field["evidence"] = None
            continue
        if name == "email" and not _check_email(value):
            rejected.append(name)
            field["value"] = None
            field["evidence"] = None
            continue
        if name == "phone" and not _check_phone(value):
            rejected.append(name)
            field["value"] = None
            field["evidence"] = None
            continue

    return Lead.model_validate(data), rejected
