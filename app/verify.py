"""Deterministic post-extraction verification (issue 2).

The model's structured output is not trusted on its own: every field with a
non-null value must carry a quote, that quote must actually appear in the
transcript, AND the value itself must be derivable from that quote (not just
from the transcript at large). Without the second check, a record could
claim evidence="my email is" (a real substring of the transcript) while
asserting value="attacker@evil.example" (never actually said anywhere near
that quote, or at all): the evidence check alone would pass it. Email and
phone additionally get a syntax check. Anything that fails is nulled out and
listed in `rejected` so the UI can say "could not confirm" instead of
silently keeping a hallucinated value.
"""

import re
from typing import List, Tuple

from app.schema import FIELD_NAMES, Lead

_WHITESPACE_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalise(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(_normalise(text)))


def quote_in_transcript(quote: str, transcript: str) -> bool:
    if not quote:
        return False
    return _normalise(quote) in _normalise(transcript)


def value_supported_by_evidence(value: str, evidence: str) -> bool:
    """A value is only as good as its own evidence quote: require the
    value's meaningful tokens to actually appear inside the evidence span
    itself, not merely somewhere else in the transcript. This is what
    catches "evidence is a real substring of the transcript, but the value
    is not actually said anywhere in that span" (e.g. evidence="my email
    is", value="attacker@evil.example").

    Exact-substring match is tried first (handles spacing/punctuation
    differences inside a single phrase, e.g. a value that is itself a
    contiguous quote); falling back to a token-subset check catches values
    normalised away from their spoken form (e.g. "john@example.com" against
    evidence "john at example dot com").
    """
    if not value or not evidence:
        return False
    value_norm = _normalise(value)
    evidence_norm = _normalise(evidence)
    if value_norm and value_norm in evidence_norm:
        return True
    value_tokens = _tokens(value)
    if not value_tokens:
        return False
    return value_tokens <= _tokens(evidence)


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
        if not value_supported_by_evidence(value, evidence):
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
