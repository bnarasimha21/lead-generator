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

# Verification-review finding 3: a prior fix only checked that the
# evidence span's *tokens* formed a superset of the value's tokens
# (an unordered bag), with no bound on how long the evidence span itself
# was allowed to be. That let an attacker pass the entire transcript as
# "evidence" for a field, and any value whose words happen to appear
# *anywhere* in that transcript (including spliced together from unrelated
# sentences) would be accepted, since token-set membership does not care
# where in the span a token came from or in what order.
#
# EVIDENCE_MAX_CHARS bounds the evidence span to roughly a sentence or two
# (every legitimate case in this codebase's tests and golden set is well
# under 100 characters), ruling out "the whole transcript" as evidence.
# _ordered_subsequence then requires the value's tokens to actually occur
# in that bounded span *in their own relative order*, which rules out
# "assembled from scattered, unrelated parts of the span" even when the
# span itself is short enough to pass the length check.
EVIDENCE_MAX_CHARS = 200


def _normalise(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def _token_list(text: str) -> list:
    return _TOKEN_RE.findall(_normalise(text))


def _ordered_subsequence_span(value_tokens: list, evidence_tokens: list):
    """Greedy leftmost in-order match of value_tokens inside
    evidence_tokens. Returns (start_index, end_index) of the matched span
    in evidence_tokens, or None if value_tokens does not occur as an
    in-order subsequence at all.
    """
    idx = 0
    start = None
    end = None
    for i, ev in enumerate(evidence_tokens):
        if idx < len(value_tokens) and ev == value_tokens[idx]:
            if start is None:
                start = i
            end = i
            idx += 1
    if idx == len(value_tokens):
        return start, end
    return None


def _ordered_and_proximate(value_tokens: list, evidence_tokens: list) -> bool:
    """True if value_tokens occur, in order, within evidence_tokens, AND
    the matched span is not far wider than value_tokens itself.

    Order alone (an in-order subsequence with unlimited gaps) is not
    enough: two unrelated phone numbers quoted a sentence apart in the
    same (short, within-cap) evidence span can still each contribute some
    of their digits in the right relative order purely by coincidence
    (e.g. "555-123-4567 ... 555-999-0000" lets a spliced "555-123-0000"
    match in order even though it was never actually said). Bounding the
    match's span close to the value's own token count catches that: a
    handful of filler/spoken words between the value's tokens (e.g. "at"
    / "dot" in a spoken email) is allowed, but a match that needs far more
    intervening tokens than the value itself has is a sign the value was
    assembled from scattered, unrelated fragments rather than said
    together.
    """
    span = _ordered_subsequence_span(value_tokens, evidence_tokens)
    if span is None:
        return False
    start, end = span
    window = end - start + 1
    max_window = max(len(value_tokens) * 3, len(value_tokens) + 4)
    return window <= max_window


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
    contiguous quote); falling back to an ordered, proximity-bound token
    check catches values normalised away from their spoken form (e.g.
    "john@example.com" against evidence "john at example dot com").

    The evidence span itself is capped to EVIDENCE_MAX_CHARS, and the
    fallback check requires the value's tokens to occur in the evidence in
    their own relative order and close together, not merely as an
    unordered set anywhere in the span (finding 3): otherwise an attacker
    could pass the entire transcript as "evidence" and have any value
    whose words appear anywhere in it, in any order, accepted - including
    a value spliced together from unrelated parts of the transcript.
    """
    if not value or not evidence:
        return False
    if len(evidence) > EVIDENCE_MAX_CHARS:
        return False
    value_norm = _normalise(value)
    evidence_norm = _normalise(evidence)
    if value_norm and value_norm in evidence_norm:
        return True
    value_tokens = _token_list(value)
    if not value_tokens:
        return False
    evidence_tokens = _token_list(evidence)
    return _ordered_and_proximate(value_tokens, evidence_tokens)


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
