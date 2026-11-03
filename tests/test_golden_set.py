"""Issue 4 acceptance criteria: the golden set has at least 25 text cases,
and every expected evidence string actually exists in its transcript.
"""

import re

from evals.run import load_golden_cases

_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text):
    return _WHITESPACE_RE.sub(" ", text).strip().casefold()


def test_at_least_25_text_cases():
    cases = load_golden_cases()
    assert len(cases) >= 25


def test_every_evidence_string_exists_in_its_transcript():
    cases = load_golden_cases()
    for case in cases:
        transcript = _normalise(case["transcript"])
        for field_name, field in case["expected"].items():
            evidence = field.get("evidence")
            if evidence is None:
                assert field.get("value") is None, f"{case['id']}.{field_name}: value without evidence"
                continue
            assert _normalise(evidence) in transcript, f"{case['id']}.{field_name}: evidence not found in transcript"


def test_case_ids_are_unique():
    cases = load_golden_cases()
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
