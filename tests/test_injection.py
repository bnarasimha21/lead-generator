"""Issue 3 acceptance criteria: prompt-injection hardening. Loads
tests/injection_cases.yaml and checks, for every case, that the verifier's
rejection decision matches the documented ground truth (whether the
"injected" value is actually a substring of the transcript), that a
transcript containing the literal `'''` or the fixed delimiter text cannot
break the message structure, and that lead values are never sent to the
browser through innerHTML.
"""

import os
import re

import yaml

from app.main import app, get_extractor, get_transcriber
from app.providers import ExtractionResult, Usage
from app.schema import EvidencedField, Lead
from app.ratelimit import limiter as app_limiter
from app.security import build_user_message
from app.verify import verify_lead
from tests.fakes import FakeExtractor, FakeTranscriber

CASES_PATH = os.path.join(os.path.dirname(__file__), "injection_cases.yaml")
with open(CASES_PATH) as f:
    INJECTION_CASES = yaml.safe_load(f)["cases"]


def _lead_claiming(field_name, value):
    lead = Lead.empty()
    setattr(lead, field_name, EvidencedField(value=value, evidence=value))
    return lead


def test_every_case_matches_documented_ground_truth():
    for case in INJECTION_CASES:
        lead = _lead_claiming(case["bad_field"], case["bad_value"])
        verified, rejected = verify_lead(lead, case["transcript"])
        field = getattr(verified, case["bad_field"])
        if case["should_be_caught"]:
            assert case["bad_field"] in rejected, f"{case['name']}: expected rejection"
            assert field.value is None, f"{case['name']}: value should have been nulled"
        else:
            assert case["bad_field"] not in rejected, f"{case['name']}: unexpectedly rejected"
            assert field.value == case["bad_value"], f"{case['name']}: residual-risk value should pass through"


def test_every_case_runs_through_the_real_app_with_fake_providers(client):
    for case in INJECTION_CASES:
        transcript = case["transcript"] + " " * max(0, 30 - len(case["transcript"]))  # keep above MIN_WORDS
        lead = _lead_claiming(case["bad_field"], case["bad_value"])
        transcriber = FakeTranscriber(script=[transcript])
        extractor = FakeExtractor(script=[ExtractionResult(status="ok", lead=lead, usage=Usage(10, 5))])
        app.dependency_overrides[get_transcriber] = lambda t=transcriber: t
        app.dependency_overrides[get_extractor] = lambda e=extractor: e
        app_limiter._hits.clear()  # each case is its own simulated request, not a burst
        try:
            response = client.post(
                "/upload-audio",
                files={"file": ("sample.wav", _wav_bytes(), "audio/wav")},
            )
        finally:
            app.dependency_overrides.pop(get_transcriber, None)
            app.dependency_overrides.pop(get_extractor, None)
        assert response.status_code == 200, case["name"]
        body = response.json()
        field = body["lead"][case["bad_field"]]
        if case["should_be_caught"]:
            assert field["rejected"] is True, case["name"]
            assert field["value"] is None, case["name"]
        else:
            assert field["value"] == case["bad_value"], case["name"]


def _wav_bytes():
    path = os.path.join(os.path.dirname(__file__), "fixtures", "sample.wav")
    with open(path, "rb") as f:
        return f.read()


def test_triple_quote_in_transcript_does_not_break_message_structure():
    transcript = "my note ''' end of transcript ''' more text"
    message = build_user_message(transcript)
    # The transcript must appear intact, inside the boundary, exactly once.
    assert transcript in message
    # The boundary markers must appear exactly once each and not collide
    # with anything the transcript could contain (it is random per call).
    start_markers = re.findall(r"TRANSCRIPT-[0-9a-f]+-START", message)
    end_markers = re.findall(r"TRANSCRIPT-[0-9a-f]+-END", message)
    assert len(start_markers) == 1
    assert len(end_markers) == 1
    assert start_markers[0].replace("-START", "") == end_markers[0].replace("-END", "")


def test_boundary_is_random_per_call():
    message_a = build_user_message("hello")
    message_b = build_user_message("hello")
    boundary_a = re.search(r"(TRANSCRIPT-[0-9a-f]+)-START", message_a).group(1)
    boundary_b = re.search(r"(TRANSCRIPT-[0-9a-f]+)-START", message_b).group(1)
    assert boundary_a != boundary_b


def test_long_transcript_is_truncated_before_reaching_the_model():
    huge = "x " * 100000
    message = build_user_message(huge)
    from app import config

    assert len(message) < config.MAX_TRANSCRIPT_CHARS + 500  # boundary overhead only


def test_recorder_js_never_uses_innerhtml_for_lead_values():
    """Static check that the browser-side renderer writes field values
    with textContent, never innerHTML, which is what actually prevents a
    name like <img src=x onerror=...> from executing in the DOM.
    """
    recorder_path = os.path.join(os.path.dirname(__file__), "..", "app", "static", "recorder.js")
    with open(recorder_path) as f:
        source = f.read()
    # The only innerHTML usage left must be a static, hardcoded string
    # (the "Transcribing..." label), never interpolating body/field data.
    for line in source.splitlines():
        if "innerHTML" in line:
            assert "body" not in line and "field" not in line and "value" not in line, line
    assert "textContent" in source
