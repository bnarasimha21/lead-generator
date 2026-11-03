"""Issue 2 acceptance criteria: strict-schema extraction with evidence
quotes, verified against the transcript. A hallucinated field gets nulled
and listed as rejected; refusal, truncated and invalid replies each
produce a defined response and never a stack trace.
"""

import pytest

from app.main import app, get_extractor, get_transcriber
from app.providers import ExtractionResult, Usage
from app.schema import EvidencedField, Lead
from app.verify import normalise_spoken_email, quote_in_transcript, verify_lead
from tests.fakes import FakeExtractor, FakeTranscriber

TRANSCRIPT = "Hi, I'm Priya from Acme Corp, I'm interested in GPU pricing, let's follow up next week."


def _lead_with(**fields):
    lead = Lead.empty()
    for name, (value, evidence) in fields.items():
        setattr(lead, name, EvidencedField(value=value, evidence=evidence))
    return lead


def test_hallucinated_field_is_nulled_and_rejected():
    lead = _lead_with(company=("Globex Corp", "Globex Corp"))  # not in transcript
    verified, rejected = verify_lead(lead, TRANSCRIPT)
    assert verified.company.value is None
    assert "company" in rejected


def test_verified_field_with_real_quote_passes():
    lead = _lead_with(name=("Priya", "I'm Priya"))
    verified, rejected = verify_lead(lead, TRANSCRIPT)
    assert verified.name.value == "Priya"
    assert "name" not in rejected


@pytest.mark.parametrize(
    "spoken,expected",
    [
        ("john at example dot com", "john@example.com"),
        ("jane underscore doe at example dot org", "jane_doe@example.org"),
        ("priya dash k at acme dot io", "priya-k@acme.io"),
        ("already@valid.com", "already@valid.com"),
    ],
)
def test_spoken_email_forms_normalise(spoken, expected):
    assert normalise_spoken_email(spoken) == expected


def test_unparseable_email_is_rejected_not_guessed():
    assert normalise_spoken_email("call me maybe") == ""
    lead = _lead_with(email=("call me maybe", "call me maybe"))
    verified, rejected = verify_lead(lead, "please call me maybe sometime")
    assert verified.email.value is None
    assert "email" in rejected


@pytest.mark.parametrize("status", ["refusal", "incomplete", "invalid"])
def test_non_happy_extraction_results_produce_defined_response_not_a_lead(status):
    result = ExtractionResult(status=status, lead=Lead.empty(), usage=Usage(), message="defined message")
    assert result.status == status
    assert result.lead.name.value is None


def test_end_to_end_hallucination_is_nulled_via_api(client, sample_wav_bytes):
    transcript = "Hi I'm Priya from Acme, interested in GPUs."
    hallucinated_lead = _lead_with(
        name=("Priya", "I'm Priya"),
        company=("Totally Made Up Inc", "Totally Made Up Inc"),
    )
    transcriber = FakeTranscriber(script=[transcript])
    extractor = FakeExtractor(script=[ExtractionResult(status="ok", lead=hallucinated_lead, usage=Usage(10, 5))])
    app.dependency_overrides[get_transcriber] = lambda: transcriber
    app.dependency_overrides[get_extractor] = lambda: extractor
    try:
        response = client.post("/upload-audio", files={"file": ("sample.wav", sample_wav_bytes, "audio/wav")})
    finally:
        app.dependency_overrides.pop(get_transcriber, None)
        app.dependency_overrides.pop(get_extractor, None)

    assert response.status_code == 200
    body = response.json()
    assert body["lead"]["name"]["value"] == "Priya"
    assert body["lead"]["company"]["rejected"] is True
    assert body["lead"]["company"]["value"] is None
    assert "company" in body["rejected"]


@pytest.mark.parametrize("status", ["refusal", "incomplete", "invalid"])
def test_non_happy_extraction_via_api_returns_200_with_message(client, sample_wav_bytes, status):
    transcript = "Hi I'm Priya from Acme, interested in GPUs, this is long enough."
    transcriber = FakeTranscriber(script=[transcript])
    extractor = FakeExtractor(script=[ExtractionResult(status=status, lead=Lead.empty(), usage=Usage(), message=f"{status} happened")])
    app.dependency_overrides[get_transcriber] = lambda: transcriber
    app.dependency_overrides[get_extractor] = lambda: extractor
    try:
        response = client.post("/upload-audio", files={"file": ("sample.wav", sample_wav_bytes, "audio/wav")})
    finally:
        app.dependency_overrides.pop(get_transcriber, None)
        app.dependency_overrides.pop(get_extractor, None)

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == f"{status} happened"
    assert all(body["lead"][f]["value"] is None for f in body["lead"])
