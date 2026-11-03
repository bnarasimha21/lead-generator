"""Issue 4 acceptance criteria: the eval runner works offline in CI and
produces every metric; a configured accuracy floor fails the run with a
non-zero exit code when a metric falls below it.
"""

import asyncio

from app.providers import ExtractionResult, Usage
from app.schema import EvidencedField, Lead
from evals.run import run_fake
from evals.run import main


def test_fake_eval_run_prints_every_metric(capsys):
    exit_code = main(["--providers", "fake"])
    out = capsys.readouterr().out
    for metric in ("field accuracy", "field precision", "field recall", "hallucinated fields", "evidence validity rate", "word error rate"):
        assert metric in out
    assert exit_code == 0  # default floor is 0.0, the scripted-wrong fake run still passes it


def test_fake_eval_run_fails_when_floor_is_unreachable(capsys):
    exit_code = main(["--providers", "fake", "--accuracy-floor", "1.01"])
    assert exit_code == 1


def test_openai_eval_run_refuses_without_yes(capsys):
    exit_code = main(["--providers", "openai"])
    out = capsys.readouterr().out
    assert "Refusing to spend money without --yes" in out
    assert exit_code == 1


class _ScriptedExtractor:
    """Always returns one fixed ExtractionResult, used to check the eval
    scores what the real pipeline would actually produce, not the raw
    extractor output (finding 11)."""

    def __init__(self, result: ExtractionResult):
        self._result = result

    async def extract(self, transcript: str) -> ExtractionResult:
        return self._result


def test_golden_eval_scores_the_verified_pipeline_not_the_raw_extractor():
    """evals/run.py must call verify_lead before scoring (finding 11): a
    hallucinated field whose "evidence" is not a real quote from the
    transcript is exactly what app/main.py's own verify_lead() nulls out
    before a user ever sees it, so the eval's hallucinated_fields count
    must be 0 for this case, not 1, even though the raw extractor output
    (if scored directly) would show a hallucination.
    """
    transcript = "Hi, I'm Priya, calling about the product."
    hallucinated_lead = Lead.empty()
    hallucinated_lead.company = EvidencedField(value="Hallucinated Inc", evidence="Hallucinated Inc")
    result = ExtractionResult(status="ok", lead=hallucinated_lead, usage=Usage(10, 5))

    case = {
        "id": "synthetic-hallucination-case",
        "transcript": transcript,
        "expected": {"company": {"value": None, "evidence": None}},
    }

    report = asyncio.run(run_fake([case], extractor=_ScriptedExtractor(result)))

    assert report.total_hallucinated == 0
    assert report.field_accuracy == 1.0
