"""Issue 5 acceptance criteria: timing marks, usage/cost reporting, the
latency budget check in the eval runner, the rate limit's 429 path, and
that logs never contain transcript text.
"""

import asyncio
import logging

from app import config
from app.main import app, get_extractor, get_transcriber
from app.providers import ExtractionResult, Usage
from app.ratelimit import RateLimiter
from app.schema import Lead
from app.timing import MARK_NAMES, Timing
from evals.run import LatencyBudgetExceeded, run_fake
from tests.fakes import FakeExtractor, FakeTranscriber


def test_every_successful_response_has_all_six_marks_in_order_and_header(client, fake_providers, sample_wav_bytes):
    response = client.post("/upload-audio", files={"file": ("sample.wav", sample_wav_bytes, "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert list(body["timing"].keys()) == list(MARK_NAMES)
    values = list(body["timing"].values())
    assert values == sorted(values)
    assert "Server-Timing" in response.headers
    for name in MARK_NAMES:
        assert name in response.headers["Server-Timing"]


def test_usage_and_cost_reported_matches_hand_calculation(client, sample_wav_bytes, monkeypatch):
    monkeypatch.setattr(config, "PRICE_PER_INPUT_TOKEN", 0.0000005)
    monkeypatch.setattr(config, "PRICE_PER_OUTPUT_TOKEN", 0.0000015)
    monkeypatch.setattr(config, "PRICE_PER_AUDIO_SECOND", 0.0001)

    from app.providers import TranscriptionResult

    transcriber = FakeTranscriber(
        script=[TranscriptionResult(text="a reasonably long transcript for testing purposes here", audio_seconds=12.0)]
    )
    extractor = FakeExtractor(script=[ExtractionResult(status="ok", lead=Lead.empty(), usage=Usage(input_tokens=1000, output_tokens=200))])

    app.dependency_overrides[get_transcriber] = lambda: transcriber
    app.dependency_overrides[get_extractor] = lambda: extractor
    try:
        response = client.post("/upload-audio", files={"file": ("sample.wav", sample_wav_bytes, "audio/wav")})
    finally:
        app.dependency_overrides.pop(get_transcriber, None)
        app.dependency_overrides.pop(get_extractor, None)

    body = response.json()
    usage = body["usage"]
    assert usage["input_tokens"] == 1000
    assert usage["output_tokens"] == 200
    assert usage["audio_seconds"] == 12.0
    expected_cost = round(1000 * 0.0000005 + 200 * 0.0000015 + 12.0 * 0.0001, 6)
    assert usage["estimated_cost"] == expected_cost


def test_latency_budget_trips_in_eval_runner():
    slow_extractor = FakeExtractor(delay_seconds=0.3)
    cases = [{"id": "slow1", "transcript": "hello world", "expected": {}}]
    try:
        asyncio.run(run_fake(cases, extractor=slow_extractor, latency_budget_ms=50))
        assert False, "expected LatencyBudgetExceeded"
    except LatencyBudgetExceeded:
        pass


def test_nth_plus_one_request_returns_429_with_retry_after(client, fake_providers, sample_wav_bytes):
    from app import ratelimit as ratelimit_module

    original = ratelimit_module.limiter
    ratelimit_module.limiter = RateLimiter(limit=3, window_seconds=60, daily_budget=1000)
    # app.main imported `limiter` by reference at module load time, so patch
    # the name it actually holds.
    import app.main as main_module
    main_module.limiter = ratelimit_module.limiter
    try:
        for _ in range(3):
            resp = client.post("/upload-audio", files={"file": ("sample.wav", sample_wav_bytes, "audio/wav")})
            assert resp.status_code == 200
        fourth = client.post("/upload-audio", files={"file": ("sample.wav", sample_wav_bytes, "audio/wav")})
        assert fourth.status_code == 429
        assert "Retry-After" in fourth.headers
    finally:
        ratelimit_module.limiter = original
        main_module.limiter = original


def test_logs_never_contain_transcript_text(client, caplog, sample_wav_bytes):
    secret_transcript = "the secret phrase xyzzy-marker-12345 should never be logged"
    transcriber = FakeTranscriber(script=[secret_transcript])
    extractor = FakeExtractor()
    app.dependency_overrides[get_transcriber] = lambda: transcriber
    app.dependency_overrides[get_extractor] = lambda: extractor
    try:
        with caplog.at_level(logging.INFO):
            client.post("/upload-audio", files={"file": ("sample.wav", sample_wav_bytes, "audio/wav")})
    finally:
        app.dependency_overrides.pop(get_transcriber, None)
        app.dependency_overrides.pop(get_extractor, None)

    for record in caplog.records:
        assert "xyzzy-marker-12345" not in record.getMessage()
