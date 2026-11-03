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


def test_received_mark_reflects_the_asgi_middleware_timestamp_not_the_handler_start(client, fake_providers, sample_wav_bytes, monkeypatch):
    """The "received" mark used to start the clock inside the route
    handler, i.e. after Starlette had already parsed the multipart body
    to build the UploadFile, so it always read ~0ms and quietly excluded
    that parsing time. RateLimitAndBodySizeMiddleware now stamps the
    real receipt time onto the ASGI scope before anything touches the
    body (app/request_guard.py's RECEIVED_AT_SCOPE_KEY); prove the route
    handler actually uses that stamp, not its own construction time, by
    backdating only that one stamp and checking the backdating shows up
    in the "received" mark.

    time.monotonic is a single shared function looked up on the same
    module object everywhere, so it cannot be patched for just the
    middleware's stamp and not for the handler's own later marks by
    patching the name globally. Instead, patch it, let the middleware's
    stamp line pick up the backdated value, then restore the real clock
    as soon as RateLimiter.reserve (the very next line in the real
    middleware, after the stamp) is called, so every mark to make after
    that point uses the real clock and only the stamp is skewed.
    """
    import time

    from app import ratelimit as ratelimit_module

    real_monotonic = time.monotonic
    offset_seconds = 0.25

    def backdated_monotonic():
        return real_monotonic() - offset_seconds

    monkeypatch.setattr(time, "monotonic", backdated_monotonic)

    original_reserve = ratelimit_module.limiter.reserve

    def restoring_reserve(*args, **kwargs):
        monkeypatch.setattr(time, "monotonic", real_monotonic)
        return original_reserve(*args, **kwargs)

    monkeypatch.setattr(ratelimit_module.limiter, "reserve", restoring_reserve)

    response = client.post("/upload-audio", files={"file": ("sample.wav", sample_wav_bytes, "audio/wav")})
    assert response.status_code == 200
    received_ms = response.json()["timing"]["received"]
    # Generous slack below the backdating, still nowhere near the ~0ms
    # this mark would read if it timed from handler-construction instead
    # of the middleware's stamp.
    assert received_ms >= offset_seconds * 1000 * 0.5


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
