"""Adversarial-review findings 5, 6, 8: the rate-limit/budget race under
concurrent requests, the ASGI-level body-size cap, and the X-Forwarded-For
trust setting.
"""

import asyncio
import os

import httpx
import pytest

from app import ratelimit as ratelimit_module
from app.main import app, get_extractor, get_transcriber
from app.ratelimit import RateLimiter
from tests.fakes import FakeExtractor, FakeTranscriber


def _big_wav_bytes(size: int) -> bytes:
    header = b"RIFF\x00\x00\x00\x00WAVEfmt "
    return header + (b"\x00" * (size - len(header)))


@pytest.fixture
def low_limit():
    original = ratelimit_module.limiter
    custom = RateLimiter(limit=2, window_seconds=60, daily_budget=1000)
    ratelimit_module.limiter = custom
    yield custom
    ratelimit_module.limiter = original


def test_concurrent_large_uploads_from_one_ip_do_not_exceed_the_limit(low_limit):
    """The review's exact probe: several large concurrent uploads from one
    IP against a low limit. Before the fix, check() then an awaited
    (chunked) body read then record() left a window where every
    concurrently-scheduled request could observe "allowed" before any of
    them recorded, letting all of them through regardless of the limit.
    """
    payload = _big_wav_bytes(5 * 1024 * 1024)  # 5 MB, comfortably under MAX_UPLOAD_BYTES
    transcriber = FakeTranscriber(delay_seconds=0.05)
    extractor = FakeExtractor()
    app.dependency_overrides[get_transcriber] = lambda: transcriber
    app.dependency_overrides[get_extractor] = lambda: extractor

    concurrency = 8

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as client:
            tasks = [
                client.post("/upload-audio", files={"file": ("sample.wav", payload, "audio/wav")})
                for _ in range(concurrency)
            ]
            return await asyncio.gather(*tasks)

    try:
        responses = asyncio.run(run())
    finally:
        app.dependency_overrides.pop(get_transcriber, None)
        app.dependency_overrides.pop(get_extractor, None)

    statuses = [r.status_code for r in responses]
    assert statuses.count(200) == low_limit.limit, statuses
    assert statuses.count(429) == concurrency - low_limit.limit, statuses
    # The provider must never have been called more times than the limit
    # allows, regardless of how many requests raced to get there.
    assert len(transcriber.calls) == low_limit.limit


def test_oversized_request_with_known_content_length_is_rejected_without_reaching_the_handler(low_limit):
    """A Content-Length that already exceeds the limit must be rejected by
    the ASGI middleware itself, before the route handler (and therefore
    the provider) ever sees the request, and must still consume the
    reserved rate-limit slot (finding 6: a flood of oversized requests
    must not dodge the budget check)."""
    from app import config

    oversized = _big_wav_bytes(config.MAX_UPLOAD_BYTES + 1024)
    transcriber = FakeTranscriber()
    extractor = FakeExtractor()
    app.dependency_overrides[get_transcriber] = lambda: transcriber
    app.dependency_overrides[get_extractor] = lambda: extractor

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/upload-audio", files={"file": ("big.wav", oversized, "audio/wav")})

    try:
        response = asyncio.run(run())
    finally:
        app.dependency_overrides.pop(get_transcriber, None)
        app.dependency_overrides.pop(get_extractor, None)

    assert response.status_code == 413
    assert len(transcriber.calls) == 0
    # The rejection must have consumed one of the two available slots.
    # httpx's ASGITransport reports a fixed fake peer address by default.
    assert len(low_limit._hits["127.0.0.1"]) == 1


def test_repeated_oversized_requests_eventually_hit_the_rate_limit(low_limit):
    """The exact concern in finding 6: if oversized rejections rolled back
    the reservation, a client could send unlimited garbage and never once
    see a 429. With limit=2, a third oversized request in the same window
    must itself be rate-limited.
    """
    from app import config

    oversized = _big_wav_bytes(config.MAX_UPLOAD_BYTES + 1024)

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return [
                (await client.post("/upload-audio", files={"file": ("big.wav", oversized, "audio/wav")})).status_code
                for _ in range(3)
            ]

    statuses = asyncio.run(run())
    assert statuses == [413, 413, 429], statuses


def test_oversized_streaming_body_without_content_length_is_aborted_before_full_buffering(low_limit, monkeypatch):
    """A request that does not declare Content-Length (e.g. chunked
    transfer) must still be capped: the middleware counts bytes as they
    stream in and aborts before the whole oversized body reaches the
    downstream app, let alone gets fully buffered.

    Exercises the middleware directly against a minimal downstream ASGI
    app that just keeps calling receive() in a loop, the way Starlette's
    own body/form parsing does, so this does not depend on FastAPI's
    multipart parser's exact behaviour on a malformed body.
    """
    from app import config, request_guard

    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 10)

    received = []

    async def inner_app(scope, receive, send):
        while True:
            message = await receive()
            received.append(message)

    middleware = request_guard.RateLimitAndBodySizeMiddleware(inner_app)
    scope = {
        "type": "http",
        "path": "/upload-audio",
        "method": "POST",
        "client": ("203.0.113.5", 1234),
        "headers": [],
    }

    chunks = [b"x" * 4, b"x" * 4, b"x" * 4, b"x" * 4]  # 16 bytes total, over the 10-byte cap

    async def receive():
        if chunks:
            body = chunks.pop(0)
            return {"type": "http.request", "body": body, "more_body": bool(chunks)}
        return {"type": "http.disconnect"}

    sent = []

    async def send(message):
        sent.append(message)

    asyncio.run(middleware(scope, receive, send))

    assert sent[0]["type"] == "http.response.start"
    assert sent[0]["status"] == 413
    # Aborted after the 3rd chunk pushed the running total (12 bytes) over
    # the 10-byte cap: the inner app must only have seen the first two.
    assert len(received) == 2


def test_forwarded_allow_ips_documented_assumption_is_configured():
    """Finding 8: uvicorn must be told which proxy to trust for
    X-Forwarded-For, or every user behind a reverse proxy (e.g. Railway)
    shares one rate-limit bucket via the proxy's own IP. Procfile and
    Dockerfile both must opt in explicitly, and the assumption (this app
    is only reachable through its platform's own proxy) must be written
    down, not just coded.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, "Procfile")) as f:
        procfile = f.read()
    with open(os.path.join(repo_root, "Dockerfile")) as f:
        dockerfile = f.read()
    with open(os.path.join(repo_root, "README.md")) as f:
        readme = f.read()

    assert "--proxy-headers" in procfile
    assert "--forwarded-allow-ips" in procfile
    assert "--proxy-headers" in dockerfile
    assert "--forwarded-allow-ips" in dockerfile
    assert "forwarded-allow-ips" in readme.lower() or "x-forwarded-for" in readme.lower()
