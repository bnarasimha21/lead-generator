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


def test_oversized_chunked_upload_against_the_real_app_returns_413_not_400(low_limit, monkeypatch):
    """Verification-review finding 4: a prior fix's _BodyTooLarge was a
    plain Exception. Raised from inside FastAPI's own multipart body
    parsing (triggered here by File(...) on /upload-audio), FastAPI's
    routing only re-raises an exception unchanged if it is already an
    HTTPException; a plain Exception got rewritten into a generic 400
    "There was an error parsing the body", which Starlette's own
    exception-handling middleware (inside the real FastAPI app) sent
    before this middleware's own except-clause ever saw anything
    propagate. The client got 400, never the intended 413.

    This drives a REAL chunked request (no Content-Length header at all,
    httpx uses Transfer-Encoding: chunked because the body is an async
    generator of unknown length) through the real ASGI app
    (app.main.app), exactly the path that hid the bug: a unit test that
    calls the middleware directly, or synthesizes the multipart body
    itself, would never go through FastAPI's own request.form() parsing
    and so would never exercise this failure mode.
    """
    from app import config

    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 100)

    boundary = "leadgenteststreamboundary"
    payload = b"x" * 1000  # comfortably over the 100-byte cap
    header = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="big.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode()
    footer = f"\r\n--{boundary}--\r\n".encode()

    async def chunked_body():
        yield header
        # Stream the payload in small pieces so the middleware's running
        # byte count crosses MAX_UPLOAD_BYTES mid-stream, the same way a
        # real large upload would arrive.
        for i in range(0, len(payload), 50):
            yield payload[i : i + 50]
            await asyncio.sleep(0)
        yield footer

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/upload-audio",
                content=chunked_body(),
                headers={"content-type": f"multipart/form-data; boundary={boundary}"},
            )

    response = asyncio.run(run())

    # Confirms this really was a chunked request with no declared length,
    # i.e. the exact code path the finding describes.
    assert "content-length" not in {h.lower() for h in response.request.headers}

    assert response.status_code == 413
    assert response.json() == {"error": "The uploaded file is too large."}


def test_proxy_trust_assumption_is_documented_and_not_the_leftmost_wildcard():
    """Verification-review finding 2: --forwarded-allow-ips='*' trusts the
    LEFT-MOST (client-controlled) entry of X-Forwarded-For, letting an
    attacker mint an unlimited number of per-IP buckets by rotating the
    header. Neither Procfile nor the Dockerfile's CMD may pass that flag
    (or --proxy-headers, which only matters together with it) anymore, and
    the right-most-entry assumption must still be written down in the
    README, not just coded.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, "Procfile")) as f:
        procfile = f.read()
    with open(os.path.join(repo_root, "Dockerfile")) as f:
        dockerfile = f.read()
    with open(os.path.join(repo_root, "README.md")) as f:
        readme = f.read()

    assert "--proxy-headers" not in procfile
    assert "--forwarded-allow-ips" not in procfile
    assert "--proxy-headers" not in dockerfile
    assert "--forwarded-allow-ips" not in dockerfile
    assert "x-forwarded-for" in readme.lower()
    assert "right-most" in readme.lower() or "rightmost" in readme.lower()


def test_client_ip_uses_rightmost_x_forwarded_for_entry():
    """_client_ip must trust the right-most X-Forwarded-For entry (the one
    appended by the real reverse proxy connecting to this process), not
    any entry a client could have put in front of it."""
    from app.request_guard import _client_ip

    scope = {
        "headers": [
            (b"x-forwarded-for", b"203.0.113.9, 198.51.100.4, 10.0.0.7"),
        ],
    }
    assert _client_ip(scope) == "10.0.0.7"


def test_client_ip_falls_back_to_scope_client_without_xff_header():
    from app.request_guard import _client_ip

    scope = {"headers": [], "client": ("192.0.2.1", 5000)}
    assert _client_ip(scope) == "192.0.2.1"


def test_spoofed_leftmost_xff_cannot_evade_the_per_ip_limit(low_limit):
    """The reviewer's exact repro: an attacker rotates the left-most,
    client-supplied X-Forwarded-For entry on every request, in front of
    the real IP a trusted proxy appended at the end. The per-IP limit must
    still bind to that real (right-most) IP, regardless of what the client
    puts earlier in the chain, so the attacker cannot mint a fresh bucket
    per request and drain DAILY_CALL_BUDGET.
    """
    from app import request_guard

    real_proxy_appended_ip = "10.0.0.7"

    async def inner_app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    middleware = request_guard.RateLimitAndBodySizeMiddleware(inner_app)

    async def run_one(spoofed_leftmost: str):
        headers = [
            (b"content-length", b"0"),
            (
                b"x-forwarded-for",
                f"{spoofed_leftmost}, {real_proxy_appended_ip}".encode(),
            ),
        ]
        scope = {
            "type": "http",
            "path": request_guard.UPLOAD_PATH,
            "method": "POST",
            "client": ("198.51.100.200", 4321),  # the proxy's own TCP peer address
            "headers": headers,
        }

        async def receive():
            return {"type": "http.disconnect"}

        sent = []

        async def send(message):
            sent.append(message)

        await middleware(scope, receive, send)
        return sent[0]["status"]

    async def run():
        # A different, attacker-chosen left-most IP on every single
        # request, but the same real right-most IP every time.
        return [await run_one(f"1.2.3.{i}") for i in range(low_limit.limit + 1)]

    statuses = asyncio.run(run())

    assert statuses.count(200) == low_limit.limit
    assert statuses.count(429) == 1
    # The limiter's bucket must be keyed on the real, right-most IP, not
    # any of the spoofed left-most values.
    assert len(low_limit._hits[real_proxy_appended_ip]) == low_limit.limit
    for i in range(low_limit.limit + 1):
        assert f"1.2.3.{i}" not in low_limit._hits
