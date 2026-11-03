"""Issue 1 acceptance criteria: the fake-provider flow works with no API
key, oversized/garbage uploads are rejected before any provider is called,
provider failures map to a fixed 502, the event loop is never blocked by a
provider call, and no file is left behind in the app directory.
"""

import asyncio
import os

import httpx
import pytest

from app.main import app, get_extractor, get_transcriber, BASE_DIR
from tests.fakes import FakeExtractor, FakeTranscriber


def test_fake_provider_flow_completes_with_no_api_key(client, fake_providers, sample_wav_bytes):
    transcriber, extractor = fake_providers
    response = client.post("/upload-audio", files={"file": ("sample.wav", sample_wav_bytes, "audio/wav")})
    assert response.status_code == 200
    body = response.json()
    assert "lead" in body
    assert "transcript" in body
    assert len(transcriber.calls) == 1
    assert len(extractor.calls) == 1


def test_oversized_upload_returns_413_without_calling_transcriber(client, fake_providers):
    transcriber, extractor = fake_providers
    big_payload = b"RIFF" + b"\x00" * 50 * 1024 * 1024  # not even valid WAV, just big
    response = client.post(
        "/upload-audio",
        files={"file": ("big.wav", big_payload, "audio/wav")},
    )
    assert response.status_code == 413
    assert len(transcriber.calls) == 0


def test_renamed_text_file_returns_400(client, fake_providers):
    transcriber, extractor = fake_providers
    response = client.post(
        "/upload-audio",
        files={"file": ("fake.wav", b"this is not audio, just text", "audio/wav")},
    )
    assert response.status_code == 400
    assert len(transcriber.calls) == 0


def test_provider_exception_returns_502_without_leaking_exception_text(client, sample_wav_bytes):
    secret_detail = "super secret traceback detail: /etc/passwd"
    transcriber = FakeTranscriber(script=[Exception(secret_detail)])
    extractor = FakeExtractor()
    app.dependency_overrides[get_transcriber] = lambda: transcriber
    app.dependency_overrides[get_extractor] = lambda: extractor
    try:
        response = client.post("/upload-audio", files={"file": ("sample.wav", sample_wav_bytes, "audio/wav")})
    finally:
        app.dependency_overrides.pop(get_transcriber, None)
        app.dependency_overrides.pop(get_extractor, None)
    assert response.status_code == 502
    assert secret_detail not in response.text


def test_no_file_left_in_app_directory_after_request(client, fake_providers, sample_wav_bytes):
    before = set(os.listdir(BASE_DIR))
    client.post("/upload-audio", files={"file": ("sample.wav", sample_wav_bytes, "audio/wav")})
    after = set(os.listdir(BASE_DIR))
    assert after == before


def test_no_file_left_in_app_directory_after_failed_request(client, fake_providers):
    before = set(os.listdir(BASE_DIR))
    client.post("/upload-audio", files={"file": ("fake.wav", b"not audio", "audio/wav")})
    after = set(os.listdir(BASE_DIR))
    assert after == before


def test_slow_provider_does_not_block_health_endpoint(sample_wav_bytes):
    """Proves the handler awaits the provider instead of blocking the
    event loop: while one request is parked on a slow fake provider, a
    concurrent /health request still returns promptly.
    """
    slow_transcriber = FakeTranscriber(delay_seconds=1.0)
    extractor = FakeExtractor()
    app.dependency_overrides[get_transcriber] = lambda: slow_transcriber
    app.dependency_overrides[get_extractor] = lambda: extractor

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            slow_task = asyncio.ensure_future(
                ac.post("/upload-audio", files={"file": ("sample.wav", sample_wav_bytes, "audio/wav")})
            )
            await asyncio.sleep(0.1)  # let the slow request start first
            start = asyncio.get_event_loop().time()
            health_response = await ac.get("/health")
            health_elapsed = asyncio.get_event_loop().time() - start
            slow_response = await slow_task
            return health_response, health_elapsed, slow_response

    try:
        health_response, health_elapsed, slow_response = asyncio.run(run())
    finally:
        app.dependency_overrides.pop(get_transcriber, None)
        app.dependency_overrides.pop(get_extractor, None)

    assert health_response.status_code == 200
    assert health_elapsed < 0.5, "a slow provider call blocked the event loop"
    assert slow_response.status_code == 200
