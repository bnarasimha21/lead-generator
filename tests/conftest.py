import os
import wave

import pytest

os.environ.setdefault("LEAD_PROVIDERS", "fake")
os.environ.setdefault("ENV", "test")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app, get_extractor, get_transcriber  # noqa: E402
from app.ratelimit import limiter as app_limiter  # noqa: E402
from tests.fakes import FakeExtractor, FakeTranscriber  # noqa: E402

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
SAMPLE_WAV_PATH = os.path.join(FIXTURES_DIR, "sample.wav")


def _ensure_sample_wav():
    os.makedirs(FIXTURES_DIR, exist_ok=True)
    if os.path.exists(SAMPLE_WAV_PATH):
        return
    with wave.open(SAMPLE_WAV_PATH, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * 800)  # 0.1s of silence


_ensure_sample_wav()


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    app_limiter._hits.clear()
    app_limiter._day = None
    app_limiter._day_count = 0
    yield


@pytest.fixture
def sample_wav_bytes():
    with open(SAMPLE_WAV_PATH, "rb") as f:
        return f.read()


@pytest.fixture
def fake_providers():
    transcriber = FakeTranscriber()
    extractor = FakeExtractor()
    app.dependency_overrides[get_transcriber] = lambda: transcriber
    app.dependency_overrides[get_extractor] = lambda: extractor
    yield transcriber, extractor
    app.dependency_overrides.pop(get_transcriber, None)
    app.dependency_overrides.pop(get_extractor, None)


@pytest.fixture
def client():
    return TestClient(app)
