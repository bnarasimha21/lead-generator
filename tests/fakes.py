"""Backward-compatible re-export (adversarial-review finding 4).

The real implementation moved to app/providers_fake.py so it ships inside
the Docker image (which only COPYs app/, not tests/) and so
app/providers.py no longer imports from the test tree at runtime. Existing
test modules import `from tests.fakes import FakeExtractor, FakeTranscriber`;
this module keeps that import path working.
"""

from app.providers_fake import FakeExtractor, FakeTranscriber

__all__ = ["FakeExtractor", "FakeTranscriber"]
