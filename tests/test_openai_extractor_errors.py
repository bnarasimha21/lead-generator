"""Adversarial-review finding 7: with the real openai SDK (1.57+),
chat.completions.parse() raises pydantic's ValidationError and
openai.LengthFinishReasonError / openai.ContentFilterFinishReasonError
directly, rather than returning a completion object the old code's
choice.finish_reason / choice.message.refusal checks could inspect.

Before the fix, a bare `except Exception` swallowed all three into a
generic ProviderError/502, which is wrong: a truncated or
content-filtered or schema-invalid reply has a defined, non-502 outcome
elsewhere in this same method (incomplete / refusal / invalid), and
should reach it the same way whether the SDK raises or returns.

These tests mock the OpenAI client directly; no real API call is made.
"""

import asyncio
from types import SimpleNamespace

import pytest
from openai import ContentFilterFinishReasonError, LengthFinishReasonError
from pydantic import ValidationError

from app.providers import OpenAIExtractor, ProviderError

TRANSCRIPT = "Hi, I'm Priya from Acme Corp, interested in GPU pricing."


class _RaisingClient:
    """A fake OpenAI client whose chat.completions.parse() raises
    whatever exception it is constructed with, instead of making a real
    network call."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    async def _parse(self, **kwargs):
        raise self._exc


def _fake_completion_with_usage(prompt_tokens=7, completion_tokens=3):
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(usage=usage)


def test_length_finish_reason_error_maps_to_incomplete_with_usage():
    completion = _fake_completion_with_usage(prompt_tokens=11, completion_tokens=4)
    client = _RaisingClient(LengthFinishReasonError(completion=completion))
    extractor = OpenAIExtractor(client, model="gpt-6-luna")

    result = asyncio.run(extractor.extract(TRANSCRIPT))

    assert result.status == "incomplete"
    assert result.message == "model output truncated"
    assert result.lead.name.value is None
    # Usage is recovered from the exception's own completion, not dropped.
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 4


def test_length_finish_reason_error_without_usage_does_not_crash():
    completion = SimpleNamespace(usage=None)
    client = _RaisingClient(LengthFinishReasonError(completion=completion))
    extractor = OpenAIExtractor(client, model="gpt-6-luna")

    result = asyncio.run(extractor.extract(TRANSCRIPT))

    assert result.status == "incomplete"
    assert result.usage.input_tokens == 0
    assert result.usage.output_tokens == 0


def test_content_filter_finish_reason_error_maps_to_refusal():
    client = _RaisingClient(ContentFilterFinishReasonError())
    extractor = OpenAIExtractor(client, model="gpt-6-luna")

    result = asyncio.run(extractor.extract(TRANSCRIPT))

    assert result.status == "refusal"
    assert result.message == "model declined to extract"
    assert result.lead.name.value is None


def test_validation_error_from_parse_maps_to_invalid_not_502():
    exc = ValidationError.from_exception_data("Lead", [])
    client = _RaisingClient(exc)
    extractor = OpenAIExtractor(client, model="gpt-6-luna")

    result = asyncio.run(extractor.extract(TRANSCRIPT))

    assert result.status == "invalid"
    assert result.message == "model reply did not match the schema"


def test_an_unrelated_sdk_failure_still_raises_provider_error():
    """Finding 7 is about not over-catching into a 502; a genuinely
    unexpected failure (e.g. a connection error) must still map to the
    existing ProviderError/502 path, not silently succeed."""
    client = _RaisingClient(RuntimeError("boom"))
    extractor = OpenAIExtractor(client, model="gpt-6-luna")

    with pytest.raises(ProviderError):
        asyncio.run(extractor.extract(TRANSCRIPT))
