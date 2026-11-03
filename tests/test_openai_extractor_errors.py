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

Verification-review blocker: the pinned openai==1.57.4 only exposes the
structured-output .parse() helper at client.beta.chat.completions.parse,
not client.chat.completions.parse (checked against the installed SDK's
actual attributes). The fake client below mirrors exactly that shape: it
exposes beta.chat.completions.parse and deliberately has no top-level
`chat` attribute at all, so a regression back to the wrong call site
raises AttributeError (caught by the generic except and turned into a
ProviderError) instead of silently passing.

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
    """A fake OpenAI client whose beta.chat.completions.parse() raises
    whatever exception it is constructed with, instead of making a real
    network call. Matches the real openai==1.57.4 client's actual shape:
    there is intentionally no top-level `chat` attribute, only `beta.chat`,
    so a call site using the wrong (non-beta) path fails loudly here too."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=self._parse)))

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


class _SucceedingClient:
    """A fake OpenAI client whose beta.chat.completions.parse() returns a
    successful completion, matching the real openai==1.57.4 shape: only
    beta.chat.completions.parse exists, there is no top-level `chat`."""

    def __init__(self, completion):
        self._completion = completion
        self.beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=self._parse)))

    async def _parse(self, **kwargs):
        return self._completion


def test_extract_calls_beta_chat_completions_parse_not_the_non_beta_path():
    """Verification-review blocker: the pinned SDK (1.57.4) only exposes
    the structured-output .parse() helper under `beta`, not directly on
    `chat.completions`. Calling the non-beta path raises AttributeError on
    every real call. This test pins the call site itself (not just the
    outcome), so a regression back to self._client.chat.completions.parse
    fails here even though the fake client has no top-level `chat` at all
    to attribute-error against."""
    from app.schema import EvidencedField, Lead

    lead = Lead(
        name=EvidencedField(value="Priya", evidence="I'm Priya"),
    )
    message = SimpleNamespace(parsed=lead, refusal=None, content=lead.model_dump_json())
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=20, completion_tokens=8)
    completion = SimpleNamespace(choices=[choice], usage=usage)

    client = _SucceedingClient(completion)
    extractor = OpenAIExtractor(client, model="gpt-6-luna")

    result = asyncio.run(extractor.extract(TRANSCRIPT))

    assert result.status == "ok"
    assert result.lead.name.value == "Priya"
    assert result.usage.input_tokens == 20
    assert result.usage.output_tokens == 8


def test_old_call_site_shape_would_have_raised_attribute_error():
    """Documents the exact bug from the verification review: a client
    shaped like the real SDK (only beta.chat.completions.parse exists) has
    no top-level chat.completions.parse, so the old call site would have
    raised AttributeError, caught by the broad except and turned into a
    generic ProviderError/502 on every real, non-mocked extraction call."""
    client = _SucceedingClient(completion=None)

    with pytest.raises(AttributeError):
        client.chat.completions.parse  # noqa: B018 - intentional attribute probe
