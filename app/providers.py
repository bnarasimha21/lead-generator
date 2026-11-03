"""The provider seam (issue 1): Transcriber and Extractor protocols, OpenAI
implementations behind the async client (so a slow call never blocks the
event loop), and a factory picking providers from config.

Fakes used by tests live in tests/fakes.py; they implement the same
protocols so tests exercise the real app code path end to end.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

from pydantic import ValidationError

from app import config
from app.schema import Lead
from app.security import SYSTEM_PROMPT, build_user_message

logger = logging.getLogger("lead_generator.providers")


@dataclass
class TranscriptionResult:
    text: str
    audio_seconds: Optional[float] = None
    language: Optional[str] = None


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ExtractionResult:
    status: str  # "ok" | "refusal" | "incomplete" | "invalid"
    lead: Lead = field(default_factory=Lead.empty)
    usage: Usage = field(default_factory=Usage)
    message: Optional[str] = None


class ProviderError(Exception):
    """Raised when a provider call fails outside the handled, defined
    outcomes (refusal / incomplete / invalid); the handler maps this to a
    502 with a fixed message.
    """


class Transcriber(Protocol):
    async def transcribe(self, audio_bytes: bytes, filename: str, content_type: str) -> TranscriptionResult: ...


class Extractor(Protocol):
    async def extract(self, transcript: str) -> ExtractionResult: ...


class OpenAITranscriber:
    def __init__(self, client, model: str = None):
        self._client = client
        self._model = model or config.TRANSCRIBE_MODEL

    async def transcribe(self, audio_bytes: bytes, filename: str, content_type: str) -> TranscriptionResult:
        try:
            response = await self._client.audio.transcriptions.create(
                model=self._model,
                file=(filename, audio_bytes, content_type),
            )
        except Exception as exc:  # pragma: no cover - exercised via ProviderError path
            logger.exception("transcription provider call failed")
            raise ProviderError("transcription failed") from exc
        duration = getattr(response, "duration", None)
        language = getattr(response, "language", None)
        return TranscriptionResult(text=response.text or "", audio_seconds=duration, language=language)


class OpenAIExtractor:
    def __init__(self, client, model: str = None):
        self._client = client
        self._model = model or config.EXTRACT_MODEL

    async def extract(self, transcript: str) -> ExtractionResult:
        user_message = build_user_message(transcript)
        try:
            completion = await self._client.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                response_format=Lead,
            )
        except Exception as exc:
            logger.exception("extraction provider call failed")
            raise ProviderError("extraction failed") from exc

        choice = completion.choices[0]
        usage = Usage(
            input_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
        )

        if getattr(choice.message, "refusal", None):
            logger.info("extraction refused by model")
            return ExtractionResult(status="refusal", usage=usage, message="model declined to extract")

        if choice.finish_reason == "length":
            logger.info("extraction truncated (finish_reason=length)")
            return ExtractionResult(status="incomplete", usage=usage, message="model output truncated")

        parsed = getattr(choice.message, "parsed", None)
        if parsed is None:
            try:
                parsed = Lead.model_validate_json(choice.message.content or "")
            except (ValidationError, ValueError, TypeError):
                logger.info("extraction reply failed schema validation")
                return ExtractionResult(status="invalid", usage=usage, message="model reply did not match the schema")

        return ExtractionResult(status="ok", lead=parsed, usage=usage)


def build_transcriber() -> Transcriber:
    if config.LEAD_PROVIDERS == "fake":
        from tests.fakes import FakeTranscriber

        return FakeTranscriber()
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return OpenAITranscriber(client)


def build_extractor() -> Extractor:
    if config.LEAD_PROVIDERS == "fake":
        from tests.fakes import FakeExtractor

        return FakeExtractor()
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    return OpenAIExtractor(client)
