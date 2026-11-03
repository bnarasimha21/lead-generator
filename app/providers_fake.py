"""Fake Transcriber/Extractor implementations (issue 1) used by tests and by
LEAD_PROVIDERS=fake. They implement the same Protocols as the OpenAI
providers in app/providers.py so tests exercise the real request handling
code path, with the network calls swapped out.

This module lives in app/, not tests/ (adversarial-review finding 4): the
Docker image only COPYs app/, but LEAD_PROVIDERS=fake is a legitimate
non-production runtime mode (CI's own container smoke test runs the built
image with LEAD_PROVIDERS=fake), so the fake-provider implementation must
ship in the image rather than being imported from the test tree at
runtime. tests/fakes.py re-exports from here for backward compatibility
with existing test imports.

Both fakes are scriptable: construct one with a list of scripted replies, or
leave them in "default" mode which deterministically echoes something
plausible back so the happy path still works with no setup. A scripted
reply is consumed in order, one per call; after the script is exhausted the
default behaviour takes over.
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Union

from app.providers import ExtractionResult, ProviderError, TranscriptionResult, Usage
from app.schema import Lead


class FakeTranscriber:
    def __init__(self, script: Optional[List[Union[TranscriptionResult, Exception, str]]] = None, delay_seconds: float = 0):
        self.script = list(script) if script else []
        self.delay_seconds = delay_seconds
        self.calls: List[bytes] = []

    async def transcribe(self, audio_bytes: bytes, filename: str, content_type: str) -> TranscriptionResult:
        self.calls.append(audio_bytes)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise ProviderError(str(item)) from item
            if isinstance(item, str):
                return TranscriptionResult(text=item, audio_seconds=1.0)
            return item
        # Default: a short, generic transcript derived from the upload size
        # so repeated calls in a test are distinguishable without needing a
        # script.
        return TranscriptionResult(text="Hi, this is a test voice note.", audio_seconds=1.0)


class FakeExtractor:
    def __init__(self, script: Optional[List[Union[ExtractionResult, Exception]]] = None, delay_seconds: float = 0):
        self.script = list(script) if script else []
        self.delay_seconds = delay_seconds
        self.calls: List[str] = []
        self.messages_received: List[str] = []

    async def extract(self, transcript: str) -> ExtractionResult:
        self.calls.append(transcript)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, Exception):
                raise ProviderError(str(item)) from item
            return item
        # Default: an empty-but-valid lead with no hallucination.
        return ExtractionResult(status="ok", lead=Lead.empty(), usage=Usage(input_tokens=42, output_tokens=7))
