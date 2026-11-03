"""Upload handling (issue 1): sniff the real container from the first
bytes instead of trusting the filename/extension, read in chunks with a
hard size limit so an oversized upload is rejected before it is fully
buffered in memory, and always write to a TemporaryDirectory that is
cleaned up in a `finally`, never the app directory.
"""

import tempfile
from dataclasses import dataclass
from typing import Optional

from fastapi import UploadFile

from app import config


class UploadTooLarge(Exception):
    pass


class UnrecognisedAudioFormat(Exception):
    pass


@dataclass
class SniffedFormat:
    extension: str
    content_type: str


# (magic bytes, offset, extension, content type)
_SIGNATURES = (
    (b"RIFF", 0, "wav", "audio/wav"),       # followed by WAVE at offset 8, checked below
    (b"OggS", 0, "ogg", "audio/ogg"),
    (b"\x1a\x45\xdf\xa3", 0, "webm", "audio/webm"),
    (b"ftyp", 4, "mp4", "audio/mp4"),
)


def sniff_audio_format(header: bytes) -> Optional[SniffedFormat]:
    for magic, offset, extension, content_type in _SIGNATURES:
        end = offset + len(magic)
        if len(header) >= end and header[offset:end] == magic:
            if extension == "wav":
                if len(header) >= 12 and header[8:12] == b"WAVE":
                    return SniffedFormat(extension, content_type)
                continue
            return SniffedFormat(extension, content_type)
    return None


async def read_upload_limited(file: UploadFile, max_bytes: int = None, chunk_size: int = None) -> bytes:
    """Read an UploadFile in chunks, raising UploadTooLarge as soon as the
    limit is exceeded, without buffering the whole oversized file first.
    """
    max_bytes = max_bytes if max_bytes is not None else config.MAX_UPLOAD_BYTES
    chunk_size = chunk_size if chunk_size is not None else config.UPLOAD_CHUNK_BYTES
    chunks = []
    total = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise UploadTooLarge(f"upload exceeds {max_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def make_temp_dir():
    return tempfile.TemporaryDirectory(prefix="lead-generator-upload-")
