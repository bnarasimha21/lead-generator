"""Central configuration. No model id or price is a literal anywhere else
in the app; everything reads from here so a model swap is a one-line change.

Model defaults (checked against OpenAI's deprecations page and speech-to-text
guide on 2026-11-03, see README "Models" section for sources):
- TRANSCRIBE_MODEL defaults to "gpt-transcribe", the documented successor to
  "whisper-1" for recorded audio files.
- EXTRACT_MODEL defaults to "gpt-6-luna", the documented small/cheap chat
  model replacing gpt-4o-mini / gpt-4.1-mini (gpt-3.5-turbo itself has no
  direct small successor; "gpt-5.6-terra" is the like-for-like replacement
  but is not the small/cheap tier this project needs).
Both are overridable via env vars so a newer model can be dropped in without
a code change.
"""

import os


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _float_env(name: str, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


ENV = os.getenv("ENV", "development")

# Provider seam: "openai" talks to the real API, "fake" uses scripted
# in-memory providers for tests and offline CI. Refuses to start with fake
# providers in production so a misconfigured deploy cannot silently serve
# canned data.
LEAD_PROVIDERS = os.getenv("LEAD_PROVIDERS", "openai")
if LEAD_PROVIDERS == "fake" and ENV == "production":
    raise RuntimeError(
        "LEAD_PROVIDERS=fake is not allowed when ENV=production; "
        "set LEAD_PROVIDERS=openai or unset ENV."
    )

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

TRANSCRIBE_MODEL = os.getenv("TRANSCRIBE_MODEL", "gpt-transcribe")
EXTRACT_MODEL = os.getenv("EXTRACT_MODEL", "gpt-6-luna")

# Upload handling (issue 1).
MAX_UPLOAD_BYTES = _int_env("MAX_UPLOAD_BYTES", 25 * 1024 * 1024)  # 25 MB
UPLOAD_CHUNK_BYTES = _int_env("UPLOAD_CHUNK_BYTES", 1024 * 1024)

# Prompt / transcript hardening (issue 3).
MAX_TRANSCRIPT_CHARS = _int_env("MAX_TRANSCRIPT_CHARS", 4000)

# Extraction skip for very short transcripts (issue 5 speed lever).
MIN_WORDS_FOR_EXTRACTION = _int_env("MIN_WORDS_FOR_EXTRACTION", 3)

# Rate limiting / budget (issue 5). In-memory, single-process.
RATE_LIMIT_PER_IP = _int_env("RATE_LIMIT_PER_IP", 3)
RATE_LIMIT_WINDOW_SECONDS = _int_env("RATE_LIMIT_WINDOW_SECONDS", 60)
DAILY_CALL_BUDGET = _int_env("DAILY_CALL_BUDGET", 1000)

# Pricing (issue 5). USD per unit; unset (None) disables cost estimates.
# These are placeholders: read the live pricing page for the configured
# models before relying on the numbers, then set these env vars.
PRICE_PER_INPUT_TOKEN = _float_env("PRICE_PER_INPUT_TOKEN", None)
PRICE_PER_OUTPUT_TOKEN = _float_env("PRICE_PER_OUTPUT_TOKEN", None)
PRICE_PER_AUDIO_SECOND = _float_env("PRICE_PER_AUDIO_SECOND", None)

PORT = _int_env("PORT", 8000)
