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


# _ENV_EXPLICIT is the RAW env var: None if ENV was never set at all, as
# opposed to being set to some known value. The safety gate below must
# check this raw value, not a defaulted one: see verification-review
# finding 5.
_ENV_EXPLICIT = os.getenv("ENV")
ENV = _ENV_EXPLICIT or "development"

# Provider seam: "openai" talks to the real API, "fake" uses scripted
# in-memory providers for tests and offline CI. Fails closed: fake
# providers are only allowed when ENV was EXPLICITLY set by the deployer
# to one of the known-safe values below. Blocking only ENV == "production"
# would silently allow canned data in any other real deploy where ENV was
# left unset, mistyped, or set to something like "staging"/"prod" rather
# than "production" exactly.
#
# Verification-review finding 5: an earlier version of this guard checked
# the *defaulted* ENV (which falls back to "development" when the env var
# is unset at all), and "development" is itself in the safe set. That
# meant a real deploy that simply forgot to set ENV, with
# LEAD_PROVIDERS=fake left over from local testing, would silently serve
# canned fake data in what is actually production: unset is treated the
# same as the safest possible value instead of the least safe one. The
# gate now checks _ENV_EXPLICIT (None when ENV was never set), so an
# unset ENV is unsafe for fakes (fails closed) even though the *defaulted*
# ENV used elsewhere is still "development" for local-dev convenience.
# Local dev must set ENV=development explicitly (see .env.example) to use
# LEAD_PROVIDERS=fake.
LEAD_PROVIDERS = os.getenv("LEAD_PROVIDERS", "openai")
_FAKE_PROVIDER_SAFE_ENVS = {"development", "test"}
if LEAD_PROVIDERS == "fake" and _ENV_EXPLICIT not in _FAKE_PROVIDER_SAFE_ENVS:
    raise RuntimeError(
        f"LEAD_PROVIDERS=fake is only allowed when ENV is explicitly set to "
        f"one of {sorted(_FAKE_PROVIDER_SAFE_ENVS)} (got ENV={_ENV_EXPLICIT!r}); "
        "set LEAD_PROVIDERS=openai for any real deploy, or explicitly set "
        "ENV=development/test for local dev and CI."
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
