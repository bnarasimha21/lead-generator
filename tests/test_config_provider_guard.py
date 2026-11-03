"""LOW item: LEAD_PROVIDERS=fake must fail closed. The guard used to
only block the single literal ENV == "production"; any other real
deploy (ENV unset, mistyped, or set to something like "staging") would
silently start with fake, canned providers instead of refusing to
start.

app/config.py runs this check at import time from a module-level
env var read, so each case needs a fresh Python process with a
controlled environment rather than monkeypatching an already-imported
module.
"""

import os
import subprocess
import sys


def _import_config(env_overrides: dict) -> subprocess.CompletedProcess:
    # pytest's own process already has ENV=test (set by conftest.py's
    # os.environ.setdefault) by the time any test runs, so simply omitting
    # "ENV" from env_overrides does NOT produce a subprocess with ENV
    # unset: it would inherit ENV=test from os.environ below. A case that
    # wants to test truly-unset ENV must pass env_overrides={"ENV": None},
    # which this helper turns into an actual deletion from the merged env.
    env = {**os.environ, **env_overrides}
    env = {key: value for key, value in env.items() if value is not None}
    return subprocess.run(
        [sys.executable, "-c", "import app.config"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        env=env,
        capture_output=True,
        text=True,
    )


def test_fake_providers_refused_when_env_unset():
    """Verification-review finding 5: ENV left unset entirely must fail
    closed, not silently default to the safe "development" value. Without
    this, a real deploy that simply forgot to set ENV, with
    LEAD_PROVIDERS=fake left over from local testing, would silently serve
    canned fake data in what is actually production.
    """
    result = _import_config({"LEAD_PROVIDERS": "fake", "ENV": None})
    assert result.returncode != 0
    assert "LEAD_PROVIDERS=fake is only allowed" in result.stderr
    assert "ENV=None" in result.stderr


def test_fake_providers_allowed_when_env_explicitly_development():
    """Control case for the finding-5 fix: explicitly setting
    ENV=development (not relying on any default) must still allow fakes,
    distinguishing "explicitly set to a safe value" from "unset"."""
    result = _import_config({"LEAD_PROVIDERS": "fake", "ENV": "development"})
    assert result.returncode == 0, result.stderr


def test_fake_providers_refused_for_an_unrecognised_env_value():
    """The exact gap this guard closes: a deploy that sets some other
    real-sounding ENV value (a typo, or "staging") used to sail straight
    through because it just wasn't the literal string "production"."""
    result = _import_config({"LEAD_PROVIDERS": "fake", "ENV": "staging"})
    assert result.returncode != 0
    assert "LEAD_PROVIDERS=fake is only allowed" in result.stderr


def test_fake_providers_refused_in_production():
    result = _import_config({"LEAD_PROVIDERS": "fake", "ENV": "production"})
    assert result.returncode != 0
    assert "LEAD_PROVIDERS=fake is only allowed" in result.stderr


def test_fake_providers_allowed_in_development():
    result = _import_config({"LEAD_PROVIDERS": "fake", "ENV": "development"})
    assert result.returncode == 0, result.stderr


def test_fake_providers_allowed_in_test():
    result = _import_config({"LEAD_PROVIDERS": "fake", "ENV": "test"})
    assert result.returncode == 0, result.stderr


def test_openai_providers_allowed_regardless_of_env():
    result = _import_config({"LEAD_PROVIDERS": "openai", "ENV": "production"})
    assert result.returncode == 0, result.stderr
