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
    env = {**os.environ, **env_overrides}
    return subprocess.run(
        [sys.executable, "-c", "import app.config"],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        env=env,
        capture_output=True,
        text=True,
    )


def test_fake_providers_refused_when_env_unset():
    env = {"LEAD_PROVIDERS": "fake"}
    env.pop("ENV", None)
    result = _import_config(env)
    # ENV unset defaults to "development", which is explicitly safe.
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
