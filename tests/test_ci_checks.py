"""Adversarial-review findings 2 and 3: the CI secret-scan and
GPU/Droplet-copy checks in .github/workflows/ci.yml must pass against this
repo's own legitimate files, while still catching the real thing they are
meant to catch.

Verification-review finding 7: the secret-scan pattern also needed to
catch the hyphenated real-key prefixes (sk-proj-, sk-svcacct-, sk-ant-)
that the original alphanumeric-only character class could never match,
and the .env.example exclude was removed since the length floor alone
already clears that file's placeholder text.

These tests run the exact shell commands ci.yml runs (subprocess,
read-only grep) against the real repo tree.
"""

import os
import re
import subprocess

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SECRET_PATTERN = r"sk-[A-Za-z0-9_-]{20,}"
GPU_DROPLET_PATTERN = r"(amd[-_ ]?)?gpu[-_ ]?droplet"


def _grep_available():
    try:
        subprocess.run(["grep", "--version"], capture_output=True, timeout=5)
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _grep_available(), reason="grep not available")


def test_secret_scan_regex_passes_on_the_repo_as_it_stands():
    """This is the literal command from ci.yml's "Secret pattern check"
    step. grep exits 1 when nothing matches; the step only fails the build
    when grep exits 0 (something matched).

    Verification-review finding 7: the old command also hard-excluded
    .env.example; it no longer does, so this now scans that file too (it
    relies on the length floor alone to clear the placeholder, exercised
    directly by the test below)."""
    result = subprocess.run(
        ["grep", "-rIn", "--exclude-dir=.git", "-E", SECRET_PATTERN, "."],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, f"secret-scan false positive(s):\n{result.stdout}"


def test_secret_scan_regex_still_catches_a_realistic_key_shape():
    fake_real_key = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"  # 29 chars, matches a real key's length
    assert re.search(SECRET_PATTERN, fake_real_key)


def test_secret_scan_regex_catches_hyphenated_real_key_prefixes():
    """Verification-review finding 7: real OpenAI/Anthropic keys are
    issued with a hyphenated sub-prefix (sk-proj-..., sk-svcacct-...,
    sk-ant-...). The old alphanumeric-only character class
    ([A-Za-z0-9]{20,}) could never match these: the hyphen right after
    the sub-prefix breaks the run of 20+ alphanumeric characters the
    pattern required, so a leaked "sk-proj-<40 real chars>" key sailed
    straight through the scan untouched."""
    for fake_real_key in (
        "sk-proj-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4",
        "sk-svcacct-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4",
        "sk-ant-api03-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4",
    ):
        assert re.search(SECRET_PATTERN, fake_real_key), fake_real_key


def test_secret_scan_regex_does_not_match_env_example_placeholder_even_unexcluded():
    """The tightened length floor alone, with no --exclude flag at all,
    must not match the documented placeholder."""
    env_example_path = os.path.join(REPO_ROOT, ".env.example")
    with open(env_example_path) as f:
        content = f.read()
    assert "OPENAI_API_KEY=sk-placeholder" in content, "placeholder text moved; update this test"
    assert not re.search(SECRET_PATTERN, content)


def test_gpu_droplet_copy_check_passes_on_the_repos_own_legitimate_text():
    """This is the literal command from ci.yml's "README / page copy
    check" step, run against this repo's real files, which legitimately
    say things like "GPU instances" and "Self-hosted Whisper on GPU"."""
    targets = ["README.md", "app/templates/index.html"] + [
        os.path.join("app", "static", name)
        for name in os.listdir(os.path.join(REPO_ROOT, "app", "static"))
        if name.endswith(".js")
    ]
    result = subprocess.run(
        ["grep", "-riE", GPU_DROPLET_PATTERN, *targets],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, f"GPU/Droplet false positive(s):\n{result.stdout}"


def test_gpu_droplet_copy_check_still_catches_the_real_stray_phrase():
    for sample in ("AMD GPU Droplet", "gpu-droplet", "GPU Droplet project", "amd_gpu_droplet"):
        assert re.search(GPU_DROPLET_PATTERN, sample, re.IGNORECASE), sample


def test_gpu_droplet_copy_check_does_not_flag_plain_gpu_or_droplet_mentions():
    for sample in ("GPU instances", "Self-hosted Whisper on GPU", "GPU pricing", "a web droplet somewhere"):
        assert re.search(GPU_DROPLET_PATTERN, sample, re.IGNORECASE) is None
