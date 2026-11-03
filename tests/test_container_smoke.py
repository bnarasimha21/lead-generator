"""Issue 6 acceptance criteria: the built container passes its health
check and completes the fake-provider flow. Skipped unless Docker is
actually available, since this is a smoke test for CI/local docker, not
something that should block a plain `pytest` run on a laptop without
Docker.
"""

import os
import shutil
import socket
import subprocess
import time

import pytest
import requests

def _docker_daemon_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=5).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


pytestmark = pytest.mark.skipif(not _docker_daemon_available(), reason="docker daemon not available")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_TAG = "lead-generator-smoke-test:latest"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def running_container():
    subprocess.run(["docker", "build", "-t", IMAGE_TAG, REPO_ROOT], check=True, capture_output=True)
    port = _free_port()
    container_id = subprocess.run(
        [
            "docker", "run", "-d", "-p", f"{port}:8000",
            "-e", "LEAD_PROVIDERS=fake", "-e", "ENV=test",
            IMAGE_TAG,
        ],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    try:
        for _ in range(30):
            try:
                if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                    break
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        else:
            pytest.fail("container never became healthy")
        yield port
    finally:
        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True)


def test_container_health_check_passes(running_container):
    port = running_container
    response = requests.get(f"http://127.0.0.1:{port}/health", timeout=5)
    assert response.status_code == 200


def test_container_completes_fake_provider_flow(running_container):
    port = running_container
    wav_path = os.path.join(REPO_ROOT, "tests", "fixtures", "sample.wav")
    with open(wav_path, "rb") as f:
        response = requests.post(
            f"http://127.0.0.1:{port}/upload-audio",
            files={"file": ("sample.wav", f, "audio/wav")},
            timeout=10,
        )
    assert response.status_code == 200
    assert "lead" in response.json()
