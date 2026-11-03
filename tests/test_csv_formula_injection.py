"""Adversarial-review finding 12: app/static/recorder.js's escapeCsvCell()
guards CSV export against formula injection, but only covered a leading
=, +, -, or @; OWASP's documented character set for this guard also
includes a leading tab (0x09) and carriage return (0x0D), since Excel
treats either as leading whitespace and still evaluates the formula that
follows it.

These tests run the real function with Node (no reimplementation in
Python), so they exercise the actual browser-side code, not a port of
it. Skipped if Node isn't available rather than failing the build.
"""

import json
import os
import shutil
import subprocess

import pytest

RECORDER_PATH = os.path.join(os.path.dirname(__file__), "..", "app", "static", "recorder.js")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node is not available to execute recorder.js")


def _extract_function_source() -> str:
    with open(RECORDER_PATH) as f:
        source = f.read()
    start = source.index("function escapeCsvCell")
    end = source.index("\n}\n", start) + len("\n}")
    return source[start:end]


def _run_cases(cases):
    """Feed [input, ...] cases to the real escapeCsvCell via Node and
    return the list of escaped outputs."""
    fn_source = _extract_function_source()
    script = fn_source + "\nconst cases = JSON.parse(process.argv[1]);\nconsole.log(JSON.stringify(cases.map(escapeCsvCell)));\n"
    result = subprocess.run(
        ["node", "-e", script, json.dumps(cases)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_leading_equals_plus_minus_at_are_neutralised():
    cases = ["=1+1", "+1+1", "-1+1", "@SUM(A1)"]
    outputs = _run_cases(cases)
    for original, escaped in zip(cases, outputs):
        assert escaped == f"\"'{original}\""


def test_leading_tab_is_neutralised():
    """OWASP's documented set includes a leading tab: Excel still treats
    it as leading whitespace and evaluates the formula after it."""
    payload = "\t=1+1"
    (escaped,) = _run_cases([payload])
    assert escaped == "\"'\t=1+1\""


def test_leading_carriage_return_is_neutralised():
    payload = "\r=1+1"
    (escaped,) = _run_cases([payload])
    assert escaped == "\"'\r=1+1\""


def test_ordinary_text_is_left_unprefixed():
    (escaped,) = _run_cases(["Priya Sharma"])
    assert escaped == '"Priya Sharma"'


def test_embedded_quotes_are_doubled_per_csv_rules():
    (escaped,) = _run_cases(['Say "hi"'])
    assert escaped == '"Say ""hi"""'


def test_null_and_undefined_become_empty_quoted_cell():
    outputs = _run_cases([None])
    assert outputs == ['""']
