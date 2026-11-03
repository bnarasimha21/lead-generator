"""Issue 4 acceptance criteria: the eval runner works offline in CI and
produces every metric; a configured accuracy floor fails the run with a
non-zero exit code when a metric falls below it.
"""

from evals.run import main


def test_fake_eval_run_prints_every_metric(capsys):
    exit_code = main(["--providers", "fake"])
    out = capsys.readouterr().out
    for metric in ("field accuracy", "field precision", "field recall", "hallucinated fields", "evidence validity rate", "word error rate"):
        assert metric in out
    assert exit_code == 0  # default floor is 0.0, the scripted-wrong fake run still passes it


def test_fake_eval_run_fails_when_floor_is_unreachable(capsys):
    exit_code = main(["--providers", "fake", "--accuracy-floor", "1.01"])
    assert exit_code == 1


def test_openai_eval_run_refuses_without_yes(capsys):
    exit_code = main(["--providers", "openai"])
    out = capsys.readouterr().out
    assert "Refusing to spend money without --yes" in out
    assert exit_code == 1
