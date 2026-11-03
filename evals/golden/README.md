# Golden set

`cases.json` has 25 hand-written text transcripts with exact expected
`Lead` JSON, covering: full info, no name given, two people mentioned (only
the caller's name should be extracted), nothing useful, a spoken email in
several forms (at/dot/underscore/dash/plus), a phone number, a role with
no name, a budget said as words, no company, an urgent outage, dense
multi-field cases, and low-signal/rambling notes. Every expected
`evidence` string is checked by a test (`tests/test_golden_set.py`) to be
an exact substring of its transcript, normalising only whitespace and
case.

## Audio clips: not included

The issue draft asked for 8 to 10 WAV clips rendered from the same notes
via a redistributable local TTS, to measure the transcriber's word error
rate. That part is not done in this change: this environment has no local
TTS tool available, and a text-to-speech call is itself a provider call
this project is trying to keep offline-by-default. `evals/score.py`
implements `word_error_rate()` via `jiwer` and is ready to score clips the
moment they exist; `evals/run.py` reports `word error rate: n/a (no audio
clips scored)` until then.

To add them later: record or synthesize short WAV/MP3 reads of a subset of
`cases.json`'s transcripts, name them `<case_id>.wav`, note the TTS engine
and its license here, and wire a loop into `evals/run.py` that calls the
real transcriber on each clip and compares against the case's `transcript`
field via `jiwer.wer`.

## Your own recordings

A `local/` folder (gitignored) is the place for your own real voice note
recordings if you want to eyeball accuracy on something less clean than
synthetic audio. Nothing in `local/` is committed or read by CI.
