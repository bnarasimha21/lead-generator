"""Unit tests for evals/score.py metrics, including the hallucination
count and the (documented: none) partial-credit rule, with hand-built
inputs independent of the golden set.
"""

from evals.score import aggregate, score_case


def _field(value=None, evidence=None):
    return {"value": value, "evidence": evidence, "rejected": False}


def test_exact_match_is_correct():
    expected = {"name": _field("Priya", "Priya")}
    actual = {"name": _field("Priya", "Priya")}
    score = score_case("c1", "Hi Priya here", expected, actual)
    assert score.field_correct >= 1
    assert "name" not in score.wrong_fields


def test_case_insensitive_whitespace_normalised_match_counts_as_correct():
    expected = {"company": _field("Acme Corp", "Acme Corp")}
    actual = {"company": _field("  acme   corp ", "Acme Corp")}
    score = score_case("c2", "from Acme Corp", expected, actual)
    assert "company" not in score.wrong_fields


def test_near_miss_is_not_partial_credit():
    """Deliberately close but not equal: the documented rule is exact
    match only, no partial credit."""
    expected = {"company": _field("Acme Corporation", "Acme Corporation")}
    actual = {"company": _field("Acme Corp", "Acme Corp")}
    score = score_case("c3", "from Acme Corporation", expected, actual)
    assert "company" in score.wrong_fields


def test_hallucinated_field_counted_separately_from_wrong():
    expected = {"company": _field(None, None)}
    actual = {"company": _field("Globex", "Globex")}
    score = score_case("c4", "no company mentioned", expected, actual)
    assert "company" in score.hallucinated_fields
    assert "company" not in score.wrong_fields


def test_correct_null_counts_as_correct():
    # score_case always scores every field in app.schema.FIELD_NAMES; a
    # field missing from the expected/actual dicts is treated as null on
    # both sides (correct). Here only "email" is specified explicitly, and
    # it being correctly null contributes to the same all-fields-correct
    # count as every other, unspecified field.
    from app.schema import FIELD_NAMES

    expected = {"email": _field(None, None)}
    actual = {"email": _field(None, None)}
    score = score_case("c5", "no email given", expected, actual)
    assert "email" not in score.wrong_fields
    assert "email" not in score.hallucinated_fields
    assert score.field_correct == len(FIELD_NAMES)


def test_evidence_validity_rate_catches_invalid_evidence():
    expected = {"name": _field("Sam", "Sam")}
    actual = {"name": _field("Sam", "this quote is not in the transcript")}
    score = score_case("c6", "Hi, this is Sam", expected, actual)
    assert score.evidence_checked == 1
    assert score.evidence_valid == 0


def test_aggregate_field_precision_recall():
    expected_a = {"name": _field("Sam", "Sam")}
    actual_a = {"name": _field("Sam", "Sam")}
    expected_b = {"name": _field("Lee", "Lee")}
    actual_b = {"name": _field(None, None)}  # missed a real field: hurts recall
    expected_c = {"name": _field(None, None)}
    actual_c = {"name": _field("Made Up", "Made Up")}  # hallucination: hurts precision

    score_a = score_case("a", "Sam here", expected_a, actual_a)
    score_b = score_case("b", "Lee here", expected_b, actual_b)
    score_c = score_case("c", "nothing here", expected_c, actual_c)

    report = aggregate([score_a, score_b, score_c])
    assert report.total_hallucinated == 1
    assert report.field_precision < 1.0
    assert report.field_recall < 1.0


def test_word_error_rate_is_none_without_jiwer_or_audio():
    from evals.score import word_error_rate

    result = word_error_rate("hello world", "hello world")
    # jiwer is a dev dependency; if present this is 0.0, if absent None.
    assert result is None or result == 0.0
