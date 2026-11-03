"""Scoring for the golden-set eval (issue 4).

Partial-credit rule, written down because the draft asked for one: there
is none. A field counts as a match only if, after normalising whitespace
and case, the extracted value equals the gold value exactly. This is
deliberately strict: "roughly right" on a contact field (an email, a
phone number) is as useless to a salesperson as wrong, and exact match is
simple enough to audit by eye. If this proves too strict in practice, loosen
it in one place (`_values_match`) rather than scattering fuzz logic.

Metrics:
- field precision / recall over "is this field non-null": precision counts
  how many of the model's non-null answers were actually right (right
  field, right value, or correctly null when gold is null); recall counts
  how many of the gold's non-null fields the model actually produced.
- hallucinated_fields: count of fields where the model gave a non-null
  value but gold says null. This is reported separately from "wrong value"
  because inventing a field that was never there is the specific failure
  mode this project cares about catching.
- evidence_validity_rate: fraction of the model's non-null fields whose
  evidence is genuinely a substring of the transcript (reuses
  app.verify.quote_in_transcript, the same check the live app runs).
- word_error_rate: via jiwer, only computed when both a reference and a
  hypothesis transcript are supplied (the audio half of the golden set);
  this repo currently ships text-only golden cases (see
  evals/golden/README.md), so WER is reported as None until audio clips
  are added.
"""

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional

from app.schema import FIELD_NAMES
from app.verify import quote_in_transcript

try:
    import jiwer
except ImportError:  # pragma: no cover
    jiwer = None


def _normalise(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return " ".join(value.split()).casefold()


def _values_match(expected: Optional[str], actual: Optional[str]) -> bool:
    return _normalise(expected) == _normalise(actual)


@dataclass
class CaseScore:
    case_id: str
    field_correct: int = 0
    field_total_gold_nonnull: int = 0
    field_total_actual_nonnull: int = 0
    correct_nonnull_fields: List[str] = dc_field(default_factory=list)
    hallucinated_fields: List[str] = dc_field(default_factory=list)
    evidence_valid: int = 0
    evidence_checked: int = 0
    wrong_fields: List[str] = dc_field(default_factory=list)


def score_case(case_id: str, transcript: str, expected: Dict[str, dict], actual: Dict[str, dict]) -> CaseScore:
    score = CaseScore(case_id=case_id)
    for name in FIELD_NAMES:
        gold = expected.get(name, {}) or {}
        got = actual.get(name, {}) or {}
        gold_value = gold.get("value")
        actual_value = got.get("value")

        if gold_value is not None:
            score.field_total_gold_nonnull += 1
        if actual_value is not None:
            score.field_total_actual_nonnull += 1
            evidence = got.get("evidence")
            score.evidence_checked += 1
            if evidence and quote_in_transcript(evidence, transcript):
                score.evidence_valid += 1

        if gold_value is None and actual_value is not None:
            score.hallucinated_fields.append(name)
            continue

        if _values_match(gold_value, actual_value):
            score.field_correct += 1
            if gold_value is not None:
                score.correct_nonnull_fields.append(name)
        else:
            score.wrong_fields.append(name)

    return score


@dataclass
class AggregateReport:
    num_cases: int = 0
    total_fields_correct: int = 0
    total_fields_checked: int = 0
    total_gold_nonnull: int = 0
    total_actual_nonnull: int = 0
    total_correct_nonnull: int = 0
    total_hallucinated: int = 0
    total_evidence_valid: int = 0
    total_evidence_checked: int = 0
    wer: Optional[float] = None
    case_scores: List[CaseScore] = dc_field(default_factory=list)

    @property
    def field_accuracy(self) -> float:
        return self.total_fields_correct / self.total_fields_checked if self.total_fields_checked else 0.0

    @property
    def field_precision(self) -> float:
        return self.total_correct_nonnull / self.total_actual_nonnull if self.total_actual_nonnull else 1.0

    @property
    def field_recall(self) -> float:
        return self.total_correct_nonnull / self.total_gold_nonnull if self.total_gold_nonnull else 1.0

    @property
    def evidence_validity_rate(self) -> float:
        return self.total_evidence_valid / self.total_evidence_checked if self.total_evidence_checked else 1.0

    def as_table(self) -> str:
        lines = [
            "| metric | value |",
            "|---|---|",
            f"| cases | {self.num_cases} |",
            f"| field accuracy | {self.field_accuracy:.3f} |",
            f"| field precision | {self.field_precision:.3f} |",
            f"| field recall | {self.field_recall:.3f} |",
            f"| hallucinated fields | {self.total_hallucinated} |",
            f"| evidence validity rate | {self.evidence_validity_rate:.3f} |",
            f"| word error rate | {self.wer if self.wer is not None else 'n/a (no audio clips scored)'} |",
        ]
        return "\n".join(lines)


def aggregate(case_scores: List[CaseScore], wer: Optional[float] = None) -> AggregateReport:
    report = AggregateReport(num_cases=len(case_scores), wer=wer, case_scores=case_scores)
    for score in case_scores:
        report.total_fields_correct += score.field_correct
        report.total_fields_checked += len(FIELD_NAMES)
        report.total_gold_nonnull += score.field_total_gold_nonnull
        report.total_actual_nonnull += score.field_total_actual_nonnull
        report.total_hallucinated += len(score.hallucinated_fields)
        report.total_evidence_valid += score.evidence_valid
        report.total_evidence_checked += score.evidence_checked
        report.total_correct_nonnull += len(score.correct_nonnull_fields)
    return report


def word_error_rate(reference: str, hypothesis: str) -> Optional[float]:
    if jiwer is None:
        return None
    return jiwer.wer(reference, hypothesis)
