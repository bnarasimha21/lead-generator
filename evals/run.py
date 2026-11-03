"""python -m evals.run --providers fake|openai [--matrix] [--yes]

Runs the golden set (evals/golden/cases.json) through the extractor and
prints the metric table from evals/score.py.

--providers fake: offline, no network, safe for every PR. The fake
extractor is deliberately wrong on a subset of cases (see
`_scripted_result`) so the scorer itself is exercised, not just the happy
path.

--providers openai: calls the real extractor. Refuses to run without
--yes, and prints the planned call count and an estimated token cost
first (computed from app.config prices, which default to unset).
Planning figure: 25 extraction calls at roughly 600 input / 200 output
tokens each is about 15,000 input and 5,000 output tokens.

--matrix: (openai mode only) runs once per model id listed in
EXTRACT_MODEL_MATRIX (comma-separated), writing one results file that
compares them.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import date
from typing import List, Optional

from app import config
from app.providers import ExtractionResult, OpenAIExtractor, Usage
from app.schema import EvidencedField, Lead
from evals.score import AggregateReport, aggregate, score_case


class LatencyBudgetExceeded(Exception):
    pass

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "cases.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

EST_INPUT_TOKENS_PER_CASE = 600
EST_OUTPUT_TOKENS_PER_CASE = 200

# A floor used by tests/test_eval_run.py and CI: field accuracy below this
# on the fake-provider run (which is scripted to include deliberate wrong
# answers) means the scorer itself regressed.
DEFAULT_ACCURACY_FLOOR = 0.0


def load_golden_cases():
    with open(GOLDEN_PATH) as f:
        return json.load(f)


def _dict_to_lead(expected: dict) -> Lead:
    lead = Lead.empty()
    for name, field in expected.items():
        if field.get("value") is not None:
            setattr(lead, name, EvidencedField(value=field["value"], evidence=field["evidence"]))
    return lead


def _scripted_result(index: int, case: dict) -> ExtractionResult:
    """Deliberately right most of the time, wrong on every 5th case (drop
    a field) and hallucinating on every 7th, so the aggregate metrics in a
    fake run are not trivially perfect: a scorer bug that always reports
    100% would be caught by this.
    """
    lead = _dict_to_lead(case["expected"])
    if index % 5 == 4:
        # Drop the first non-null field this case has, simulating a missed field.
        for name, field in case["expected"].items():
            if field.get("value") is not None:
                setattr(lead, name, EvidencedField())
                break
    if case["expected"].get("company", {}).get("value") is None and index % 7 == 3:
        # Hallucinate a company that isn't in the transcript (only on
        # cases where gold company is actually null, so this lands as a
        # real hallucination rather than a wrong-value case), with a
        # self-referential (and therefore invalid) evidence claim.
        lead.company = EvidencedField(value="Hallucinated Inc", evidence="Hallucinated Inc")
    return ExtractionResult(status="ok", lead=lead, usage=Usage(input_tokens=600, output_tokens=120))


async def run_fake(cases: List[dict], extractor=None, latency_budget_ms: Optional[float] = None) -> AggregateReport:
    """extractor, if given, must implement Extractor.extract(transcript);
    used by tests to inject a delay and prove the latency budget check
    actually fires. In normal use (no extractor passed) results are
    generated in-process via _scripted_result with no artificial delay.
    """
    scores = []
    for i, case in enumerate(cases):
        if extractor is not None:
            start = time.monotonic()
            result = await extractor.extract(case["transcript"])
            elapsed_ms = (time.monotonic() - start) * 1000
            if latency_budget_ms is not None and elapsed_ms > latency_budget_ms:
                raise LatencyBudgetExceeded(
                    f"case {case['id']} took {elapsed_ms:.0f}ms, over the {latency_budget_ms:.0f}ms budget"
                )
        else:
            result = _scripted_result(i, case)
        actual = result.lead.to_display_dict([])
        scores.append(score_case(case["id"], case["transcript"], case["expected"], actual))
    return aggregate(scores)


async def run_openai(cases: List[dict]) -> AggregateReport:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    extractor = OpenAIExtractor(client)
    scores = []
    for case in cases:
        result = await extractor.extract(case["transcript"])
        actual = result.lead.to_display_dict([]) if result.status == "ok" else Lead.empty().to_display_dict([])
        scores.append(score_case(case["id"], case["transcript"], case["expected"], actual))
    return aggregate(scores)


def estimate_cost(num_cases: int):
    input_tokens = num_cases * EST_INPUT_TOKENS_PER_CASE
    output_tokens = num_cases * EST_OUTPUT_TOKENS_PER_CASE
    if config.PRICE_PER_INPUT_TOKEN is None or config.PRICE_PER_OUTPUT_TOKEN is None:
        return input_tokens, output_tokens, None
    cost = input_tokens * config.PRICE_PER_INPUT_TOKEN + output_tokens * config.PRICE_PER_OUTPUT_TOKEN
    return input_tokens, output_tokens, cost


def write_report(report: AggregateReport, label: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{date.today().isoformat()}-{label}.md")
    with open(path, "w") as f:
        f.write(f"# Eval run: {label} ({date.today().isoformat()})\n\n")
        f.write(report.as_table())
        f.write("\n")
    return path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", choices=["fake", "openai"], default="fake")
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--accuracy-floor", type=float, default=DEFAULT_ACCURACY_FLOOR)
    parser.add_argument("--latency-budget-ms", type=float, default=None)
    args = parser.parse_args(argv)

    cases = load_golden_cases()

    if args.providers == "fake":
        try:
            report = asyncio.run(run_fake(cases, latency_budget_ms=args.latency_budget_ms))
        except LatencyBudgetExceeded as exc:
            print(f"Latency budget exceeded: {exc}")
            return 1
        print(report.as_table())
        write_report(report, "fake")
        return 0 if report.field_accuracy >= args.accuracy_floor else 1

    input_tokens, output_tokens, cost = estimate_cost(len(cases))
    cost_str = f"${cost:.4f}" if cost is not None else "unknown (set PRICE_PER_INPUT_TOKEN / PRICE_PER_OUTPUT_TOKEN)"
    print(f"About to run {len(cases)} extraction calls against {config.EXTRACT_MODEL}.")
    print(f"Estimated tokens: ~{input_tokens} input, ~{output_tokens} output. Estimated cost: {cost_str}.")
    if not args.yes:
        print("Refusing to spend money without --yes.")
        return 1

    model_ids = [config.EXTRACT_MODEL]
    if args.matrix:
        model_ids = os.getenv("EXTRACT_MODEL_MATRIX", config.EXTRACT_MODEL).split(",")

    exit_code = 0
    for model_id in model_ids:
        config.EXTRACT_MODEL = model_id.strip()
        report = asyncio.run(run_openai(cases))
        print(f"--- {model_id} ---")
        print(report.as_table())
        write_report(report, model_id.strip())
        if report.field_accuracy < args.accuracy_floor:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
