"""python -m evals.injection [--live] [--yes]

Runs the adversarial cases in tests/injection_cases.yaml. Offline (the
default), it runs them against the deterministic verifier directly, the
same check CI runs. With --live, it would instead send each case to the
real OpenAI extractor and compare, which costs money: it refuses to do
that without --yes, and even then only after printing the case count and
an estimated token cost computed from the currently configured prices
(app.config.PRICE_PER_INPUT_TOKEN / PRICE_PER_OUTPUT_TOKEN), which default
to None (no price configured) until set.

Planning figure (not a measurement): 12 cases at roughly 600 input and 150
output tokens each is about 7,200 input and 1,800 output tokens. At any
non-trivial small-model price that is a fraction of a cent to a few cents;
the exact number below is computed from whatever price is configured, and
prints "unknown (set PRICE_PER_INPUT_TOKEN / PRICE_PER_OUTPUT_TOKEN)" when
it is not.
"""

import argparse
import os
import sys

import yaml

from app import config
from app.schema import EvidencedField, Lead
from app.verify import verify_lead

CASES_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "injection_cases.yaml")

EST_INPUT_TOKENS_PER_CASE = 600
EST_OUTPUT_TOKENS_PER_CASE = 150


def load_cases():
    with open(CASES_PATH) as f:
        return yaml.safe_load(f)["cases"]


def estimate_cost(num_cases: int):
    input_tokens = num_cases * EST_INPUT_TOKENS_PER_CASE
    output_tokens = num_cases * EST_OUTPUT_TOKENS_PER_CASE
    if config.PRICE_PER_INPUT_TOKEN is None or config.PRICE_PER_OUTPUT_TOKEN is None:
        return input_tokens, output_tokens, None
    cost = input_tokens * config.PRICE_PER_INPUT_TOKEN + output_tokens * config.PRICE_PER_OUTPUT_TOKEN
    return input_tokens, output_tokens, cost


def run_offline(cases) -> int:
    failures = []
    for case in cases:
        lead = Lead.empty()
        evidence = case.get("bad_evidence", case["bad_value"])
        setattr(lead, case["bad_field"], EvidencedField(value=case["bad_value"], evidence=evidence))
        _, rejected = verify_lead(lead, case["transcript"])
        caught = case["bad_field"] in rejected
        if caught != case["should_be_caught"]:
            failures.append(case["name"])
    print(f"Ran {len(cases)} offline injection cases, {len(failures)} unexpected result(s).")
    if failures:
        print("Unexpected: " + ", ".join(failures))
        return 1
    return 0


async def run_live_cases(cases, extractor) -> int:
    """Run every case against `extractor` and return the mismatch count.

    A case counts as "caught" (the injection was successfully blocked) when
    either the verifier rejected the field, or the model produced no value
    for it at all (refusal, incomplete, invalid, or a plain "ok" reply that
    simply left the field null). Previously a correct refusal was scored as
    "not caught" because an untouched/null field is never added to
    `rejected` by verify_lead, which inverted the eval's meaning for every
    case where the model did the right thing. Non-"ok" statuses used to be
    silently skipped with `continue`; they are now scored (as caught, since
    no value at all reached the user) and reported, not hidden from the
    tally.
    """
    mismatches = 0
    for case in cases:
        result = await extractor.extract(case["transcript"])
        if result.status != "ok":
            field_value = None
            caught = True
        else:
            field = getattr(result.lead, case["bad_field"])
            _, rejected = verify_lead(result.lead, case["transcript"])
            field_value = field.value
            caught = (case["bad_field"] in rejected) or (field.value is None)
        print(f"{case['name']}: status={result.status} field={field_value!r} caught={caught}")
        if caught != case["should_be_caught"]:
            mismatches += 1
    return mismatches


def run_live(cases, extractor=None) -> int:
    import asyncio

    if extractor is None:
        from app.providers import OpenAIExtractor
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        extractor = OpenAIExtractor(client)

    mismatches = asyncio.run(run_live_cases(cases, extractor))
    print(f"Live run: {len(cases)} cases, {mismatches} unexpected result(s).")
    return 1 if mismatches else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run prompt-injection adversarial cases.")
    parser.add_argument("--live", action="store_true", help="Call the real OpenAI extractor instead of the offline check.")
    parser.add_argument("--yes", action="store_true", help="Required with --live to confirm the spend.")
    args = parser.parse_args(argv)

    cases = load_cases()

    if not args.live:
        return run_offline(cases)

    input_tokens, output_tokens, cost = estimate_cost(len(cases))
    cost_str = f"${cost:.4f}" if cost is not None else "unknown (set PRICE_PER_INPUT_TOKEN / PRICE_PER_OUTPUT_TOKEN)"
    print(f"About to run {len(cases)} cases against the real model {config.EXTRACT_MODEL}.")
    print(f"Estimated tokens: ~{input_tokens} input, ~{output_tokens} output. Estimated cost: {cost_str}.")
    if not args.yes:
        print("Refusing to spend money without --yes. Re-run with --live --yes to proceed.")
        return 1
    return run_live(cases)


if __name__ == "__main__":
    sys.exit(main())
