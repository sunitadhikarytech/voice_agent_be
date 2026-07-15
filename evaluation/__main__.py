"""``python -m evaluation`` — run the seed set and print an accuracy + latency report (VA-65).

    python -m evaluation                        # mock providers (offline harness smoke)
    python -m evaluation --min-accuracy 0.8      # exit non-zero below the threshold

Set the provider selection + keys (DEEPGRAM_/GOOGLE_/CARTESIA_* env) and a SOURCE_DOC_PATH to
measure real, document-grounded quality.
"""
from __future__ import annotations

import argparse
import asyncio
import sys

from evaluation.dataset import load_seed_set
from evaluation.grounding import evaluate_grounding
from evaluation.harness import evaluate
from evaluation.runners import AppTurnRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evaluation")
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="exit non-zero if accuracy is below this fraction (0..1)",
    )
    args = parser.parse_args(argv)

    cases = load_seed_set()
    runner = AppTurnRunner()
    report = asyncio.run(evaluate(cases, runner))
    print(report.format())

    # Grounding (VA-66) — only meaningful when a source document is loaded.
    document_text = runner.document_text
    if document_text:
        grounding = asyncio.run(evaluate_grounding(cases, runner, document_text))
        print()
        print(grounding.format())

    return 0 if report.accuracy >= args.min_accuracy else 1


if __name__ == "__main__":
    sys.exit(main())
