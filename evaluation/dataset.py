"""The evaluation seed set (VA-65).

The questions are grounded in the configured source document (the project ships against the
Constitution of India). ``expected_keywords`` are the terms a correct, document-grounded
answer must contain — kept lowercase and phrase-level so the scorer stays simple and robust.
"""
from __future__ import annotations

import json
from pathlib import Path

from evaluation.harness import EvalCase

SEED_PATH = Path(__file__).with_name("seed_set.json")


def load_seed_set(path: Path | None = None) -> list[EvalCase]:
    """Load the seed set from JSON into typed :class:`EvalCase` objects."""
    raw = json.loads((path or SEED_PATH).read_text(encoding="utf-8"))
    return [
        EvalCase(
            id=item["id"],
            question=item["question"],
            expected_keywords=tuple(item["expected_keywords"]),
            note=item.get("note", ""),
        )
        for item in raw
    ]
