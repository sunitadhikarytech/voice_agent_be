# Evaluation harness

An offline harness that scores the assistant on a **seed set** for **accuracy** and
**latency** (VA-65), extended with a **document-grounding** metric in VA-66.

It is provider-agnostic: `evaluate(cases, runner, scorer)` runs each `EvalCase` through a
`TurnRunner` and scores the answer. Back the runner with the real app + real providers to
measure real quality, or with a stub in tests.

## Run

```bash
# offline smoke on the mock providers (answers won't match — this just proves it runs)
python -m evaluation

# measure real, document-grounded quality (set providers + keys + SOURCE_DOC_PATH first)
python -m evaluation --min-accuracy 0.8   # exit non-zero below the threshold
```

## Pieces
- `seed_set.json` — questions + `expected_keywords` a correct answer must contain
- `harness.py` — `EvalCase`, `RunResult`, `evaluate`, `keyword_scorer`, `EvalReport`
- `runners.py` — `AppTurnRunner` (drives `/voice/complete`, measures end-to-end latency)
- `__main__.py` — the `python -m evaluation` CLI
