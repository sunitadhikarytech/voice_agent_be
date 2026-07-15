# Tests

Fast, fully offline suite: every test runs against the **mock providers** — no network, no API
keys. Providers, pipelines and the app are exercised end to end through injected mocks.

## Run

```bash
pytest                                        # the whole suite
pytest tests/test_dispatch.py                 # one module
pytest --cov=app --cov-report=term-missing    # with coverage (VA-62)
```

Coverage is **~95%** of `app/`. The uncovered lines are almost entirely the live-network
branches of the provider adapters (real Deepgram/Gemini/Cartesia/OpenAI calls), which are
deliberately not unit-tested — the adapters are built to be driven by an injected transport so
tests never make live calls.

## Map (test file → area · ticket)

| Area | Test file | Ticket |
| --- | --- | --- |
| Service scaffold / liveness | `test_scaffold.py`, `test_healthz.py`, `test_app.py` | VA-01/06 |
| Config (typed, fail-fast) | `test_config.py` | VA-19 |
| Request + SSE schemas / contract | `test_streaming_schemas.py`, `test_contract_api.py` | VA-20/29 |
| Problem-shaped errors | `test_errors.py` | VA-28 |
| Dispatch (no router) | `test_dispatch.py`, `test_unit_coverage.py` | VA-21/62 |
| Provider interfaces + mocks | `test_providers.py` | VA-30 |
| STT / LLM / TTS / realtime adapters | `test_deepgram_stt.py`, `test_gemini_llm.py`, `test_cartesia_tts.py`, `test_openai_realtime.py` | VA-31/34/43/46 |
| Document context + grounding | `test_context.py`, `test_grounding.py` | VA-35/36/37 |
| Tools framework + booking | `test_tools.py`, `test_appointment.py` | VA-38/39 |
| Session / memory / state | `test_session.py`, `test_memory.py`, `test_state.py` | VA-40/41/42 |
| Pipelines | `test_traditional_pipeline.py`, `test_realtime_pipeline.py` | VA-45/48 |
| Voice endpoints + contract | `test_voice_endpoints.py`, `test_endpoint_contracts.py` | VA-23–27/63 |
| Integration (full turn per path) | `test_integration_turns.py` | VA-64 |
| Evaluation harness + grounding eval | `test_eval_harness.py`, `test_grounding_eval.py` | VA-65/66 |
| Endpoint smoke | `test_smoke_script.py` | VA-67 |
| Observability | `test_logging.py`, `test_metrics.py`, `test_usage.py`, `test_counters.py` | VA-57–60 |
| Reference dashboard | `test_frontend.py` | VA-51–56 |
