"""VA-62 (QA-01) — unit coverage for the dispatch registry and provider factory seams.

The broader suite exercises dispatch, adapters and tools through end-to-end turns; this closes
the small remaining unit gaps: the registry membership check + lookup error, the ``register_*``
extension helpers, and the unknown-provider error path.
"""
from __future__ import annotations

import pytest

from app.dispatch import Architecture, PipelineNotRegistered, PipelineRegistry
from app.providers import factory
from app.providers.factory import UnknownProvider, _build


def _dummy_ctor(_settings):
    return "adapter"


def test_pipeline_registry_membership_and_lookup_error():
    registry = PipelineRegistry()
    assert Architecture.TRADITIONAL not in registry  # nothing registered yet
    with pytest.raises(PipelineNotRegistered):
        registry.get(Architecture.TRADITIONAL)


def test_build_resolves_registered_and_raises_on_unknown():
    assert _build({"demo": _dummy_ctor}, "demo", "stt", None) == "adapter"
    with pytest.raises(UnknownProvider) as exc:
        _build({}, "nope", "llm", None)
    assert "unknown llm provider 'nope'" in str(exc.value)
    assert "registered: (none)" in str(exc.value)


@pytest.mark.parametrize(
    ("register", "table"),
    [
        (factory.register_stt, factory._STT),
        (factory.register_llm, factory._LLM),
        (factory.register_tts, factory._TTS),
        (factory.register_realtime, factory._RT),
    ],
)
def test_register_helpers_add_to_their_table(register, table):
    key = "unit-test-fake"
    register(key, _dummy_ctor)
    try:
        assert table[key] is _dummy_ctor
    finally:
        table.pop(key, None)
