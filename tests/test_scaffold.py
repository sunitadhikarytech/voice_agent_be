"""VA-01 scaffold tests: the expected module layout exists and imports cleanly."""
import importlib

import pytest

MODULES = [
    "app.main",
    "app.config",
    "app.dispatch",
    "app.providers.base",
    "app.pipelines.base",
    "app.pipelines.traditional",
    "app.pipelines.realtime",
    "app.context",
    "app.tools",
    "app.streaming",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module):
    assert importlib.import_module(module) is not None


def test_dispatch_vocabulary():
    from app.dispatch import Architecture, Delivery

    assert {a.value for a in Architecture} == {"traditional", "realtime"}
    assert {d.value for d in Delivery} == {"complete", "stream"}
