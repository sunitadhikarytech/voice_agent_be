"""VA-38 — tool / function-calling framework."""
import asyncio

import pytest
from pydantic import BaseModel, Field

from app.tools import (
    Tool,
    ToolRegistry,
    ToolResult,
    ToolValidationError,
    UnknownTool,
)


class AddParams(BaseModel):
    a: int
    b: int


async def _add(p: AddParams) -> int:
    return p.a + p.b


def _registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool("add", "Add two integers.", AddParams, _add))
    return reg


# --- declarations (what the model sees) -------------------------------------------------

def test_declaration_exposes_name_description_and_typed_params():
    (decl,) = _registry().declarations()
    assert decl["name"] == "add"
    assert decl["description"] == "Add two integers."
    props = decl["parameters"]["properties"]
    assert props["a"]["type"] == "integer" and props["b"]["type"] == "integer"


# --- dispatch ---------------------------------------------------------------------------

def test_registered_tool_is_invoked_and_result_returned():
    result = asyncio.run(_registry().dispatch("add", {"a": 2, "b": 3}))
    assert isinstance(result, ToolResult)
    assert result.name == "add"
    assert result.content == 5


def test_unknown_tool_raises():
    with pytest.raises(UnknownTool) as ei:
        asyncio.run(_registry().dispatch("subtract", {"a": 1, "b": 2}))
    assert "subtract" in str(ei.value) and "add" in str(ei.value)  # lists available


def test_invalid_arguments_are_rejected():
    with pytest.raises(ToolValidationError):
        asyncio.run(_registry().dispatch("add", {"a": "not-an-int", "b": 3}))


def test_missing_argument_is_rejected():
    with pytest.raises(ToolValidationError):
        asyncio.run(_registry().dispatch("add", {"a": 1}))


# --- registration -----------------------------------------------------------------------

def test_decorator_registration():
    reg = ToolRegistry()

    class EchoParams(BaseModel):
        text: str = Field(min_length=1)

    @reg.tool("echo", "Echo the text.", EchoParams)
    async def _echo(p: EchoParams) -> str:
        return p.text

    assert "echo" in reg
    assert reg.names() == ["echo"]
    assert asyncio.run(reg.dispatch("echo", {"text": "hi"})).content == "hi"


def test_duplicate_registration_raises():
    reg = _registry()
    with pytest.raises(Exception):
        reg.register(Tool("add", "dup", AddParams, _add))
