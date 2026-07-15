"""Tool / function-calling framework (VA-38).

A registry of tools with **typed parameters** (pydantic). Each tool exposes a function
declaration for the model (Gemini/ADK function-calling); when the model chooses a tool, the
registry validates the arguments against the tool's schema, runs the handler, and returns a
``ToolResult`` to feed back into the turn.

VA-39 registers the first concrete tool (book_appointment); VA-45 wires the registry's
declarations into the LLM and dispatches the model's tool calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, TypeVar

from pydantic import BaseModel, ValidationError

P = TypeVar("P", bound=BaseModel)


class ToolError(RuntimeError):
    """Base class for tool errors."""


class UnknownTool(ToolError):
    """Raised when the model calls a tool that isn't registered."""


class ToolValidationError(ToolError):
    """Raised when tool-call arguments don't satisfy the tool's typed parameters."""


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of a tool call, fed back into the conversation."""

    name: str
    content: Any


@dataclass(frozen=True, slots=True)
class Tool(Generic[P]):
    """A callable tool: a name, a description, a typed parameter model, and an async handler."""

    name: str
    description: str
    params_model: type[P]
    handler: Callable[[P], Awaitable[Any]]

    def declaration(self) -> dict[str, Any]:
        """The function declaration the model sees (name / description / JSON-schema params)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.params_model.model_json_schema(),
        }


class ToolRegistry:
    """Holds the available tools and dispatches the model's tool calls."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ToolError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def tool(
        self, name: str, description: str, params_model: type[P]
    ) -> Callable[[Callable[[P], Awaitable[Any]]], Callable[[P], Awaitable[Any]]]:
        """Decorator: register the decorated async handler as a tool."""

        def decorator(handler: Callable[[P], Awaitable[Any]]) -> Callable[[P], Awaitable[Any]]:
            self.register(Tool(name, description, params_model, handler))
            return handler

        return decorator

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools)

    def declarations(self) -> list[dict[str, Any]]:
        """Function declarations for every tool (passed to the LLM as its tool list)."""
        return [tool.declaration() for tool in self._tools.values()]

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Validate ``arguments`` against the tool's parameters, run it, and return the result.

        Raises :class:`UnknownTool` for an unregistered name and :class:`ToolValidationError`
        for arguments that don't satisfy the typed parameters.
        """
        try:
            tool = self._tools[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._tools)) or "(none)"
            raise UnknownTool(f"unknown tool '{name}'; registered: {available}") from exc

        try:
            params = tool.params_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolValidationError(f"invalid arguments for tool '{name}': {exc}") from exc

        result = await tool.handler(params)
        return ToolResult(name=name, content=result)
