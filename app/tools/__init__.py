"""Tool / function-calling registry.

Holds the typed tool registry the model can call during a turn (built in VA-38; the first
concrete tool is added in VA-39, and VA-45 wires it into the LLM turn).
"""
from app.tools.appointment import (
    BookAppointmentParams,
    default_registry,
    make_book_appointment_tool,
    register_default_tools,
)
from app.tools.registry import (
    Tool,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolValidationError,
    UnknownTool,
)

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "ToolError",
    "ToolValidationError",
    "UnknownTool",
    "BookAppointmentParams",
    "make_book_appointment_tool",
    "register_default_tools",
    "default_registry",
]
