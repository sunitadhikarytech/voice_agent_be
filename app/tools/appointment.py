"""book_appointment demo tool (VA-39).

The first concrete tool, demonstrating the end-to-end tool loop: the model collects name/day/
time, the registry validates them and runs this handler, and a confirmation id is returned to
be read back out loud. The backend booking is stubbed for the spike — no real calendar write.
"""
from __future__ import annotations

import uuid
from typing import Any, Callable

from pydantic import BaseModel, Field

from app.tools.registry import Tool, ToolRegistry


class BookAppointmentParams(BaseModel):
    """Typed parameters the model must supply to book an appointment."""

    name: str = Field(min_length=1, description="Who the appointment is for.")
    day: str = Field(min_length=1, description="Day of the appointment, e.g. 'Tuesday' or '2026-07-20'.")
    time: str = Field(min_length=1, description="Time of the appointment, e.g. '3pm'.")


def _default_confirmation_id() -> str:
    return "appt-" + uuid.uuid4().hex[:8]


def make_book_appointment_tool(
    id_factory: Callable[[], str] = _default_confirmation_id,
) -> Tool:
    """Build the book_appointment tool. ``id_factory`` is injectable for deterministic tests."""

    async def _book(params: BookAppointmentParams) -> dict[str, Any]:
        # Stub: a real implementation would write to a calendar/CRM here.
        return {
            "confirmation_id": id_factory(),
            "name": params.name,
            "day": params.day,
            "time": params.time,
            "status": "confirmed",
        }

    return Tool(
        name="book_appointment",
        description=(
            "Book an appointment for the given person, day, and time. "
            "Returns a confirmation id to read back to the caller."
        ),
        params_model=BookAppointmentParams,
        handler=_book,
    )


def register_default_tools(
    registry: ToolRegistry, *, id_factory: Callable[[], str] = _default_confirmation_id
) -> None:
    """Register the built-in demo tools into ``registry`` (used by the pipeline in VA-45)."""
    registry.register(make_book_appointment_tool(id_factory=id_factory))


def default_registry(id_factory: Callable[[], str] = _default_confirmation_id) -> ToolRegistry:
    """A registry pre-loaded with the built-in demo tools."""
    registry = ToolRegistry()
    register_default_tools(registry, id_factory=id_factory)
    return registry
