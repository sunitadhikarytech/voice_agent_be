"""VA-39 — book_appointment demo tool."""
import asyncio

import pytest

from app.tools import ToolValidationError, default_registry, make_book_appointment_tool
from app.tools.registry import ToolRegistry


def test_declaration_exposes_name_day_time():
    (decl,) = default_registry().declarations()
    assert decl["name"] == "book_appointment"
    props = decl["parameters"]["properties"]
    assert set(props) == {"name", "day", "time"}


def test_booking_returns_confirmation_and_echoes_details():
    reg = default_registry(id_factory=lambda: "appt-TEST123")
    result = asyncio.run(
        reg.dispatch("book_appointment", {"name": "Ada", "day": "Tuesday", "time": "3pm"})
    )
    assert result.name == "book_appointment"
    assert result.content == {
        "confirmation_id": "appt-TEST123",
        "name": "Ada",
        "day": "Tuesday",
        "time": "3pm",
        "status": "confirmed",
    }


def test_default_confirmation_id_has_prefix_and_is_unique():
    reg = ToolRegistry()
    reg.register(make_book_appointment_tool())
    args = {"name": "Ada", "day": "Mon", "time": "9am"}
    a = asyncio.run(reg.dispatch("book_appointment", args)).content["confirmation_id"]
    b = asyncio.run(reg.dispatch("book_appointment", args)).content["confirmation_id"]
    assert a.startswith("appt-") and b.startswith("appt-")
    assert a != b  # each booking gets its own id


@pytest.mark.parametrize(
    "args",
    [
        {"name": "Ada", "day": "Tuesday"},          # missing time
        {"name": "", "day": "Tuesday", "time": "3pm"},  # empty name
        {"day": "Tuesday", "time": "3pm"},           # missing name
    ],
)
def test_incomplete_details_are_rejected(args):
    with pytest.raises(ToolValidationError):
        asyncio.run(default_registry().dispatch("book_appointment", args))
