"""Standardized error handling (VA-28).

Every error leaves the service as a consistent ``Problem`` JSON body carrying a correlation
id, and never a stack trace. A lightweight middleware assigns/propagates the correlation id
(``X-Request-ID``); structured logging and richer tracing build on this in VA-57.
"""
from __future__ import annotations

import logging
import uuid
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("app.errors")

REQUEST_ID_HEADER = "X-Request-ID"


class Problem(BaseModel):
    """A consistent error body (loosely RFC 7807)."""

    status: int = Field(description="HTTP status code.")
    title: str = Field(description="Short, human-readable summary of the error type.")
    detail: str | None = Field(default=None, description="Human-readable explanation.")
    correlation_id: str = Field(description="Request correlation id (also in X-Request-ID).")
    errors: list[dict] | None = Field(
        default=None, description="Field-level validation errors, when applicable."
    )


# Reusable OpenAPI documentation for the error responses (attached to routers/routes).
ERROR_RESPONSES: dict[int | str, dict] = {
    422: {"model": Problem, "description": "Validation error"},
    500: {"model": Problem, "description": "Internal server error"},
}


def _correlation_id(request: Request) -> str:
    return getattr(request.state, "correlation_id", "unknown")


def _problem(
    status: int, title: str, detail: str | None, correlation_id: str, errors=None
) -> JSONResponse:
    body = Problem(
        status=status, title=title, detail=detail, correlation_id=correlation_id, errors=errors
    )
    return JSONResponse(
        status_code=status,
        content=body.model_dump(exclude_none=True),
        headers={REQUEST_ID_HEADER: correlation_id},
    )


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Assign a correlation id per request (honouring an inbound ``X-Request-ID``) and echo
    it on the response."""

    async def dispatch(self, request: Request, call_next):
        correlation_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = correlation_id
        return response


async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Report loc/msg/type only — never echo the raw input, which may carry sensitive data.
    errors = [
        {"loc": list(e.get("loc", ())), "msg": e.get("msg", ""), "type": e.get("type", "")}
        for e in exc.errors()
    ]
    return _problem(422, "Unprocessable Entity", "Request validation failed.",
                    _correlation_id(request), errors=errors)


async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    try:
        title = HTTPStatus(exc.status_code).phrase
    except ValueError:
        title = "Error"
    detail = exc.detail if isinstance(exc.detail, str) else None
    return _problem(exc.status_code, title, detail, _correlation_id(request))


async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    correlation_id = _correlation_id(request)
    # Log the full traceback server-side; never expose it to the client.
    logger.exception("unhandled error [correlation_id=%s]", correlation_id)
    return _problem(500, "Internal Server Error", "An unexpected error occurred.", correlation_id)


def install_error_handling(app: FastAPI) -> None:
    """Register the correlation-id middleware and the problem-shaped exception handlers."""
    app.add_middleware(CorrelationIdMiddleware)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_exception_handler)
    app.add_exception_handler(Exception, _unhandled_handler)
