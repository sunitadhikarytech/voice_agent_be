# syntax=docker/dockerfile:1
#
# VA-02 — Containerize the service (slim, multi-stage).
# The same image runs identically locally, in CI, and on Cloud Run, which injects
# $PORT at runtime. Built on python:3.12-slim and run as a non-root user.

# ---- builder: install runtime dependencies into an isolated virtualenv ----
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- runtime: slim final image with only the venv + app source ----
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    PATH="/opt/venv/bin:$PATH"

# run as a non-root user
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY app ./app
COPY frontend ./frontend

USER appuser
EXPOSE 8080

# Liveness check for local/CI use (Cloud Run uses its own probes). No curl in slim,
# so use the stdlib.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
url='http://127.0.0.1:'+os.environ.get('PORT','8080')+'/healthz'; \
sys.exit(0 if urllib.request.urlopen(url, timeout=2).status==200 else 1)"

# `exec` so uvicorn receives SIGTERM directly (clean Cloud Run shutdowns).
# ${PORT} is expanded at runtime; Cloud Run sets it, otherwise defaults to 8080.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
