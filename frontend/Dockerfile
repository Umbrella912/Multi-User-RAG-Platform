# syntax=docker/dockerfile:1
# Multi-stage build for the Streamlit frontend.
#
# Stage 1 (builder): create a virtualenv with all dependencies.
# Stage 2 (runtime): copy only the venv + code, run as a non-root user. This
# keeps the runtime image small (no build tools, no pip cache).

# ---------- Stage 1: build the virtualenv ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PIP_TARGET=/opt/venv

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/opt/venv -r requirements.txt

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH" \
    BACKEND_URL=http://backend:8000

# Non-root user.
RUN groupadd --system --gid 1001 appgroup \
    && useradd --system --uid 1001 --gid appgroup --no-create-home --home-dir /app appuser \
    && mkdir -p /app \
    && chown -R appuser:appgroup /app

WORKDIR /app

# Copy the virtualenv and application code.
COPY --from=builder --chown=appuser:appgroup /opt/venv /opt/venv
COPY --chown=appuser:appgroup . .

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8501/_stcore/health', timeout=3).status_code==200 else 1)"

# Streamlit headless server bound to all interfaces.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
