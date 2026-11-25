# syntax=docker/dockerfile:1
# Multi-stage build for the FastAPI backend.
#
# Stage 1 (model-cache): download the embedding model once into a dedicated
#   layer. This layer is cached independently of application code, so rebuilds
#   after code changes do NOT re-download the model.
# Stage 2 (runtime): slim runtime with deps + code + the cached model, running
#   as a non-root user.

# ---------- Stage 1: pre-download the embedding model ----------
FROM python:3.11-slim AS model-cache

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/models

# Install only what is needed to fetch the model (sentence-transformers pulls
# the model + its weights). Pinned via requirements to stay reproducible.
WORKDIR /cache
COPY requirements.txt .
RUN pip install --no-cache-dir sentence-transformers==3.3.1

# Download the default embedding model into the HF cache (a cached layer).
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5')"

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# Create a non-root user and a writable data directory.
RUN groupadd --system --gid 1001 appgroup \
    && useradd --system --uid 1001 --gid appgroup --no-create-home --home-dir /app appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appgroup /app

WORKDIR /app

# Install Python dependencies (cached layer).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the pre-downloaded embedding model from the cache stage.
COPY --from=model-cache --chown=appuser:appgroup /opt/models /opt/models

# Copy application code.
COPY --chown=appuser:appgroup . .

# Alembic + uploads/chroma (if local) live under /app/data.
VOLUME ["/app/data"]

USER appuser

EXPOSE 8000

# Run migrations then start the API. Healthcheck hits the liveness probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://127.0.0.1:8000/health/live', timeout=3).status_code==200 else 1)"

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
