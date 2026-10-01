# syntax=docker/dockerfile:1
#
# Multi-stage build. The wheel-building stage carries gcc/g++ (LightGBM and
# SHAP need them); the runtime stage does not, which keeps the shipped image
# smaller and removes a compiler from the production attack surface.

# --------------------------------------------------------------------------- #
# stage 1: build wheels
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt


# --------------------------------------------------------------------------- #
# stage 2: runtime
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime

# libgomp1 is the OpenMP runtime LightGBM links against. Without it the image
# builds fine and then fails at import time -- a slow way to find out.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

COPY src/ ./src/
COPY dashboard/ ./dashboard/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY tests/ ./tests/

# Artifacts and data are mounted or generated at runtime, not baked in.
RUN mkdir -p /app/artifacts /app/data/raw /app/data/processed \
    && useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
