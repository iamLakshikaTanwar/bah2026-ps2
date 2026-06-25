# syntax=docker/dockerfile:1
# Minimal CPU image for the cloudremoval API. Installs ONLY the core stack plus
# the serving extra (rio-tiler/redis client); heavy geospatial/accelerator deps
# are optional and not baked into the default image.
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CLOUDREMOVAL_LOG=INFO

# Minimal build deps (kept small; geospatial libs intentionally excluded here).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU torch first from the official index, then the package core.
COPY requirements.txt ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Project sources.
COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

# Editable install with the serving extra so /tiles works; core already present.
RUN pip install -e ".[serve]" || pip install -e .

EXPOSE 8000

# Healthcheck hits the FastAPI /health endpoint (B4's app).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Default: serve the API. The app module is owned by B4 (serving/app.py).
CMD ["uvicorn", "cloudremoval.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
