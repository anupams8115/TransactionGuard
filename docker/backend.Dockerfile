# Backend image (API + Celery worker share it). CPU only; no GPU needed.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# libgomp1: OpenMP runtime needed by XGBoost and scikit-learn wheels
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./
RUN uv sync --frozen --no-dev

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 app && mkdir -p /app/data /app/artifacts /app/reports \
    && chown -R app:app /app
USER app
ENV PATH="/app/.venv/bin:$PATH" TG_LOG_JSON=true
EXPOSE 8000
CMD ["uvicorn", "transactionguard.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
