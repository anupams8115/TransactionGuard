"""FastAPI application factory.

Run locally:  uv run uvicorn transactionguard.api.main:app --reload --port 8000
Interactive docs: http://localhost:8000/docs
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from transactionguard.core.config import get_settings
from transactionguard.core.logging import configure_logging, get_logger
from transactionguard.db.session import session_scope
from transactionguard.monitoring.metrics import METRICS

log = get_logger(__name__)
API_PREFIX = "/api/v1"
UNMETERED = {"/api/v1/health", "/api/v1/ready"}


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    s = get_settings()
    s.assert_safe_for_production()
    from transactionguard.inference.service import sync_model_versions

    try:
        with session_scope() as db:
            n = sync_model_versions(db)
        log.info("api_started", environment=s.environment, model_versions=n)
    except Exception as exc:  # e.g. migrations not applied yet; /ready will report it
        log.error("startup_sync_failed", error=f"{type(exc).__name__}: {exc}")
    yield
    METRICS.flush()


def create_app() -> FastAPI:
    s = get_settings()
    configure_logging(s.log_level, s.log_json)
    app = FastAPI(
        title="TransactionGuard API",
        version="0.1.0",
        description=(
            "Explainable transaction-risk signals for analyst review. Scores are risk signals, "
            "**not proof of fraud**."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With", "Idempotency-Key"],
    )

    @app.middleware("http")
    async def observe(request: Request, call_next) -> Response:  # type: ignore[no-untyped-def]
        """Request id + structured access log + operational metrics (no payloads logged)."""
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log.exception("unhandled_error", path=request.url.path)
            response = JSONResponse({"detail": "Internal server error", "request_id": request_id}, 500)
        latency = (time.perf_counter() - t0) * 1000
        # Use the route TEMPLATE (/predictions/{prediction_id}), never the raw path, so metrics
        # group correctly and ids never end up in logs. FastAPI >= 0.140 reports included-router
        # paths without the router prefix, so re-attach it.
        route_path = getattr(request.scope.get("route"), "path", None)
        if route_path is None:
            template = "unmatched"
        elif request.url.path.startswith(API_PREFIX) and not route_path.startswith(API_PREFIX):
            template = API_PREFIX + route_path
        else:
            template = route_path
        response.headers["X-Request-ID"] = request_id
        if template.startswith(API_PREFIX) and template not in UNMETERED:
            log.info(
                "request",
                method=request.method,
                route=template,
                status=response.status_code,
                ms=round(latency, 1),
            )
            METRICS.record(template, request.method, response.status_code, latency)
        return response

    from transactionguard.api.routers import auth, batches, models, monitoring, predictions, system

    for r in (system, auth, models, predictions, batches, monitoring):
        app.include_router(r.router, prefix=API_PREFIX)
    return app


app = create_app()
