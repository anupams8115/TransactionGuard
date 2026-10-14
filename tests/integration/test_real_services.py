"""Tests against REAL services (not mocks). Skipped unless the service URLs are provided.

    TG_TEST_REDIS_URL=redis://localhost:6379/0     -> Celery with a real broker + real worker thread
    TG_TEST_DATABASE_URL=postgresql+psycopg://...   -> see test_migrations.py

CI provides both as GitHub Actions service containers.
"""

from __future__ import annotations

import io
import os
import time

import pytest

REDIS_URL = os.environ.get("TG_TEST_REDIS_URL")


@pytest.mark.redis
@pytest.mark.skipif(not REDIS_URL, reason="TG_TEST_REDIS_URL not set")
def test_batch_job_runs_on_a_real_celery_worker(clients, sample_rows, monkeypatch):
    from celery.contrib.testing.worker import start_worker

    from transactionguard.core.config import get_settings
    from transactionguard.data.contract import SYNTHETIC
    from transactionguard.worker.celery_app import celery_app, configure_celery

    monkeypatch.setenv("TG_CELERY_EAGER", "false")
    monkeypatch.setenv("TG_REDIS_URL", REDIS_URL)
    get_settings.cache_clear()
    configure_celery()
    assert celery_app.conf.task_always_eager is False

    _, df = sample_rows
    content = df.head(120)[SYNTHETIC.input_column_names].to_csv(index=False).encode()
    with start_worker(celery_app, pool="solo", perform_ping_check=False, shutdown_timeout=30):
        r = clients.analyst.post(
            "/api/v1/batches",
            data={"contract": "synthetic"},
            files={"file": ("batch.csv", io.BytesIO(content), "text/csv")},
        )
        assert r.status_code == 202
        job_id = r.json()["id"]
        deadline = time.time() + 60
        status = "queued"
        while time.time() < deadline and status in ("queued", "running"):
            time.sleep(0.5)
            status = clients.analyst.get(f"/api/v1/batches/{job_id}").json()["status"]
    assert status == "succeeded"
    assert clients.analyst.get(f"/api/v1/batches/{job_id}").json()["processed_rows"] == 120

    monkeypatch.setenv("TG_CELERY_EAGER", "true")
    get_settings.cache_clear()
    configure_celery()
