"""Batch jobs (validation, idempotency, ownership, retries) and monitoring reports."""

from __future__ import annotations

import io
import uuid

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

from transactionguard.db.models import BatchJob, JobStatus, Prediction
from transactionguard.db.session import session_scope


def _csv(df: pd.DataFrame) -> bytes:
    from transactionguard.data.contract import SYNTHETIC

    return df[SYNTHETIC.input_column_names].to_csv(index=False).encode()


def _upload(client, content: bytes, key: str | None = None):
    headers = {"Idempotency-Key": key} if key else {}
    return client.post(
        "/api/v1/batches",
        data={"contract": "synthetic"},
        files={"file": ("batch.csv", io.BytesIO(content), "text/csv")},
        headers=headers,
    )


def test_batch_end_to_end_with_invalid_rows(clients, sample_rows):
    _, df = sample_rows
    bad = df.head(300).copy()
    bad["channel"] = bad["channel"].astype(object)
    bad.iloc[5, bad.columns.get_loc("channel")] = "fax"
    r = _upload(clients.analyst, _csv(bad))
    assert r.status_code == 202, r.text
    job_id = r.json()["id"]
    job = clients.analyst.get(f"/api/v1/batches/{job_id}").json()  # background task ran after response
    assert job["status"] == "succeeded", job
    assert job["total_rows"] == 300 and job["invalid_rows"] == 1 and job["processed_rows"] == 299
    codes = {i["code"] for i in job["validation_report"]["issues"]}
    assert "invalid_category" in codes
    preds = clients.analyst.get(
        f"/api/v1/predictions?batch_job_id={job_id}&page_size=200&flagged=true"
    ).json()
    assert preds["total"] == job["flagged_rows"]
    assert all(p["explanation_status"] == "ok" for p in preds["items"])  # explained asynchronously


def test_batch_idempotency_key(clients, sample_rows):
    _, df = sample_rows
    content = _csv(df.head(50))
    a = _upload(clients.analyst, content, key="upload-1")
    b = _upload(clients.analyst, content, key="upload-1")
    assert a.json()["id"] == b.json()["id"] and b.json()["idempotent_replay"] is True
    c = _upload(clients.analyst, _csv(df.head(60)), key="upload-1")
    assert c.status_code == 409


def test_resubmitting_rows_never_duplicates_predictions(clients, sample_rows):
    _, df = sample_rows
    _upload(clients.analyst, _csv(df.head(80)))
    second = _upload(clients.analyst, _csv(df.head(80))).json()
    job = clients.analyst.get(f"/api/v1/batches/{second['id']}").json()
    assert job["validation_report"]["scoring"]["replayed_existing_predictions"] == 80
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(Prediction)) == 80


def test_job_ownership(clients, sample_rows):
    _, df = sample_rows
    job_id = _upload(clients.analyst, _csv(df.head(20))).json()["id"]
    assert clients.analyst2.get(f"/api/v1/batches/{job_id}").status_code == 404  # same workspace, not owner
    assert clients.other_ws.get(f"/api/v1/batches/{job_id}").status_code == 404  # other workspace
    assert clients.admin.get(f"/api/v1/batches/{job_id}").status_code == 200  # workspace admin
    assert all(j["id"] != job_id for j in clients.analyst2.get("/api/v1/batches").json())


def test_upload_rejections(clients, sample_rows):
    _, df = sample_rows
    missing = df.drop(columns=["amount"]).head(5).to_csv(index=False).encode()
    r = _upload(clients.analyst, missing)
    assert r.status_code == 422 and "missing_columns" in str(r.json())
    assert _upload(clients.viewer, _csv(df.head(5))).status_code == 403


def test_batch_crash_midway_then_retry_has_no_duplicates(clients, sample_rows, monkeypatch):
    """Simulate a worker crash after the first chunk; rerunning the job completes it without
    duplicating the rows that were already committed."""
    import transactionguard.worker.tasks as tasks

    monkeypatch.setattr(tasks, "CHUNK_ROWS", 50)
    _, df = sample_rows
    original = tasks.score_transactions
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OperationalError("UPDATE ...", {}, Exception("connection reset"))
        return original(*a, **k)

    monkeypatch.setattr(tasks, "score_transactions", flaky)
    job_id = _upload(clients.analyst, _csv(df.head(150))).json()["id"]
    # Eager Celery retried automatically with the same task; the job must end succeeded.
    job = clients.analyst.get(f"/api/v1/batches/{job_id}").json()
    assert job["status"] == "succeeded" and job["attempts"] == 2, job
    with session_scope() as db:
        n = db.scalar(
            select(func.count()).select_from(Prediction).where(Prediction.batch_job_id == uuid.UUID(job_id))
        )
    assert n == 150


def test_retries_are_bounded_and_failure_is_visible(clients, sample_rows, monkeypatch):
    import transactionguard.worker.tasks as tasks

    def always_down(*_a, **_k):
        raise OperationalError("SELECT 1", {}, Exception("database unavailable"))

    monkeypatch.setattr(tasks, "score_transactions", always_down)
    _, df = sample_rows
    job_id = _upload(clients.analyst, _csv(df.head(10))).json()["id"]
    job = clients.analyst.get(f"/api/v1/batches/{job_id}").json()
    assert job["status"] == "failed" and "database unavailable" in job["error"]
    from transactionguard.core.config import get_settings

    assert job["attempts"] == get_settings().celery_max_retries + 1
    # A manual retry after the dependency recovers completes the job.
    monkeypatch.undo()
    r = clients.analyst.post(f"/api/v1/batches/{job_id}/retry")
    assert r.status_code == 200
    assert clients.analyst.get(f"/api/v1/batches/{job_id}").json()["status"] == "succeeded"


def test_non_transient_errors_fail_fast(clients, sample_rows, monkeypatch):
    import transactionguard.worker.tasks as tasks

    def broken(*_a, **_k):
        raise ValueError("model/contract mismatch")

    monkeypatch.setattr(tasks, "score_transactions", broken)
    _, df = sample_rows
    job_id = _upload(clients.analyst, _csv(df.head(10))).json()["id"]
    with session_scope() as db:
        job = db.get(BatchJob, uuid.UUID(job_id))
        assert job.status == JobStatus.failed and job.attempts == 1


# ------------------------------------------------------------------------------ monitoring
@pytest.fixture
def scored_window(clients, sample_rows):
    _, df = sample_rows
    _upload(clients.analyst, _csv(df.head(300)))
    return df


def test_monitoring_report(clients, scored_window):
    r = clients.analyst.post("/api/v1/monitoring/reports", json={"contract": "synthetic", "window_days": 7})
    assert r.status_code == 202
    rep = clients.analyst.get(f"/api/v1/monitoring/reports/{r.json()['id']}").json()
    assert rep["status"] == "succeeded", rep
    s = rep["summary"]
    assert s["window"]["rows"] == 300 and s["reference"]["rows"] > 0
    assert "drift" in s and "score" in s["drift"]["columns"]
    assert {"statistic", "statistic_kind", "drift_detected"} <= set(s["drift"]["columns"]["score"])
    assert s["schema_violations"]["batch_rows_total"] == 300
    assert s["performance"]["available"] is False  # no confirmed labels -> no accuracy claims
    assert "not proof" in s["interpretation"]
    html = clients.analyst.get(f"/api/v1/monitoring/reports/{r.json()['id']}/html")
    assert html.status_code == 200 and "<html" in html.text.lower()
    assert clients.other_ws.get(f"/api/v1/monitoring/reports/{r.json()['id']}").status_code == 404


def test_operational_metrics_admin_only(clients, scored_window):
    assert clients.analyst.get("/api/v1/monitoring/operational").status_code == 403
    m = clients.admin.get("/api/v1/monitoring/operational").json()
    assert m["requests"] > 0 and any(r["route"] == "/api/v1/batches" for r in m["routes"])
