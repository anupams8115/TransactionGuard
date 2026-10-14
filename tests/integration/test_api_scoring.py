"""Scoring, idempotency, history features, explanations, reviews, labels, metadata, health."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from transactionguard.db.models import Prediction, Transaction
from transactionguard.db.session import session_scope


def _predict(client, row, **extra):
    return client.post("/api/v1/predictions", json={"contract": "synthetic", "transaction": row, **extra})


def test_prediction_records_full_trace(clients, sample_rows):
    rows, _ = sample_rows
    r = _predict(clients.analyst, rows[0])
    assert r.status_code == 201, r.text
    p = r.json()
    for key in (
        "transaction_id",
        "model_version",
        "preprocessing_version",
        "score",
        "score_kind",
        "score_meaning",
        "threshold",
        "recommendation",
        "created_at",
        "explanation_status",
        "latency_ms",
    ):
        assert p[key] is not None, key
    assert p["synthetic"] is True
    assert p["recommendation"] == ("review" if p["flagged"] else "no_action")
    assert p["flagged"] == (p["score"] >= p["threshold"])
    assert "not proof" in p["score_meaning"]

    detail = clients.analyst.get(f"/api/v1/predictions/{p['id']}").json()
    trace = detail["trace"]
    assert trace["bundle_sha256"] and trace["threshold_applied"] == p["threshold"]
    assert trace["preprocessing_version"] == p["preprocessing_version"]
    assert trace["threshold_policy"]["selected_on"] == "validation"
    exp = detail["explanation"]
    assert detail["explanation_status"] == "ok" and exp["status"] == "ok"
    assert exp["output_scale"].startswith("calibrated log-odds")
    assert exp["contributions"] and {"feature", "value", "contribution", "direction"} <= set(
        exp["contributions"][0]
    )


def test_duplicate_submission_is_idempotent(clients, sample_rows):
    rows, _ = sample_rows
    first = _predict(clients.analyst, rows[2])
    again = _predict(clients.analyst, rows[2])
    assert first.status_code == 201 and again.status_code == 200
    assert again.json()["id"] == first.json()["id"] and again.json()["idempotent_replay"] is True
    changed = _predict(clients.analyst, {**rows[2], "amount": rows[2]["amount"] + 1})
    assert changed.status_code == 409
    with session_scope() as db:
        n = db.scalar(
            select(func.count())
            .select_from(Transaction)
            .where(Transaction.transaction_id == rows[2]["transaction_id"])
        )
        assert n == 1
        assert db.scalar(select(func.count()).select_from(Prediction)) == 1


@pytest.mark.parametrize("patch", [{"channel": "fax"}, {"unexpected": 1}, {"amount": -3}, {"is_fraud": 1}])
def test_invalid_transactions_rejected(clients, sample_rows, patch):
    rows, _ = sample_rows
    r = _predict(clients.analyst, {**rows[3], **patch})
    assert r.status_code == 422


def test_missing_field_is_rejected_not_defaulted(clients, sample_rows):
    rows, _ = sample_rows
    row = dict(rows[3])
    row.pop("country")
    assert _predict(clients.analyst, row).status_code == 422


def test_history_features_come_from_stored_earlier_transactions(clients, sample_rows):
    rows, df = sample_rows
    acct = df["account_id"].value_counts().index[0]
    same = [r for r in rows if r["account_id"] == acct][:3]
    ids = [_predict(clients.analyst, r).json()["id"] for r in same]
    feats = [clients.analyst.get(f"/api/v1/predictions/{i}").json()["features"] for i in ids]
    assert [f["acct_prior_txn_count"] for f in feats] == [0, 1, 2]
    assert feats[0]["secs_since_prev_txn"] is None  # genuinely missing, not invented


def test_explanation_recompute_uses_stored_features(clients, sample_rows):
    rows, _ = sample_rows
    pid = _predict(clients.analyst, rows[5], explain=False).json()["id"]
    assert clients.analyst.get(f"/api/v1/predictions/{pid}").json()["explanation_status"] == "not_requested"
    r = clients.analyst.post(f"/api/v1/predictions/{pid}/explanation")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_review_workflow_and_label_separation(clients, sample_rows):
    rows, _ = sample_rows
    pid = _predict(clients.analyst, rows[6]).json()["id"]
    url = f"/api/v1/predictions/{pid}/review"
    assert clients.viewer.put(url, json={"status": "under_review"}).status_code == 403
    assert clients.analyst.put(url, json={"status": "under_review"}).json()["status"] == "under_review"
    assert clients.analyst.put(url, json={"status": "resolved"}).status_code == 422  # needs an assessment
    done = clients.analyst.put(
        url, json={"status": "resolved", "assessment": "suspicious", "note": "card testing"}
    )
    assert done.status_code == 200 and done.json()["assessment"] == "suspicious"
    history = clients.analyst.get(f"{url}/history").json()
    assert [h["to_status"] for h in history] == ["under_review", "resolved"]

    # Analyst opinion is NOT a confirmed label.
    assert clients.analyst.get(f"/api/v1/predictions/{pid}").json()["confirmed_label"] is None
    label = {"is_fraud": True, "source": "chargeback"}
    assert clients.analyst.put(f"/api/v1/predictions/{pid}/label", json=label).status_code == 403
    assert clients.admin.put(f"/api/v1/predictions/{pid}/label", json=label).status_code == 200
    assert clients.analyst.get(f"/api/v1/predictions/{pid}").json()["confirmed_label"] is True


def test_review_queue_lists_flagged_by_score(clients, sample_rows):
    rows, _ = sample_rows
    for r in rows[:120]:
        _predict(clients.analyst, r, explain=False)
    q = clients.analyst.get("/api/v1/reviews/queue?page_size=200").json()
    scores = [p["score"] for p in q["items"]]
    assert all(p["flagged"] for p in q["items"]) and scores == sorted(scores, reverse=True)


def test_prediction_filters(clients, sample_rows):
    rows, _ = sample_rows
    for r in rows[:30]:
        _predict(clients.analyst, r, explain=False)
    base = "/api/v1/predictions?page_size=200"
    all_ = clients.analyst.get(base).json()
    assert all_["total"] == 30
    hi = clients.analyst.get(base + "&score_min=0.5").json()
    assert all(p["score"] >= 0.5 for p in hi["items"])
    q = clients.analyst.get(base + f"&q={rows[0]['transaction_id']}").json()
    assert q["total"] == 1
    mv = all_["items"][0]["model_version"]
    assert clients.analyst.get(base + f"&model_version={mv}").json()["total"] == 30
    assert clients.analyst.get(base + "&model_version=nope").json()["total"] == 0
    unreviewed = clients.analyst.get(base + "&review_status=unreviewed").json()
    assert unreviewed["total"] == sum(p["flagged"] for p in all_["items"])


def test_model_metadata_contracts_and_evaluations(clients):
    models = clients.viewer.get("/api/v1/models").json()
    assert any(m["is_active"] for m in models)
    active = next(m for m in models if m["is_active"])
    detail = clients.viewer.get(f"/api/v1/models/{active['model_version']}").json()
    assert detail["global_importance"] and detail["threshold_policy"]["selected_on"] == "validation"
    contract = clients.viewer.get("/api/v1/contracts/synthetic").json()
    names = [f["name"] for f in contract["input_fields"]]
    assert "is_fraud" not in names and "merchant_id" in names
    evals = clients.viewer.get("/api/v1/evaluations").json()
    assert evals and evals[0]["synthetic"] is True
    assert clients.viewer.get(f"/api/v1/evaluations/{evals[0]['id']}").status_code == 200
    assert clients.viewer.get("/api/v1/evaluations/..%2F..%2Fsecrets").status_code == 404


def test_health_and_readiness(clients):
    assert clients.anon.get("/api/v1/health").json() == {"status": "ok"}
    ready = clients.anon.get("/api/v1/ready")
    assert ready.status_code == 200 and ready.json()["checks"]["database"] == "ok"


def test_readiness_reports_failed_dependencies(clients, monkeypatch):
    import transactionguard.api.routers.system as system

    def broken_engine(*_a, **_k):
        raise ConnectionError("database down")

    monkeypatch.setattr(system, "get_engine", broken_engine)
    r = clients.anon.get("/api/v1/ready")
    assert r.status_code == 503 and r.json()["checks"]["database"].startswith("error")


def test_scoring_without_a_trained_model_returns_503(clients, sample_rows, monkeypatch):
    from transactionguard.inference.service import MODEL_STORE, ModelNotAvailableError

    def none(_c):
        raise ModelNotAvailableError("No active model for 'synthetic'.")

    monkeypatch.setattr(MODEL_STORE, "active", none)
    rows, _ = sample_rows
    assert _predict(clients.analyst, rows[0]).status_code == 503
    assert clients.anon.get("/api/v1/ready").status_code == 503
