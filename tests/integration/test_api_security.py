"""Authentication, authorization (roles) and workspace isolation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt


def test_login_sets_httponly_cookie_and_returns_token(app):
    from fastapi.testclient import TestClient

    from tests.integration.conftest import PASSWORD

    c = TestClient(app)
    r = c.post("/api/v1/auth/login", data={"username": "analyst@a.test", "password": PASSWORD})
    assert r.status_code == 200
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert c.get("/api/v1/auth/me").json()["role"] == "analyst"  # cookie auth works for GET


def test_wrong_password_and_unknown_user_look_identical(clients):
    a = clients.anon.post(
        "/api/v1/auth/login", data={"username": "analyst@a.test", "password": "nope-nope-nope"}
    )
    b = clients.anon.post(
        "/api/v1/auth/login", data={"username": "ghost@a.test", "password": "nope-nope-nope"}
    )
    assert a.status_code == b.status_code == 401
    assert a.json() == b.json()


def test_password_is_hashed_with_argon2(app, clients):
    from sqlalchemy import select

    from transactionguard.db.models import User
    from transactionguard.db.session import session_scope

    with session_scope() as db:
        h = db.scalars(select(User.password_hash).where(User.email == "analyst@a.test")).one()
    assert h.startswith("$argon2id$")


def test_requests_without_valid_token_are_rejected(clients):
    assert clients.anon.get("/api/v1/predictions").status_code == 401
    bad = {"Authorization": "Bearer not-a-jwt"}
    assert clients.anon.get("/api/v1/predictions", headers=bad).status_code == 401


def test_forged_and_expired_tokens_are_rejected(clients):
    me = clients.analyst.get("/api/v1/auth/me").json()
    claims = {
        "sub": me["id"],
        "ws": me["workspace_id"],
        "role": "admin",
        "iss": "transactionguard",
        "exp": datetime.now(UTC) + timedelta(minutes=5),
    }
    forged = jwt.encode(claims, "attacker-secret-attacker-secret-1234567890", algorithm="HS256")
    r = clients.anon.get("/api/v1/users", headers={"Authorization": f"Bearer {forged}"})
    assert r.status_code == 401
    unsigned = jwt.encode(claims, key=None, algorithm="none")
    assert (
        clients.anon.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {unsigned}"}).status_code
        == 401
    )
    from transactionguard.core.config import get_settings

    expired = jwt.encode(
        {**claims, "role": "analyst", "exp": datetime.now(UTC) - timedelta(seconds=1)},
        get_settings().jwt_secret.get_secret_value(),
        algorithm="HS256",
    )
    assert (
        clients.anon.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired}"}).status_code == 401
    )


def test_role_based_access(clients, sample_rows):
    rows, _ = sample_rows
    body = {"contract": "synthetic", "transaction": rows[0]}
    assert clients.viewer.post("/api/v1/predictions", json=body).status_code == 403
    assert clients.viewer.get("/api/v1/predictions").status_code == 200
    assert clients.analyst.get("/api/v1/users").status_code == 403
    assert clients.admin.get("/api/v1/users").status_code == 200
    new_user = {"email": "new.user@alpha-bank.com", "password": "another-long-password", "role": "viewer"}
    assert clients.analyst.post("/api/v1/users", json=new_user).status_code == 403
    created = clients.admin.post("/api/v1/users", json=new_user)
    assert created.status_code == 201 and created.json()["workspace_name"] == "alpha"


def test_cookie_auth_requires_csrf_header_for_writes(app, sample_rows):
    from fastapi.testclient import TestClient

    from tests.integration.conftest import PASSWORD

    rows, _ = sample_rows
    c = TestClient(app)
    c.post("/api/v1/auth/login", data={"username": "analyst@a.test", "password": PASSWORD})
    body = {"contract": "synthetic", "transaction": rows[0]}
    assert c.post("/api/v1/predictions", json=body).status_code == 403
    ok = c.post("/api/v1/predictions", json=body, headers={"X-Requested-With": "fetch"})
    assert ok.status_code == 201


def test_workspace_isolation(clients, sample_rows):
    rows, _ = sample_rows
    pred = clients.analyst.post("/api/v1/predictions", json={"contract": "synthetic", "transaction": rows[0]})
    pid = pred.json()["id"]
    assert clients.other_ws.get(f"/api/v1/predictions/{pid}").status_code == 404
    assert clients.other_ws.get("/api/v1/predictions").json()["total"] == 0
    review = {"status": "under_review"}
    assert clients.other_ws.put(f"/api/v1/predictions/{pid}/review", json=review).status_code == 404
    # Same transaction_id in another workspace is a different transaction (no cross-tenant conflict).
    other = clients.other_ws.post(
        "/api/v1/predictions", json={"contract": "synthetic", "transaction": rows[0]}
    )
    assert other.status_code == 201 and other.json()["id"] != pid


def test_logs_do_not_contain_raw_payloads(clients, sample_rows, capsys):
    rows, _ = sample_rows
    clients.analyst.post("/api/v1/predictions", json={"contract": "synthetic", "transaction": rows[1]})
    err = capsys.readouterr().err
    assert rows[1]["merchant_id"] not in err and str(rows[1]["amount"]) not in err
