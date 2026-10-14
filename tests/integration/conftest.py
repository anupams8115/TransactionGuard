"""Integration fixtures: real FastAPI app + real SQLite DB (migrated with Alembic) + trained bundles.

Celery runs in eager mode here (``TG_CELERY_EAGER=true``), i.e. the broker is replaced by in-process
execution. Tests that need real PostgreSQL/Redis are marked and live in ``test_real_services.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from transactionguard.db.models import Role

PASSWORD = "correct-horse-battery-staple"


@dataclass
class Clients:
    admin: TestClient
    analyst: TestClient
    analyst2: TestClient
    viewer: TestClient
    other_ws: TestClient  # analyst in a DIFFERENT workspace
    anon: TestClient


def _login(app, email: str) -> TestClient:
    c = TestClient(app)
    r = c.post("/api/v1/auth/login", data={"username": email, "password": PASSWORD})
    assert r.status_code == 200, r.text
    c.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
    return c


@pytest.fixture
def app(trained_small):
    from transactionguard.admin import create_user, upgrade_db
    from transactionguard.api.main import create_app
    from transactionguard.db.session import get_engine, session_scope
    from transactionguard.inference.service import MODEL_STORE

    get_engine.cache_clear()
    MODEL_STORE.clear()
    upgrade_db()
    with session_scope() as db:
        for email, role, ws in [
            ("admin@a.test", Role.admin, "alpha"),
            ("analyst@a.test", Role.analyst, "alpha"),
            ("analyst2@a.test", Role.analyst, "alpha"),
            ("viewer@a.test", Role.viewer, "alpha"),
            ("analyst@b.test", Role.analyst, "beta"),
        ]:
            create_user(db, email=email, password=PASSWORD, role=role, workspace=ws)
    application = create_app()
    yield application
    get_engine.cache_clear()


@pytest.fixture
def clients(app) -> Clients:
    with TestClient(app):  # runs lifespan (model-version sync)
        pass
    return Clients(
        admin=_login(app, "admin@a.test"),
        analyst=_login(app, "analyst@a.test"),
        analyst2=_login(app, "analyst2@a.test"),
        viewer=_login(app, "viewer@a.test"),
        other_ws=_login(app, "analyst@b.test"),
        anon=TestClient(app),
    )


@pytest.fixture
def sample_rows(small_synthetic):
    """Recent transactions from the (training-time) test period as JSON-ready dicts."""
    from transactionguard.data.contract import SYNTHETIC
    from transactionguard.inference.service import _jsonable

    df = small_synthetic.sort_values("timestamp").tail(400)
    return [_jsonable(r) for r in df[SYNTHETIC.input_column_names].to_dict(orient="records")], df
