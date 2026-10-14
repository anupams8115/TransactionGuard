"""Database migrations: upgrade -> downgrade -> upgrade, and the models match the migrations.

SQLite runs everywhere. PostgreSQL runs when TG_TEST_DATABASE_URL is set (CI service container).
"""

from __future__ import annotations

import os

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, inspect

from transactionguard.core.config import PROJECT_ROOT
from transactionguard.db.models import Base

EXPECTED = {
    "workspaces",
    "users",
    "model_versions",
    "transactions",
    "predictions",
    "reviews",
    "review_events",
    "confirmed_labels",
    "batch_jobs",
    "monitoring_reports",
    "request_logs",
}


def _cfg(url: str) -> Config:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def _roundtrip(url: str) -> None:
    cfg = _cfg(url)
    command.upgrade(cfg, "head")
    engine = create_engine(url)
    assert set(inspect(engine).get_table_names()) >= EXPECTED
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == [], f"ORM models and migrations disagree: {diff}"
    command.downgrade(cfg, "base")
    assert not (EXPECTED & set(inspect(engine).get_table_names()))
    command.upgrade(cfg, "head")
    engine.dispose()


def test_sqlite_migrations_roundtrip(tmp_path):
    _roundtrip(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")


@pytest.mark.postgres
def test_postgres_migrations_roundtrip():
    url = os.environ.get("TG_TEST_DATABASE_URL")
    if not url:
        pytest.skip("TG_TEST_DATABASE_URL not set")
    _roundtrip(url)
