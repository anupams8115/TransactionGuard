"""Engine/session factory.

Synchronous SQLAlchemy on purpose: FastAPI runs sync endpoints in a threadpool and Celery
workers are synchronous, so one session model serves both.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from transactionguard.core.config import get_settings


def _make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_conn, _):  # type: ignore[no-untyped-def]
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer
            cur.close()

        return engine
    return create_engine(url, pool_pre_ping=True, pool_size=10, max_overflow=10)


@lru_cache
def get_engine(url: str | None = None) -> Engine:
    return _make_engine(url or get_settings().database_url)


def get_sessionmaker() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(get_settings().database_url), expire_on_commit=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    session = get_sessionmaker()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def as_utc(dt: datetime | None) -> datetime | None:
    """SQLite drops tzinfo; everything we store is UTC, so re-attach it."""
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
