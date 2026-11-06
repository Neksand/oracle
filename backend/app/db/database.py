"""ORACLE - SQLite engine / session factory (SQLAlchemy 2.0 style)."""
from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# Default: file DB next to backend/ ; override with ORACLE_DB_URL for Postgres etc.
DB_PATH = os.environ.get("ORACLE_DB_PATH")
if not DB_PATH:
    DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "oracle.db")

DATABASE_URL = os.environ.get("ORACLE_DB_URL", f"sqlite:///{DB_PATH}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,
    future=True,
)

if DATABASE_URL.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: Any, _record: Any) -> None:
        # Foreign keys OFF by default in SQLite -> enable per connection.
        # WAL + foreign_keys = data-integrity story for the report.
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """Declarative base for all ORACLE ORM models."""


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables (idempotent)."""
    from app.db import orm_models  # noqa: F401 ensure models are registered

    Base.metadata.create_all(engine)
