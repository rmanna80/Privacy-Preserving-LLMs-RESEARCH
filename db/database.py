"""
Database engine and session management.

A single-file SQLite database at data/wealth.db. SQLModel for schema,
SQLAlchemy 2.x under the hood.

CHANGED IN STEP 2:
    expire_on_commit=False

    Why: get_session() is a context manager. After commit() runs and the
    session closes, normal SQLAlchemy "expires" attributes on returned
    objects, so any attribute access would try to lazy-load from a closed
    session and throw DetachedInstanceError. Setting expire_on_commit=False
    keeps the loaded data hydrated, so repository functions can return
    objects that the Streamlit UI reads freely.
"""

from pathlib import Path
from contextlib import contextmanager
from typing import Iterator

from sqlmodel import SQLModel, create_engine, Session

DB_PATH = Path("data/wealth.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
)


def init_db() -> None:
    """Create all tables. Idempotent — safe to call repeatedly."""
    from db import models  # noqa: F401  (import-for-side-effect: register tables)

    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session() -> Iterator[Session]:
    """Context-managed DB session.

    Usage:
        with get_session() as s:
            s.add(family)
            # auto-commit on exit, auto-rollback on error
    """
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()