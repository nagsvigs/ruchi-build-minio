"""Database connection pool + schema bootstrap for Ruchi's memory layer.

psycopg 3 with a small connection pool and pgvector. All schema changes
use ``CREATE ... IF NOT EXISTS`` (plus ``ALTER TABLE ... ADD COLUMN IF
NOT EXISTS`` for upgrades) so this is safe to run repeatedly and never
drops data.

Everything is defensive: a missing ``DATABASE_URL``, unreachable server,
or any psycopg error is logged and re-raised as :class:`DatabaseError`,
which callers swallow so the voice loop keeps working without memory.
"""
from __future__ import annotations

import atexit
import logging
import os
import threading
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg_pool import ConnectionPool

try:  # pgvector is optional at import time so the app can boot without it
    from pgvector.psycopg import register_vector
except Exception:  # pragma: no cover
    register_vector = None  # type: ignore[assignment]

logger = logging.getLogger("ruchi.memory.db")


class DatabaseError(RuntimeError):
    """Raised when the database layer cannot service a request."""


_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def enabled() -> bool:
    return os.getenv("MEMORY_ENABLED", "1").lower() not in ("0", "false", "no", "off")


def _build_dsn() -> str:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise DatabaseError("DATABASE_URL is not set; memory is disabled.")
    return dsn


def get_pool() -> ConnectionPool:
    """Return the process-wide pool, creating it lazily.

    Connections open lazily, so an unreachable DB only fails on first
    query — the agent still boots.
    """
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is not None:
            return _pool
        dsn = _build_dsn()
        logger.info("initializing Postgres pool")
        pool = ConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=int(os.getenv("DATABASE_POOL_MAX", "5")),
            timeout=float(os.getenv("DATABASE_POOL_TIMEOUT", "5")),
            kwargs={"autocommit": False},
            open=False,
        )
        pool.open(wait=False)
        _pool = pool
        atexit.register(close_pool)
    return _pool


def close_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Yield a pooled connection with pgvector registered.

    Commits on clean exit, rolls back on error, translates psycopg
    errors into :class:`DatabaseError`.
    """
    pool = get_pool()
    try:
        with pool.connection() as conn:
            if register_vector is not None:
                try:
                    register_vector(conn)
                except Exception:  # pragma: no cover - vector type may be absent
                    pass
            try:
                yield conn
                conn.commit()
            except psycopg.Error as exc:
                conn.rollback()
                raise DatabaseError(str(exc)) from exc
    except DatabaseError:
        raise
    except psycopg.Error as exc:
        raise DatabaseError(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise DatabaseError(f"unexpected DB error: {exc}") from exc


def _supports_hnsw(conn: psycopg.Connection) -> bool:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
            row = cur.fetchone()
    except psycopg.Error as exc:
        logger.warning("could not query pgvector version: %s", exc)
        return False
    if not row or not row[0]:
        return False
    try:
        parts = [int(p) for p in str(row[0]).split(".") if p.isdigit()]
    except ValueError:
        return False
    if not parts:
        return False
    return parts[0] >= 1 or (parts[0] == 0 and len(parts) > 1 and parts[1] >= 5)


def bootstrap_schema(embedding_dimensions: int | None = None) -> None:
    """Create the extension, tables and indexes if they don't exist."""
    if embedding_dimensions is None:
        embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
    dim = int(embedding_dimensions)

    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    mode        TEXT NOT NULL DEFAULT 'cook',
                    recipe_id   TEXT,
                    state       JSONB NOT NULL DEFAULT '{}'::jsonb,
                    started_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    ended_at    TIMESTAMPTZ,
                    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                """
            )
            cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS mode TEXT NOT NULL DEFAULT 'cook';")
            cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS recipe_id TEXT;")
            cur.execute("ALTER TABLE sessions ADD COLUMN IF NOT EXISTS state JSONB NOT NULL DEFAULT '{}'::jsonb;")

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id          BIGSERIAL PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role        TEXT NOT NULL CHECK (role IN ('user','assistant','system','tool')),
                    content     TEXT NOT NULL DEFAULT '',
                    tool_calls  JSONB,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                """
            )
            cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS tool_calls JSONB;")

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS memories (
                    id          BIGSERIAL PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    memory      TEXT NOT NULL,
                    memory_type TEXT NOT NULL DEFAULT 'general',
                    importance  REAL NOT NULL DEFAULT 0.5
                                CHECK (importance >= 0.0 AND importance <= 1.0),
                    embedding   VECTOR({dim}),
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb
                );
                """
            )

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS recipes (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    region      TEXT,
                    category    TEXT,
                    data        JSONB NOT NULL,
                    embedding   VECTOR({dim}),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

        with conn.cursor() as cur:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON sessions(started_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_user_session ON conversations(user_id, session_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_conv_created_at ON conversations(created_at);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);")

        if _supports_hnsw(conn):
            for stmt in (
                "CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw "
                "ON memories USING hnsw (embedding vector_cosine_ops);",
                "CREATE INDEX IF NOT EXISTS idx_recipes_embedding_hnsw "
                "ON recipes USING hnsw (embedding vector_cosine_ops);",
            ):
                try:
                    with conn.cursor() as cur:
                        cur.execute(stmt)
                except psycopg.Error as exc:
                    logger.warning("could not create HNSW index: %s", exc)

    logger.info("schema bootstrap complete (embedding_dimensions=%d)", dim)


def vector_dimensions() -> int | None:
    """Declared dimension of ``memories.embedding`` (or ``None``)."""
    try:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT atttypmod FROM pg_attribute "
                    "WHERE attrelid = 'memories'::regclass AND attname = 'embedding'"
                )
                row = cur.fetchone()
    except DatabaseError:
        return None
    if not row or row[0] is None or row[0] <= 0:
        return None
    return int(row[0])


__all__ = [
    "DatabaseError",
    "enabled",
    "get_pool",
    "close_pool",
    "connection",
    "bootstrap_schema",
    "vector_dimensions",
]
