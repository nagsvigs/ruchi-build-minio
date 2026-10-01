"""High-level CRUD over Ruchi's memory tables.

Every public function is defensive: DB/embedding errors are logged and
converted to a safe default (``[]``, ``None``, ``False``) so a single
broken call never crashes the voice loop.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from pgvector import Vector

from .database import DatabaseError, connection, enabled
from .embeddings import embed_for_query, embed_for_storage

logger = logging.getLogger("ruchi.memory")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def memory_enabled() -> bool:
    return enabled() and bool(os.getenv("DATABASE_URL"))


@dataclass
class MemoryHit:
    id: int
    memory: str
    memory_type: str
    importance: float
    similarity: float
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "memory": self.memory,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "similarity": round(self.similarity, 4),
        }


@dataclass
class ConversationTurn:
    role: str
    content: str
    created_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #
def create_session(
    user_id: str,
    session_id: str,
    mode: str = "cook",
    recipe_id: str | None = None,
    state: dict[str, Any] | None = None,
) -> bool:
    if not memory_enabled() or not user_id or not session_id:
        return False
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (id, user_id, mode, recipe_id, state)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (id) DO UPDATE
                  SET user_id = EXCLUDED.user_id,
                      mode = EXCLUDED.mode,
                      recipe_id = EXCLUDED.recipe_id,
                      state = EXCLUDED.state
                """,
                (session_id, user_id, mode, recipe_id, json.dumps(state or {})),
            )
        return True
    except DatabaseError as exc:
        logger.warning("create_session failed: %s", exc)
        return False


def update_session_state(
    session_id: str,
    state: dict[str, Any],
    recipe_id: str | None = None,
) -> bool:
    if not memory_enabled() or not session_id:
        return False
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET state = %s::jsonb, recipe_id = COALESCE(%s, recipe_id) "
                "WHERE id = %s",
                (json.dumps(state or {}), recipe_id, session_id),
            )
        return True
    except DatabaseError as exc:
        logger.warning("update_session_state failed: %s", exc)
        return False


def get_session(session_id: str) -> dict[str, Any] | None:
    if not memory_enabled() or not session_id:
        return None
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, user_id, mode, recipe_id, state, started_at, ended_at "
                "FROM sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
    except DatabaseError as exc:
        logger.warning("get_session failed: %s", exc)
        return None
    if not row:
        return None
    return {
        "id": row[0],
        "user_id": row[1],
        "mode": row[2],
        "recipe_id": row[3],
        "state": row[4] or {},
        "started_at": row[5],
        "ended_at": row[6],
    }


def end_session(session_id: str) -> bool:
    if not memory_enabled() or not session_id:
        return False
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute("UPDATE sessions SET ended_at = NOW() WHERE id = %s", (session_id,))
        return True
    except DatabaseError as exc:
        logger.warning("end_session failed: %s", exc)
        return False


# --------------------------------------------------------------------------- #
# Conversation history
# --------------------------------------------------------------------------- #
def save_conversation_turn(
    user_id: str,
    session_id: str,
    role: str,
    content: str,
    tool_calls: list[dict[str, Any]] | None = None,
) -> bool:
    if not memory_enabled() or not user_id or not session_id:
        return False
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO conversations (user_id, session_id, role, content, tool_calls) "
                "VALUES (%s, %s, %s, %s, %s::jsonb)",
                (
                    user_id,
                    session_id,
                    role,
                    content or "",
                    json.dumps(tool_calls) if tool_calls else None,
                ),
            )
        return True
    except DatabaseError as exc:
        logger.warning("save_conversation_turn failed: %s", exc)
        return False


def get_recent_conversation(
    user_id: str, session_id: str, limit: int | None = None
) -> list[ConversationTurn]:
    if not memory_enabled() or not user_id:
        return []
    limit = limit or _env_int("MEMORY_RECENT_TURNS", 12)
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT role, content, created_at FROM conversations "
                "WHERE user_id = %s AND session_id = %s "
                "ORDER BY id DESC LIMIT %s",
                (user_id, session_id, limit),
            )
            rows = cur.fetchall()
    except DatabaseError as exc:
        logger.warning("get_recent_conversation failed: %s", exc)
        return []
    return [ConversationTurn(role=r[0], content=r[1], created_at=r[2]) for r in reversed(rows)]


# --------------------------------------------------------------------------- #
# Long-term memories
# --------------------------------------------------------------------------- #
def search_memories(
    user_id: str,
    query: str,
    top_k: int | None = None,
    threshold: float | None = None,
) -> list[MemoryHit]:
    if not memory_enabled() or not user_id or not query.strip():
        return []
    top_k = top_k or _env_int("MEMORY_TOP_K", 5)
    threshold = threshold if threshold is not None else _env_float("MEMORY_SEARCH_THRESHOLD", 0.55)
    vec = embed_for_query(query)
    if vec is None:
        return []
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, memory, memory_type, importance, updated_at, "
                "       1 - (embedding <=> %s) AS similarity "
                "FROM memories "
                "WHERE user_id = %s AND embedding IS NOT NULL "
                "ORDER BY embedding <=> %s LIMIT %s",
                (Vector(vec), user_id, Vector(vec), top_k),
            )
            rows = cur.fetchall()
    except DatabaseError as exc:
        logger.warning("search_memories failed: %s", exc)
        return []
    hits = [
        MemoryHit(
            id=r[0], memory=r[1], memory_type=r[2], importance=r[3],
            updated_at=r[4], similarity=float(r[5]),
        )
        for r in rows
    ]
    return [h for h in hits if h.similarity >= threshold]


def upsert_memory(
    user_id: str,
    memory: str,
    memory_type: str = "general",
    importance: float = 0.5,
) -> bool:
    """Insert a memory, or update a near-duplicate one (newer text wins)."""
    if not memory_enabled() or not user_id or not (memory or "").strip():
        return False
    memory = memory.strip()
    importance = max(0.0, min(1.0, float(importance or 0.5)))
    vec = embed_for_storage([memory])
    embedding = Vector(vec[0]) if vec else None
    dedup = _env_float("MEMORY_DEDUP_THRESHOLD", 0.85)
    try:
        with connection() as conn, conn.cursor() as cur:
            if embedding is not None:
                cur.execute(
                    "SELECT id, 1 - (embedding <=> %s) AS similarity FROM memories "
                    "WHERE user_id = %s AND embedding IS NOT NULL "
                    "ORDER BY embedding <=> %s LIMIT 1",
                    (embedding, user_id, embedding),
                )
                row = cur.fetchone()
                if row and float(row[1]) >= dedup:
                    cur.execute(
                        "UPDATE memories SET memory = %s, memory_type = %s, "
                        "importance = %s, embedding = %s, updated_at = NOW() WHERE id = %s",
                        (memory, memory_type, importance, embedding, row[0]),
                    )
                    return True
            cur.execute(
                "INSERT INTO memories (user_id, memory, memory_type, importance, embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                (user_id, memory, memory_type, importance, embedding),
            )
        return True
    except DatabaseError as exc:
        logger.warning("upsert_memory failed: %s", exc)
        return False


def list_memories(user_id: str, limit: int = 200) -> list[MemoryHit]:
    if not memory_enabled() or not user_id:
        return []
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, memory, memory_type, importance, updated_at "
                "FROM memories WHERE user_id = %s ORDER BY updated_at DESC LIMIT %s",
                (user_id, limit),
            )
            rows = cur.fetchall()
    except DatabaseError as exc:
        logger.warning("list_memories failed: %s", exc)
        return []
    return [
        MemoryHit(id=r[0], memory=r[1], memory_type=r[2], importance=r[3],
                  updated_at=r[4], similarity=1.0)
        for r in rows
    ]


def delete_memories(user_id: str) -> int:
    if not memory_enabled() or not user_id:
        return 0
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM memories WHERE user_id = %s", (user_id,))
            return cur.rowcount or 0
    except DatabaseError as exc:
        logger.warning("delete_memories failed: %s", exc)
        return 0


def list_important_memories(
    user_id: str, min_importance: float = 0.9, limit: int = 5
) -> list[MemoryHit]:
    """High-importance memories (allergies, restrictions) to always inject."""
    if not memory_enabled() or not user_id:
        return []
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT id, memory, memory_type, importance, updated_at "
                "FROM memories WHERE user_id = %s AND importance >= %s "
                "ORDER BY importance DESC, updated_at DESC LIMIT %s",
                (user_id, float(min_importance), limit),
            )
            rows = cur.fetchall()
    except DatabaseError as exc:
        logger.warning("list_important_memories failed: %s", exc)
        return []
    return [
        MemoryHit(id=r[0], memory=r[1], memory_type=r[2], importance=r[3],
                  updated_at=r[4], similarity=1.0)
        for r in rows
    ]


# --------------------------------------------------------------------------- #
# Recipes (RAG)
# --------------------------------------------------------------------------- #
def upsert_recipe(
    recipe_id: str,
    name: str,
    data: dict[str, Any],
    region: str = "",
    category: str = "",
    text: str = "",
) -> bool:
    if not memory_enabled() or not recipe_id:
        return False
    vec = embed_for_storage([text or name])
    embedding = Vector(vec[0]) if vec else None
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO recipes (id, name, region, category, data, embedding, updated_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, NOW())
                ON CONFLICT (id) DO UPDATE
                  SET name = EXCLUDED.name, region = EXCLUDED.region,
                      category = EXCLUDED.category, data = EXCLUDED.data,
                      embedding = EXCLUDED.embedding, updated_at = NOW()
                """,
                (recipe_id, name, region, category, json.dumps(data), embedding),
            )
        return True
    except DatabaseError as exc:
        logger.warning("upsert_recipe failed: %s", exc)
        return False


def count_recipes() -> int:
    if not memory_enabled():
        return 0
    try:
        with connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM recipes")
            return int(cur.fetchone()[0])
    except DatabaseError as exc:
        logger.warning("count_recipes failed: %s", exc)
        return 0


_RECIPE_STOPWORDS = {
    "something", "anything", "recipe", "recipes", "make", "cook", "with",
    "using", "want", "need", "have", "from", "that", "this", "what", "some",
    "quick", "easy", "good", "best", "dish", "dishes", "food",
}


def search_recipes(query: str, top_k: int = 5, threshold: float = 0.25) -> list[dict[str, Any]]:
    """Keyword + semantic recipe search.

    Keyword hits (ingredient/name matches) are ranked first, then
    semantically similar recipes fill the remaining slots.
    """
    if not memory_enabled() or not query.strip():
        return []
    import re as _re

    tokens = [
        t for t in _re.split(r"\W+", query.lower())
        if len(t) >= 4 and t not in _RECIPE_STOPWORDS
    ]
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    # 1) keyword pass on the raw recipe JSON (ingredients, name, category).
    #    Name matches weigh more than ingredient mentions.
    if tokens:
        score_parts = []
        params: list[Any] = []
        for t in tokens:
            score_parts.append("(CASE WHEN name ILIKE %s THEN 3 ELSE 0 END)")
            params.append(f"%{t}%")
            score_parts.append("(CASE WHEN data::text ILIKE %s THEN 1 ELSE 0 END)")
            params.append(f"%{t}%")
        score_expr = " + ".join(score_parts)
        sql = (
            "SELECT id, name, region, category, score FROM ("
            f"  SELECT id, name, region, category, ({score_expr}) AS score "
            "   FROM recipes"
            ") t WHERE score > 0 ORDER BY score DESC, name LIMIT %s"
        )
        try:
            with connection() as conn, conn.cursor() as cur:
                cur.execute(sql, params + [top_k])
                for r in cur.fetchall():
                    if r[0] not in seen:
                        seen.add(r[0])
                        results.append(
                            {"id": r[0], "name": r[1], "region": r[2],
                             "category": r[3], "similarity": 1.0, "match": "keyword"}
                        )
        except DatabaseError as exc:
            logger.warning("search_recipes keyword pass failed: %s", exc)

    # 2) semantic pass
    vec = embed_for_query(query)
    if vec is not None:
        try:
            with connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, region, category, 1 - (embedding <=> %s) AS similarity "
                    "FROM recipes WHERE embedding IS NOT NULL "
                    "ORDER BY embedding <=> %s LIMIT %s",
                    (Vector(vec), Vector(vec), top_k),
                )
                for r in cur.fetchall():
                    if r[0] in seen:
                        continue
                    sim = float(r[4])
                    if sim < threshold:
                        continue
                    seen.add(r[0])
                    results.append(
                        {"id": r[0], "name": r[1], "region": r[2],
                         "category": r[3], "similarity": round(sim, 4),
                         "match": "semantic"}
                    )
        except DatabaseError as exc:
            logger.warning("search_recipes semantic pass failed: %s", exc)

    return results[:top_k]


__all__ = [
    "MemoryHit",
    "ConversationTurn",
    "memory_enabled",
    "create_session",
    "update_session_state",
    "get_session",
    "end_session",
    "save_conversation_turn",
    "get_recent_conversation",
    "search_memories",
    "upsert_memory",
    "list_memories",
    "list_important_memories",
    "delete_memories",
    "upsert_recipe",
    "count_recipes",
    "search_recipes",
]
