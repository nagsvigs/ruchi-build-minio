"""Tests for the persistent memory layer (Postgres + pgvector).

DB tests are skipped automatically when ``DATABASE_URL`` is unset or the
database is unreachable, so the suite still passes in a bare checkout.
"""
from __future__ import annotations

import os
import uuid

import pytest

# Load .env so the local dev DATABASE_URL is picked up when present.
try:  # pragma: no cover - convenience only
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

from voice_agent_functions.memory import database, manager
from voice_agent_functions.memory import embeddings as mem_embeddings
from voice_agent_functions.memory import extractor


def _db_ready() -> bool:
    if not os.getenv("DATABASE_URL") or not manager.memory_enabled():
        return False
    try:
        database.bootstrap_schema()
        return True
    except Exception:
        return False


requires_db = pytest.mark.skipif(not _db_ready(), reason="Postgres not available")


@pytest.fixture(scope="module")
def user_id() -> str:
    uid = f"pytest_{uuid.uuid4().hex[:8]}"
    yield uid
    try:
        manager.delete_memories(uid)
    except Exception:
        pass


# ---------------------------------------------------------------- embeddings
def test_embedding_model_configured():
    assert mem_embeddings.configured_dimensions() == 384
    assert "multilingual" in mem_embeddings.model_name()


def test_embedding_never_raises(monkeypatch):
    # Force the model loader to fail; the helpers must swallow it.
    monkeypatch.setattr(mem_embeddings, "_model", None)
    monkeypatch.setenv("EMBEDDING_MODEL", "definitely/not-a-model")
    assert mem_embeddings.embed_for_query("hello") is None
    assert mem_embeddings.embed_for_storage(["hello"]) == []
    monkeypatch.setattr(mem_embeddings, "_model", None)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)


# ---------------------------------------------------------------- extractor
def test_extractor_prefilter_skips_trivial():
    assert extractor._trivial("సరే")
    assert extractor._trivial("okay")
    assert extractor._trivial("next")
    assert not extractor._trivial("నాకు వేరుశనగ అలర్జీ ఉంది")


@pytest.mark.asyncio
async def test_extractor_disabled_without_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    result = await extractor.extract_memories("I am allergic to peanuts")
    assert result.is_noop


# ---------------------------------------------------------------- database
@requires_db
def test_bootstrap_is_idempotent():
    database.bootstrap_schema()
    database.bootstrap_schema()
    assert database.vector_dimensions() == 384


@requires_db
def test_session_round_trip(user_id):
    sid = f"s_{uuid.uuid4().hex[:8]}"
    assert manager.create_session(
        user_id, sid, mode="cook", recipe_id="SI041", state={"a": 1}
    )
    row = manager.get_session(sid)
    assert row and row["user_id"] == user_id and row["mode"] == "cook"
    assert manager.update_session_state(sid, {"a": 2}, recipe_id="SI041")
    assert manager.get_session(sid)["state"] == {"a": 2}
    manager.end_session(sid)


@requires_db
def test_conversation_history(user_id):
    sid = f"s_{uuid.uuid4().hex[:8]}"
    manager.create_session(user_id, sid)
    manager.save_conversation_turn(user_id, sid, "user", "నాకు ఇడ్లీ")
    manager.save_conversation_turn(user_id, sid, "assistant", "సరే")
    turns = manager.get_recent_conversation(user_id, sid)
    assert [t.role for t in turns] == ["user", "assistant"]
    assert turns[0].content == "నాకు ఇడ్లీ"


@requires_db
def test_memory_upsert_search_dedup(user_id):
    assert manager.upsert_memory(user_id, "User prefers less spicy food", "preference", 0.8)
    hits = manager.search_memories(user_id, "make it spicy")
    assert any("less spicy" in h.memory for h in hits)

    # A near-duplicate updates rather than inserting a new row.
    before = len(manager.list_memories(user_id))
    manager.upsert_memory(user_id, "User prefers mildly spicy food", "preference", 0.8)
    after = manager.list_memories(user_id)
    assert len(after) == before  # deduped

    assert manager.delete_memories(user_id) >= 1
    assert manager.list_memories(user_id) == []


@requires_db
def test_recipe_search_keyword_and_semantic():
    if manager.count_recipes() == 0:
        pytest.skip("recipe catalog not synced")
    hits = manager.search_recipes("paneer biryani", top_k=3)
    assert hits and "Paneer" in hits[0]["name"]


@requires_db
def test_memory_disabled_returns_defaults(monkeypatch):
    monkeypatch.setenv("MEMORY_ENABLED", "0")
    assert not manager.memory_enabled()
    assert manager.list_memories("anyone") == []
    assert manager.search_recipes("paneer") == []
