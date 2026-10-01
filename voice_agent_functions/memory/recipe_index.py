"""Recipe RAG — mirror the JSON catalog into Postgres for semantic search.

Runs at startup in a background thread (embedding 200 recipes takes a
few seconds). If the DB or embeddings are unavailable it simply does
nothing.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from . import manager

logger = logging.getLogger("ruchi.memory.recipes")


def _recipe_text(entry) -> str:
    rec = entry.recipe or {}
    names = [
        str(i.get("name", "")).strip()
        for i in (rec.get("ingredients") or [])
        if isinstance(i, dict) and i.get("name")
    ]
    parts = [entry.name]
    if entry.category:
        parts.append(f"category: {entry.category}")
    if entry.region:
        parts.append(f"region: {entry.region}")
    if names:
        parts.append("ingredients: " + ", ".join(names))
    return ". ".join(parts)


def sync_from_catalog() -> int:
    """Upsert every catalog recipe (with its embedding) into Postgres."""
    if not manager.memory_enabled():
        return 0
    try:
        from .. import recipe_catalog

        catalog = recipe_catalog.get_catalog()
    except Exception as exc:  # noqa: BLE001
        logger.warning("recipe catalog unavailable: %s", exc)
        return 0
    count = 0
    for item in catalog.list_recipes():
        entry = catalog.get(item["id"])
        if entry is None:
            continue
        if manager.upsert_recipe(
            entry.id,
            entry.name,
            entry.recipe,
            region=entry.region,
            category=entry.category,
            text=_recipe_text(entry),
        ):
            count += 1
    logger.info("synced %d recipes to Postgres", count)
    return count


def sync_async() -> None:
    """Kick off the sync in a daemon thread (startup must not block)."""
    threading.Thread(target=sync_from_catalog, name="recipe-sync", daemon=True).start()


def suggest(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Semantic recipe search, e.g. 'something with paneer and rice'."""
    return manager.search_recipes(query, top_k=top_k)


__all__ = ["sync_from_catalog", "sync_async", "suggest"]
