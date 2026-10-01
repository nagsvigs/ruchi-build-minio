"""Recipe catalog — loads the bundled South-Indian dataset and resolves
dishes by name.

The dataset (``data/recipes.json``, 200 recipes) is already structured:
``ingredients`` with exact quantities, English ``utensils``, and per-step
``instruction`` / ``heat_level_spoken_telugu`` / ``duration_minutes`` /
``requires_close_attention`` / ``can_parallelize``. This module adapts
those records into the dict shape :mod:`recipe_engine` understands and
provides fuzzy name lookup so Ruchi can load a dish by voice.

Set ``RECIPES_PATH`` to point at a different dataset file.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("ruchi.recipe_catalog")

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
DEFAULT_PATH = _PROJECT_ROOT / "data" / "recipes.json"

_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)


def _norm(text: str) -> str:
    """Normalize a name for matching: lowercase, dashes/punctuation → space."""
    if not text:
        return ""
    s = str(text).lower()
    s = s.replace("–", " ").replace("—", " ").replace("-", " ")
    s = _PUNCT_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(text: str) -> set[str]:
    return {t for t in _norm(text).split() if len(t) >= 3}


@dataclass
class CatalogEntry:
    id: str
    name: str
    recipe: dict[str, Any]           # adapted for recipe_engine.normalize_recipe
    region: str = ""
    category: str = ""
    aliases: list[str] = field(default_factory=list)

    @property
    def norm_names(self) -> list[str]:
        return [_norm(a) for a in self.aliases if a]


class RecipeCatalog:
    """In-memory recipe catalog with fuzzy name search."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or os.getenv("RECIPES_PATH") or DEFAULT_PATH)
        self._entries: dict[str, CatalogEntry] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- load
    def load(self) -> int:
        """(Re)load the dataset. Returns the number of recipes loaded."""
        with self._lock:
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                logger.warning("recipe catalog not found at %s", self.path)
                self._entries, self._order = {}, []
                return 0
            except (OSError, ValueError) as exc:
                logger.error("could not read recipe catalog %s: %s", self.path, exc)
                return len(self._entries)
            records = raw.get("recipes") if isinstance(raw, dict) else raw
            entries: dict[str, CatalogEntry] = {}
            order: list[str] = []
            for rec in records or []:
                try:
                    entry = self._adapt(rec)
                except Exception as exc:  # noqa: BLE001 - skip bad record
                    logger.warning("skipping malformed recipe: %s", exc)
                    continue
                if entry and entry.id not in entries:
                    entries[entry.id] = entry
                    order.append(entry.id)
            self._entries, self._order = entries, order
            logger.info("loaded %d recipes from %s", len(entries), self.path)
            return len(entries)

    def _adapt(self, rec: dict[str, Any]) -> Optional[CatalogEntry]:
        if not isinstance(rec, dict) or not rec.get("name") or not rec.get("steps"):
            return None
        steps: list[str] = []
        details: dict[str, Any] = {}
        for i, s in enumerate(rec.get("steps") or []):
            if isinstance(s, dict):
                steps.append(str(s.get("instruction") or s.get("title") or "").strip())
                details[str(i)] = {
                    "heat_level_spoken_telugu": s.get("heat_level_spoken_telugu"),
                    "heat_level_internal": s.get("heat_level_internal"),
                    "duration_minutes": s.get("duration_minutes"),
                    "requires_close_attention": s.get("requires_close_attention"),
                    "can_parallelize": s.get("can_parallelize"),
                    "utensils": s.get("utensils") or [],
                }
            else:
                steps.append(str(s).strip())
        name = str(rec["name"]).strip()
        base = re.split(r"\s[–—-]\s", name)[0].strip()
        aliases = [name]
        if base and base != name:
            aliases.append(base)
        recipe = {
            "recipeName": name,
            "servings": rec.get("servings", 1),
            "ingredients": rec.get("ingredients") or [],
            "utensils": rec.get("utensils") or [],
            "steps": steps,
            "stepDetails": details,
        }
        return CatalogEntry(
            id=str(rec.get("id") or name),
            name=name,
            recipe=recipe,
            region=str(rec.get("region") or ""),
            category=str(rec.get("category") or ""),
            aliases=aliases,
        )

    # ---------------------------------------------------------------- read
    def __len__(self) -> int:
        return len(self._entries)

    def get(self, recipe_id: str) -> Optional[CatalogEntry]:
        return self._entries.get(str(recipe_id))

    def list_recipes(self) -> list[dict[str, Any]]:
        return [
            {
                "id": e.id,
                "name": e.name,
                "region": e.region,
                "category": e.category,
                "servings": e.recipe.get("servings", 1),
                "steps": len(e.recipe.get("steps", [])),
            }
            for e in (self._entries[i] for i in self._order)
        ]

    # ---------------------------------------------------------------- search
    def _score(self, q: str, q_tokens: set[str], entry: CatalogEntry) -> float:
        if not q:
            return 0.0
        best = 0.0
        for alias in entry.norm_names:
            if alias == q:
                return 1.0
            a_tokens = _tokens(alias)
            if q in a_tokens or alias.startswith(q) or q.startswith(alias):
                best = max(best, 0.9)
            elif q in alias:
                best = max(best, 0.6)
            if q_tokens and a_tokens:
                overlap = len(q_tokens & a_tokens) / len(q_tokens | a_tokens)
                best = max(best, 0.4 + 0.5 * overlap)
        return best

    def search(self, query: str, limit: int = 5) -> list[tuple[CatalogEntry, float]]:
        """Return ``(entry, score)`` ranked best-first."""
        q = _norm(query)
        if not q:
            return []
        # exact id match wins
        by_id = self.get(str(query).strip().upper())
        if by_id is not None:
            return [(by_id, 1.0)]
        q_tokens = _tokens(query)
        scored = [
            (e, self._score(q, q_tokens, e))
            for e in (self._entries[i] for i in self._order)
        ]
        scored = [s for s in scored if s[1] > 0.5]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]


_catalog: Optional[RecipeCatalog] = None
_catalog_lock = threading.Lock()


def get_catalog() -> RecipeCatalog:
    """Return the process-wide catalog, loading it on first use."""
    global _catalog
    if _catalog is None:
        with _catalog_lock:
            if _catalog is None:
                cat = RecipeCatalog()
                cat.load()
                _catalog = cat
    return _catalog


def resolve(query: str, limit: int = 5) -> dict[str, Any]:
    """Resolve a spoken dish name into a load/ambiguous/none decision."""
    matches = get_catalog().search(query, limit=limit)
    if not matches:
        return {"status": "none", "query": query, "candidates": []}
    top, top_score = matches[0]
    second = matches[1][1] if len(matches) > 1 else 0.0
    # Keep only plausible candidates (within a small margin of the best).
    close = [e for e, s in matches if s >= top_score - 0.15]
    if top_score >= 0.95 and top_score - second > 0.05:
        return {"status": "loaded", "query": query, "entry": top, "candidates": [top]}
    if top_score >= 0.9 and top_score - second >= 0.15:
        return {"status": "loaded", "query": query, "entry": top, "candidates": [top]}
    return {
        "status": "ambiguous",
        "query": query,
        "entry": top,
        "candidates": close,
    }


__all__ = ["RecipeCatalog", "CatalogEntry", "get_catalog", "resolve", "DEFAULT_PATH"]
