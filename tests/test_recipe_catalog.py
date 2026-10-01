"""Tests for the recipe catalog, adapter, search and servings scaling."""
from __future__ import annotations

import json

import pytest

from voice_agent_functions import recipe_catalog as rc
from voice_agent_functions.recipe_engine import normalize_recipe, heat_phrase
from voice_agent_functions.state import RecipeContext, VoiceSession
from voice_agent_functions.cooking_tools import load_recipe, set_servings
from voice_agent_functions.dispatcher import dispatch


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def catalog() -> rc.RecipeCatalog:
    cat = rc.RecipeCatalog()
    n = cat.load()
    if n == 0:
        pytest.skip("data/recipes.json not present")
    return cat


# ---------------------------------------------------------------- loading
def test_catalog_loads_200(catalog):
    assert len(catalog) == 200


def test_catalog_ids_unique(catalog):
    ids = [r["id"] for r in catalog.list_recipes()]
    assert len(ids) == len(set(ids))


def test_adapt_maps_fields(catalog):
    entry = catalog.get("SI001")
    assert entry is not None
    rec = entry.recipe
    assert rec["recipeName"].startswith("Pulihora")
    assert rec["utensils"] and all(isinstance(u, str) for u in rec["utensils"])
    assert rec["steps"]
    # heat phrase is reverse-mapped into a canonical level
    step0 = rec["stepDetails"]["0"]
    assert step0["heat_level_spoken_telugu"]
    parsed = normalize_recipe(rec)
    assert parsed is not None
    assert parsed.steps[0].heat_level in ("low", "medium-low", "medium", "medium-high", "high")
    assert heat_phrase(parsed.steps[0].heat_level) == step0["heat_level_spoken_telugu"]


# ---------------------------------------------------------------- search
def test_search_by_id(catalog):
    matches = catalog.search("SI001")
    assert matches and matches[0][0].id == "SI001"


def test_search_exact_name(catalog):
    res = rc.resolve("Idli")
    assert res["status"] == "loaded"
    assert res["entry"].id == "SI041"


def test_search_ambiguous_pulihora(catalog):
    res = rc.resolve("Pulihora")
    assert res["status"] == "ambiguous"
    assert len(res["candidates"]) >= 3


def test_search_none(catalog):
    assert rc.resolve("zzzznotarecipe")["status"] == "none"


def test_search_fuzzy_token(catalog):
    res = rc.resolve("lemon rice")
    assert res["status"] in ("loaded", "ambiguous")
    names = " ".join(c.name for c in res["candidates"])
    assert "Lemon" in names or "lemon" in names.lower()


# ---------------------------------------------------------------- scaling
def test_scaling_multiplies_quantities(catalog):
    entry = catalog.get("SI001")
    recipe = normalize_recipe(entry.recipe)
    rice = next(i for i in recipe.ingredients if i.name == "cooked rice")
    base_qty = rice.quantity
    recipe.scale_to(4)
    assert rice.quantity == pytest.approx(base_qty * 4)
    # scaling twice to the same target is idempotent
    recipe.scale_to(4)
    assert rice.quantity == pytest.approx(base_qty * 4)
    # scaling back to 2 resets correctly
    recipe.scale_to(2)
    assert rice.quantity == pytest.approx(base_qty * 2)


def test_scale_ignores_string_quantities():
    from voice_agent_functions.recipe_engine import Ingredient, Recipe, RecipeStep
    ing = Ingredient(name="salt", quantity="to taste")
    recipe = Recipe(name="x", steps=[RecipeStep(0, "add salt", ingredients=[ing])], ingredients=[ing])
    recipe.scale_to(4)
    assert ing.quantity == "to taste"


# ---------------------------------------------------------------- tools
def test_load_recipe_tool_sets_recipe():
    session = VoiceSession(session_id="c", mode="cook")
    result = load_recipe(session, "Idli")
    assert result["type"] == "recipe_loaded"
    assert session.recipe is not None
    assert session.recipe.recipe_name.startswith("Idli")


def test_load_recipe_ambiguous_returns_candidates():
    session = VoiceSession(session_id="c", mode="cook")
    result = load_recipe(session, "Pulihora")
    assert result["type"] == "recipe_ambiguous"
    assert len(result["candidates"]) >= 3
    assert session.recipe is None


def test_load_recipe_not_found(monkeypatch):
    # Disable the online fallback so this tests the catalog-miss path only.
    monkeypatch.setenv("ONLINE_RECIPE", "0")
    session = VoiceSession(session_id="c", mode="cook")
    result = load_recipe(session, "zzzznotarecipe")
    assert result["type"] == "recipe_not_found"


def test_set_servings_tool_scales(catalog):
    session = VoiceSession(session_id="c", mode="cook")
    load_recipe(session, "Idli")
    ing = next(
        i for i in session.recipe.structured.ingredients
        if isinstance(i.quantity, (int, float))
    )
    before = ing.quantity
    result = set_servings(session, 3)
    assert result["type"] == "servings_set"
    assert result["servings"] == 3
    assert ing.quantity == pytest.approx(before * 3)


def test_dispatch_load_and_servings():
    session = VoiceSession(session_id="c", mode="cook")
    r1 = dispatch(session, "load_recipe", {"name": "Sambar"})
    assert "error" not in r1
    r2 = dispatch(session, "set_servings", {"servings": 2})
    assert "error" not in r2
    assert session.recipe.structured.target_servings == 2
