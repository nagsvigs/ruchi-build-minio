"""Unit tests for recipe_engine (heat, utensils, quantities, parallel)."""
from __future__ import annotations

import pytest

from voice_agent_functions import recipe_engine as re_mod
from voice_agent_functions.recipe_engine import (
    Ingredient,
    format_quantity,
    heat_phrase,
    infer_heat_level,
    normalize_recipe,
    normalize_utensil,
    parse_duration_seconds,
    parse_ingredient,
    requires_attention,
    spoken_ingredient_list,
    find_parallel_step,
)
from voice_agent_functions.server_voice import format_for_tts


# ---------------------------------------------------------------- heat
def test_heat_phrase_is_telugu():
    assert heat_phrase("low") == "లో ఫ్లేమ్‌లో"
    assert heat_phrase("medium") == "మీడియం ఫ్లేమ్‌లో"
    assert heat_phrase("medium-low") == "మీడియం-లో ఫ్లేమ్‌లో"
    assert heat_phrase("medium-high") == "మీడియం-హై ఫ్లేమ్‌లో"
    assert heat_phrase("high") == "హై ఫ్లేమ్‌లో"
    assert heat_phrase(None) is None


def test_infer_heat_level():
    assert infer_heat_level("Bring the water to a boil") == "high"
    assert infer_heat_level("Simmer for 10 minutes") == "low"
    assert infer_heat_level("Sauté the onions until golden brown") == "medium"
    assert infer_heat_level("Cook on medium-high heat") == "medium-high"
    assert infer_heat_level("Garnish with coriander") is None


def test_requires_attention():
    assert requires_attention("Splutter mustard seeds in hot oil")
    assert requires_attention("Temper the cumin seeds")
    assert not requires_attention("Soak the rice for 30 minutes")


# ---------------------------------------------------------------- utensils
def test_utensils_normalized_to_english():
    assert normalize_utensil("kadai") == "Pan"
    assert normalize_utensil("tawa") == "Griddle"
    assert normalize_utensil("handi") == "Pot"
    assert normalize_utensil("karahi") == "Wok"
    assert normalize_utensil("pressure cooker") == "Pressure cooker"
    assert normalize_utensil("కడాయి") == "Pan"
    assert normalize_utensil("గిన్నె") == "Pot"


# ---------------------------------------------------------------- quantities
def test_format_quantity_fractions():
    assert format_quantity(0.5) == "1/2"
    assert format_quantity(0.25) == "1/4"
    assert format_quantity(0.75) == "3/4"
    assert format_quantity(1) == "1"
    assert format_quantity(2) == "2"
    assert format_quantity(1.5) == "1 1/2"
    assert format_quantity("to taste") == "to taste"


def test_parse_ingredient_quantity_unit():
    ing = parse_ingredient("1/2 tablespoon oil")
    assert ing.quantity == pytest.approx(0.5)
    assert ing.unit == "tablespoon"
    assert ing.name == "oil"
    assert ing.spoken() == "1/2 tablespoon oil"


def test_parse_ingredient_prep_preserved():
    ing = parse_ingredient("2 finely chopped onions")
    assert ing.quantity == pytest.approx(2)
    assert ing.prep == "finely chopped"
    assert ing.name == "onions"
    assert ing.spoken() == "2 finely chopped onions"


def test_parse_ingredient_no_quantity():
    ing = parse_ingredient("salt to taste")
    assert ing.quantity is None
    assert "salt" in ing.name


def test_ingredient_from_dict():
    ing = Ingredient.from_any({"name": "salt", "quantity": 1, "unit": "pinch"})
    assert ing.spoken() == "1 pinch salt"


def test_spoken_ingredient_list():
    ings = [
        Ingredient(name="oil", quantity=0.5, unit="tablespoon"),
        Ingredient(name="cumin seeds", quantity=1, unit="teaspoon"),
    ]
    assert spoken_ingredient_list(ings) == "1/2 tablespoon oil, and 1 teaspoon cumin seeds"


# ---------------------------------------------------------------- durations
def test_parse_duration_seconds():
    assert parse_duration_seconds("Soak the rice for 30 minutes.") == 1800
    assert parse_duration_seconds("Cook for 1 hour") == 3600
    assert parse_duration_seconds("ఒక 5 నిమిషాలు వేయించండి") == 300
    assert parse_duration_seconds("no time here") is None


# ---------------------------------------------------------------- recipe
def _sample_recipe():
    return {
        "recipeName": "Test Curry",
        "steps": [
            "Heat 1/2 tablespoon oil in a pan on medium heat.",
            "Cook the onions in the pan for 10 minutes until golden brown.",
            "Boil 2 cups water in a pot on high heat.",
            "Add tomatoes to the onions and cook for 5 minutes.",
        ],
        "ingredients": [
            {"name": "oil", "quantity": 0.5, "unit": "tablespoon"},
            {"name": "onions", "quantity": 2, "prep": "finely chopped"},
            {"name": "water", "quantity": 2, "unit": "cups"},
        ],
        "utensils": ["kadai", "pot"],
    }


def test_normalize_recipe_structure():
    recipe = normalize_recipe(_sample_recipe())
    assert recipe is not None
    assert len(recipe.steps) == 4
    assert "Pan" in recipe.utensils
    assert "Pot" in recipe.utensils
    # duration parsed on step 1
    assert recipe.steps[1].duration_seconds == 600
    # heat inferred
    assert recipe.steps[2].heat_level == "high"
    # ingredients attached by name
    names = [i.name for i in recipe.steps[1].ingredients]
    assert "onions" in names


def test_find_parallel_step_independent():
    recipe = normalize_recipe(_sample_recipe())
    # step 1 (onions, 10 min, Pan) → step 2 (boil water, Pot) is safe.
    idx = find_parallel_step(recipe, 1, set())
    assert idx == 2


def test_find_parallel_step_skips_dependent():
    recipe = normalize_recipe(
        {
            "recipeName": "Dep",
            "steps": [
                "Cook onions in a pan on medium heat for 10 minutes.",
                "Add tomatoes to the onions and cook 5 minutes.",
            ],
        }
    )
    assert find_parallel_step(recipe, 0, set()) is None


def test_find_parallel_step_respects_completed():
    recipe = normalize_recipe(_sample_recipe())
    # if the water step is already done, there is no other safe candidate
    assert find_parallel_step(recipe, 1, {2}) is None


# ---------------------------------------------------------------- TTS lists
def test_format_for_tts_bullets():
    text = "Utensils:\n- Pan\n- Bowl\n- Knife"
    out = format_for_tts(text)
    assert "…" in out
    assert out.count("…") == 2


def test_format_for_tts_normal_text_unchanged():
    text = "Cook the onions for 5 minutes."
    assert format_for_tts(text) == text


def test_strip_emotion_tags():
    from voice_agent_functions.server_voice import strip_emotion_tags

    assert strip_emotion_tags("[warm] హాయ్") == "హాయ్"
    assert strip_emotion_tags("[warm, curious] ఏం వండుదాం?") == "ఏం వండుదాం?"
    assert strip_emotion_tags("[laughs] బాగుంది!") == "బాగుంది!"
    assert strip_emotion_tags("no tags here") == "no tags here"
    assert strip_emotion_tags("") == ""

