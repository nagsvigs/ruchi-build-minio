"""Tests for the Ruchi system prompt (master brain prompt)."""
from __future__ import annotations

from voice_agent_functions.prompts_loader import load_instructions


def test_cooking_prompt_loads():
    p = load_instructions("cooking")
    assert len(p) > 1500


def test_prompt_enforces_telugu_only():
    p = load_instructions("cooking")
    assert "ALWAYS speaks ONLY Telugu" in p
    # must not tell the model to mirror the user's language
    assert "respond in English" not in p


def test_prompt_has_fixed_greeting_rule():
    p = load_instructions("cooking")
    assert "హాయ్, నేను రుచి, మీ వంటింటి సఖిని" in p
    assert "do not repeat" in p.lower()


def test_prompt_has_telugu_heat_phrases():
    p = load_instructions("cooking")
    for phrase in (
        "లో ఫ్లేమ్‌లో",
        "మీడియం-లో ఫ్లేమ్‌లో",
        "మీడియం ఫ్లేమ్‌లో",
        "మీడియం-హై ఫ్లేమ్‌లో",
        "హై ఫ్లేమ్‌లో",
    ):
        assert phrase in p


def test_prompt_lists_real_tools():
    p = load_instructions("cooking")
    for tool in (
        "load_recipe", "fetch_recipe_online", "set_servings", "suggest_recipes",
        "next_step", "start_timer", "mark_parallel_complete",
        "report_issue", "finish_cooking",
    ):
        assert tool in p


def test_prompt_has_emotion_tags_and_safety():
    p = load_instructions("cooking")
    assert "[warm]" in p
    assert "SAFETY" in p.upper()
