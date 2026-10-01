"""Tests that Gemini 'thought' parts never leak into Ruchi's speech."""
from __future__ import annotations

import asyncio

from voice_agent_functions.runner import GeminiFunctionCallingRunner, RunnerConfig
from voice_agent_functions.state import VoiceSession


def _run(coro):
    return asyncio.run(coro)


def _hist(text):
    return [{"role": "user", "parts": [{"text": text}]}]


def test_situation_hint_healthy_vague():
    s = VoiceSession(session_id="h1", mode="cook")
    hint = GeminiFunctionCallingRunner._situation_hint(s, _hist("I want something healthy"))
    assert "health goal" in hint.lower()
    assert "MUST be a single question" in hint


def test_situation_hint_healthy_specific_is_empty():
    s = VoiceSession(session_id="h2", mode="cook")
    assert GeminiFunctionCallingRunner._situation_hint(s, _hist("protein rich food")) == ""


def test_situation_hint_discovery_asks_ingredients():
    s = VoiceSession(session_id="h3", mode="cook")
    hint = GeminiFunctionCallingRunner._situation_hint(s, _hist("I don't know what to do"))
    assert "ingredient" in hint.lower()


def test_situation_hint_skipped_when_recipe_loaded():
    from voice_agent_functions.state import RecipeContext

    s = VoiceSession(
        session_id="h4", mode="cook",
        recipe=RecipeContext(recipe_name="Idli", steps=["a", "b"]),
    )
    assert GeminiFunctionCallingRunner._situation_hint(s, _hist("I want something healthy")) == ""


def test_thought_parts_are_excluded_from_speech():
    session = VoiceSession(session_id="t1", mode="cook")
    runner = GeminiFunctionCallingRunner(RunnerConfig(api_key="test-key"))

    async def fake_call(payload):
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "The user wants paneer biryani, I should ask quantity.",
                                "thought": True,
                            },
                            {"text": "[warm] పన్నీర్ బిర్యానీ చాలా బాగుంటుంది!"},
                        ]
                    }
                }
            ]
        }

    runner._call_gemini = fake_call  # type: ignore[method-assign]
    turn = _run(runner.run_turn(session, "paneer biryani"))
    assert "The user wants" not in turn.assistant_text
    assert "పన్నీర్ బిర్యానీ" in turn.assistant_text


def test_thought_only_response_retries():
    session = VoiceSession(session_id="t2", mode="cook")
    runner = GeminiFunctionCallingRunner(RunnerConfig(api_key="test-key"))
    calls = {"n": 0}

    async def fake_call(payload):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "candidates": [
                    {"content": {"parts": [{"text": "private reasoning", "thought": True}]}}
                ]
            }
        return {
            "candidates": [{"content": {"parts": [{"text": "[warm] సరే!"}]}}]
        }

    runner._call_gemini = fake_call  # type: ignore[method-assign]
    turn = _run(runner.run_turn(session, "hi"))
    assert turn.assistant_text == "[warm] సరే!"
    assert calls["n"] == 2
