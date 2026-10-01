"""Unit tests for the voice_agent_function_calls tools.

Run with::

    python -m pytest tests

The tests don't hit the network — they exercise the dispatcher +
session state directly so the whole tool surface can be validated
without a Gemini API key.
"""
from __future__ import annotations

import asyncio

import pytest

from voice_agent_functions.cooking_tools import (
    ISSUE_CATEGORIES,
    ISSUE_LEAD_INS,
    cancel_timer,
    finish_cooking,
    goto_step,
    mark_parallel_complete,
    next_step,
    pause,
    previous_step,
    report_issue,
    resolve_issue,
    resume,
    start_timer,
)
from voice_agent_functions.dispatcher import dispatch, dispatch_function_call, known_tool_names
from voice_agent_functions.intake_tools import finish_interview, save_answer
from voice_agent_functions.state import INTAKE_FIELDS, RecipeContext, VoiceSession


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def intake_session() -> VoiceSession:
    return VoiceSession(session_id="i1", mode="intake")


@pytest.fixture
def cooking_session() -> VoiceSession:
    return VoiceSession(
        session_id="c1",
        mode="cook",
        recipe=RecipeContext(
            recipe_name="Paneer Biryani",
            steps=[
                "Soak the rice for 30 minutes.",
                "Marinate the paneer in yogurt and spices.",
                "Layer rice and paneer in a heavy pot.",
                "Cook on low heat for 20 minutes.",
            ],
        ),
    )


# ============================================================
# Intake
# ============================================================
class TestIntake:
    def test_first_save_advances_to_next_field(self, intake_session):
        result = save_answer(intake_session, "name", "Paneer Biryani")
        assert result["type"] == "field_saved"
        assert result["field"] == "name"
        assert result["next_field"] == "time"
        assert intake_session.current_field == "time"
        assert intake_session.answers == {"name": "Paneer Biryani"}

    def test_save_rejects_wrong_field(self, intake_session):
        with pytest.raises(ValueError):
            save_answer(intake_session, "time", "45 minutes")  # current is "name"

    def test_save_rejects_unknown_field(self, intake_session):
        with pytest.raises(ValueError):
            save_answer(intake_session, "made_up", "x")

    def test_walk_full_interview(self, intake_session):
        answers = {
            "name": "Paneer Biryani",
            "time": "45 minutes",
            "servings": "4",
            "level": "medium",
            "ingredients": "paneer, rice, spices",
            "utensils": "heavy pot",
            "tips": "",  # skipped
            "steps": "1. soak 2. cook",
        }
        for field, value in answers.items():
            result = save_answer(intake_session, field, value)
            assert result["type"] in ("field_saved",)
        assert intake_session.answers == answers

    def test_save_steps_triggers_finish_interview(self, intake_session):
        save_answer(intake_session, "name", "X")
        save_answer(intake_session, "time", "10")
        save_answer(intake_session, "servings", "2")
        save_answer(intake_session, "level", "easy")
        save_answer(intake_session, "ingredients", "x")
        save_answer(intake_session, "utensils", "pan")
        save_answer(intake_session, "tips", "")
        last = save_answer(intake_session, "steps", "1. cook")
        assert last["next_field"] is None
        assert "interview_complete" in last
        assert last["interview_complete"]["type"] == "interview_complete"
        assert last["interview_complete"]["answers"]["tips"] == ""
        assert intake_session.current_field == ""

    def test_skip_phrases_normalize_to_empty_tips(self, intake_session):
        save_answer(intake_session, "name", "X")
        save_answer(intake_session, "time", "10")
        save_answer(intake_session, "servings", "2")
        save_answer(intake_session, "level", "easy")
        save_answer(intake_session, "ingredients", "x")
        save_answer(intake_session, "utensils", "pan")
        # tips can be skipped
        result = save_answer(intake_session, "tips", "skip")
        assert result["value"] == ""

    def test_finish_interview_includes_all_fields(self, intake_session):
        intake_session.answers = {f: "x" for f in INTAKE_FIELDS}
        intake_session.answers["tips"] = ""
        event = finish_interview(intake_session)
        assert event["type"] == "interview_complete"
        assert set(event["answers"].keys()) == set(INTAKE_FIELDS)
        assert event["answers"]["tips"] == ""


# ============================================================
# Step navigation
# ============================================================
class TestStepNav:
    def test_next_step_increments_and_speaks_step(self, cooking_session):
        result = next_step(cooking_session)
        assert cooking_session.recipe.step_index == 1
        assert result["type"] == "step_changed"
        assert "Marinate the paneer" in result["phrase"]
        assert result["lead_in"] == "Okay, next step."

    def test_next_step_at_last_returns_no_op(self, cooking_session):
        cooking_session.recipe.step_index = 3
        result = next_step(cooking_session)
        assert result["type"] == "no_op"
        assert "last step" in result["phrase"]

    def test_previous_step_decrements(self, cooking_session):
        cooking_session.recipe.step_index = 2
        result = previous_step(cooking_session)
        assert cooking_session.recipe.step_index == 1
        assert result["lead_in"] == "Sure, here's the step before."

    def test_previous_step_at_first_returns_no_op(self, cooking_session):
        result = previous_step(cooking_session)
        assert result["type"] == "no_op"
        assert "first step" in result["phrase"]

    def test_goto_step_zero_indexed(self, cooking_session):
        result = goto_step(cooking_session, step=4)
        assert cooking_session.recipe.step_index == 3
        assert "step 4" in result["phrase"]
        assert "Cook on low heat" in result["phrase"]

    def test_goto_step_out_of_range(self, cooking_session):
        result = goto_step(cooking_session, step=99)
        assert result["type"] == "no_op"

    def test_goto_step_invalid_args(self, cooking_session):
        with pytest.raises(ValueError):
            goto_step(cooking_session, step=0)
        with pytest.raises(ValueError):
            goto_step(cooking_session, step=-1)

    def test_requires_recipe(self):
        session = VoiceSession(session_id="x", mode="cook")
        with pytest.raises(ValueError):
            next_step(session)


# ============================================================
# Timers
# ============================================================
class TestTimers:
    def test_start_timer_publishes_event(self, cooking_session):
        result = start_timer(cooking_session, seconds=1200, label="the rice")
        assert result["type"] == "timer_started"
        assert result["seconds"] == 1200
        assert result["label"] == "the rice"
        assert result["duration_phrase"] == "20 minutes"
        assert cooking_session.active_timer is not None

    def test_start_timer_rejects_non_positive(self, cooking_session):
        with pytest.raises(ValueError):
            start_timer(cooking_session, seconds=0, label="x")
        with pytest.raises(ValueError):
            start_timer(cooking_session, seconds=-5, label="x")

    def test_cancel_running_timer(self, cooking_session):
        start_timer(cooking_session, seconds=600, label="rice")
        result = cancel_timer(cooking_session)
        assert result["type"] == "timer_cancelled"
        assert cooking_session.active_timer is None

    def test_cancel_when_no_timer_returns_no_op(self, cooking_session):
        result = cancel_timer(cooking_session)
        assert result["type"] == "no_op"
        assert "no timer" in result["phrase"].lower()

    def test_start_timer_cancels_previous(self, cooking_session):
        start_timer(cooking_session, seconds=60, label="a")
        start_timer(cooking_session, seconds=120, label="b")
        assert cooking_session.active_timer.label == "b"


# ============================================================
# Pause / resume
# ============================================================
class TestPauseResume:
    def test_pause_sets_state_and_publishes(self, cooking_session):
        result = pause(cooking_session)
        assert cooking_session.paused is True
        assert result["type"] == "paused"
        assert result["phrase"] == "Okay, I'll wait."

    def test_resume_clears_state_and_replays_step(self, cooking_session):
        cooking_session.recipe.step_index = 1
        pause(cooking_session)
        result = resume(cooking_session)
        assert cooking_session.paused is False
        assert result["type"] == "resumed"
        assert "Marinate the paneer" in result["phrase"]


# ============================================================
# Fix my dish
# ============================================================
class TestFixMyDish:
    @pytest.mark.parametrize("category", list(ISSUE_CATEGORIES))
    def test_report_each_category(self, cooking_session, category):
        result = report_issue(cooking_session, issue=category, details="something")
        assert result["type"] == "issue_reported"
        assert result["issue"] == category
        assert result["lead_in"] == ISSUE_LEAD_INS[category]
        assert cooking_session.active_issue is not None

    def test_report_unknown_category_raises(self, cooking_session):
        with pytest.raises(ValueError):
            report_issue(cooking_session, issue="too_ugly", details="x")

    def test_resolve_issue_clears_state(self, cooking_session):
        report_issue(cooking_session, issue="salty", details="too much")
        result = resolve_issue(cooking_session)
        assert result["type"] == "issue_resolved"
        assert cooking_session.active_issue is None


# ============================================================
# Finish cooking
# ============================================================
class TestFinishCooking:
    def test_finish_cancels_timer(self, cooking_session):
        start_timer(cooking_session, seconds=60, label="x")
        result = finish_cooking(cooking_session)
        assert result["type"] == "cooking_finished"
        assert cooking_session.active_timer is None

    def test_finish_without_timer(self, cooking_session):
        result = finish_cooking(cooking_session)
        assert result["type"] == "cooking_finished"
        assert "Enjoy" in result["phrase"]


# ============================================================
# Dispatcher
# ============================================================
class TestDispatcher:
    def test_dispatch_returns_result(self, cooking_session):
        out = dispatch(cooking_session, "next_step", {})
        assert "result" in out
        assert out["name"] == "next_step"

    def test_dispatch_unknown_tool_returns_error(self, cooking_session):
        out = dispatch(cooking_session, "made_up_tool", {})
        assert "error" in out

    def test_dispatch_wrong_mode_returns_error(self, intake_session):
        # next_step is cook-only
        out = dispatch(intake_session, "next_step", {})
        assert "error" in out

    def test_dispatch_validation_error_returns_error(self, cooking_session):
        out = dispatch(cooking_session, "start_timer", {"seconds": -1, "label": "x"})
        assert "error" in out

    def test_dispatch_function_call_wrapper(self, cooking_session):
        out = dispatch_function_call(
            cooking_session, {"name": "goto_step", "args": {"step": 2}}
        )
        assert out["name"] == "goto_step"
        assert "result" in out

    def test_known_tool_names_per_mode(self):
        assert "save_answer" in known_tool_names("intake")
        assert "next_step" in known_tool_names("cook")
        assert "save_answer" not in known_tool_names("cook")
        with pytest.raises(ValueError):
            known_tool_names("unknown_mode")


# ============================================================
# Schema sanity
# ============================================================
class TestSchemas:
    def test_intake_schema_has_two_tools(self):
        from voice_agent_functions.tool_schemas import INTAKE_TOOL_SCHEMAS
        assert {t["name"] for t in INTAKE_TOOL_SCHEMAS} == {"save_answer", "finish_interview"}

    def test_cooking_schema_has_ten_tools(self):
        from voice_agent_functions.tool_schemas import COOKING_TOOL_SCHEMAS
        names = {t["name"] for t in COOKING_TOOL_SCHEMAS}
        assert names == {
            "load_recipe", "fetch_recipe_online", "set_servings", "suggest_recipes",
            "next_step", "previous_step", "goto_step", "mark_parallel_complete",
            "start_timer", "cancel_timer",
            "pause", "resume",
            "report_issue", "resolve_issue",
            "finish_cooking",
        }

    def test_gemini_tools_payload_shape(self):
        from voice_agent_functions.tool_schemas import gemini_tools
        cook = gemini_tools("cook")
        assert cook[0].get("functionDeclarations") is not None
        intake = gemini_tools("intake")
        assert intake[0].get("functionDeclarations") is not None


# ============================================================
# End-to-end mock runner
# ============================================================
class TestMockRunner:
    @pytest.mark.asyncio
    async def test_mock_intake_returns_canned(self, intake_session):
        from voice_agent_functions.mock_runner import run_mock_turn
        turn = await run_mock_turn(intake_session, "Paneer Biryani")
        assert turn.assistant_text  # non-empty

    @pytest.mark.asyncio
    async def test_mock_cook_next_step(self, cooking_session):
        from voice_agent_functions.mock_runner import run_mock_turn
        turn = await run_mock_turn(cooking_session, "okay, next")
        assert any(tc["name"] == "next_step" for tc in turn.tool_calls)
        assert cooking_session.recipe.step_index == 1

    @pytest.mark.asyncio
    async def test_mock_cook_timer_start(self, cooking_session):
        from voice_agent_functions.mock_runner import run_mock_turn
        turn = await run_mock_turn(cooking_session, "set a timer for 20 minutes")
        assert any(tc["name"] == "start_timer" for tc in turn.tool_calls)

    @pytest.mark.asyncio
    async def test_mock_cook_issue_report(self, cooking_session):
        from voice_agent_functions.mock_runner import run_mock_turn
        turn = await run_mock_turn(cooking_session, "this is too spicy")
        assert any(tc["name"] == "report_issue" for tc in turn.tool_calls)
        assert cooking_session.active_issue is not None

    @pytest.mark.asyncio
    async def test_mock_cook_pause(self, cooking_session):
        from voice_agent_functions.mock_runner import run_mock_turn
        turn = await run_mock_turn(cooking_session, "pause.")
        assert cooking_session.paused is True
        assert any(tc["name"] == "pause" for tc in turn.tool_calls)


# ============================================================
# Prompts loader
# ============================================================
class TestPrompts:
    def test_load_all_three_prompts(self):
        from voice_agent_functions.prompts_loader import load_instructions
        assert load_instructions("cooking").strip()
        assert load_instructions("intake").strip()
        assert load_instructions("cooking_companion").strip()


# ============================================================
# Transcript logging
# ============================================================
class TestTranscript:
    @pytest.fixture(autouse=True)
    def _tmp_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TRANSCRIPT_DIR", str(tmp_path))
        monkeypatch.setenv("TRANSCRIPT_ENABLED", "1")
        yield

    def test_log_turn_writes_markdown(self):
        from voice_agent_functions import transcript
        transcript.clear_transcript()
        transcript.log_turn(
            session_id="abcdef1234",
            mode="cook",
            user_text="next step please",
            assistant_text="Okay, next step.",
            tool_calls=[{"name": "next_step", "args": {}}],
            provider="test",
        )
        text = transcript.read_transcript()
        assert "Conversation Transcript" in text
        assert "next step please" in text
        assert "Okay, next step." in text
        assert "next_step()" in text
        assert "abcdef12" in text

    def test_log_turn_records_tool_args(self):
        from voice_agent_functions import transcript
        transcript.clear_transcript()
        transcript.log_turn(
            session_id="s1", mode="cook",
            user_text="set a timer",
            assistant_text="Timer set.",
            tool_calls=[{"name": "start_timer", "args": {"seconds": 600, "label": "rice"}}],
        )
        text = transcript.read_transcript()
        assert "start_timer" in text
        assert "600" in text
        assert "rice" in text

    def test_log_note(self):
        from voice_agent_functions import transcript
        transcript.clear_transcript()
        transcript.log_note("s1", "session started")
        assert "session started" in transcript.read_transcript()

    def test_files_listed(self):
        from voice_agent_functions import transcript
        transcript.clear_transcript()
        transcript.log_turn(session_id="s1", mode="cook",
                            user_text="hi", assistant_text="hello")
        files = transcript.transcript_files()
        names = {f["name"] for f in files}
        assert "conversation.md" in names
        assert any(n.endswith(".md") and n != "conversation.md" for n in names)

    def test_clear(self):
        from voice_agent_functions import transcript
        transcript.log_turn(session_id="s1", mode="cook",
                            user_text="hi", assistant_text="hello")
        transcript.clear_transcript()
        assert transcript.read_transcript() == ""

    def test_disabled_via_env(self, monkeypatch):
        from voice_agent_functions import transcript
        monkeypatch.setenv("TRANSCRIPT_ENABLED", "0")
        transcript.clear_transcript()
        transcript.log_turn(session_id="s1", mode="cook",
                            user_text="hi", assistant_text="hello")
        assert transcript.read_transcript() == ""


# ============================================================
# Session audio recording (mic + Ruchi TTS -> MP3)
# ============================================================
class TestRecording:
    @pytest.fixture(autouse=True)
    def _tmp_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RECORDINGS_DIR", str(tmp_path))
        monkeypatch.setenv("RECORDING_ENABLED", "1")
        yield

    @staticmethod
    def _tone(path, codec="libopus"):
        import subprocess
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "sine=frequency=440:duration=0.5", "-c:a", codec, path],
            check=True,
        )

    def test_add_segment_keeps_chronological_order(self):
        from voice_agent_functions import recording
        assert recording.add_segment("s1", "user", b"abc") is not None
        assert recording.add_segment("s1", "ruchi", b"def") is not None
        names = sorted(p.name for p in recording.segments_dir("s1").iterdir())
        assert [n.split("_", 1)[1] for n in names] == ["user.webm", "ruchi.mp3"]

    def test_finalize_mixes_to_mp3(self, tmp_path):
        import shutil
        if not shutil.which("ffmpeg"):
            pytest.skip("ffmpeg not installed")
        from voice_agent_functions import recording
        u = tmp_path / "u.webm"
        self._tone(str(u), codec="libopus")
        r = tmp_path / "r.mp3"
        self._tone(str(r), codec="libmp3lame")
        recording.add_segment("s2", "user", u.read_bytes())
        recording.add_segment("s2", "ruchi", r.read_bytes())
        out = recording.finalize("s2")
        assert out is not None and out.exists()
        assert out.stat().st_size > 0
        assert out.name == "s2.mp3"

    def test_disabled_skips_writes(self, monkeypatch):
        from voice_agent_functions import recording
        monkeypatch.setenv("RECORDING_ENABLED", "0")
        assert recording.add_segment("s3", "user", b"abc") is None
        assert recording.finalize("s3") is None


# ============================================================
# Parallel / completed-step tracking
# ============================================================
class TestCompletedSteps:
    def test_next_step_skips_completed_parallel_step(self, cooking_session):
        # Mark step 2 (index 1) done early, then advance from step 1.
        mark_parallel_complete(cooking_session, 2)
        assert cooking_session.recipe.is_completed(1)
        event = next_step(cooking_session)
        assert event["type"] == "step_changed"
        # step 1 (index 0) done, step 2 already done → lands on index 2
        assert event["stepIndex"] == 2

    def test_mark_parallel_complete_validates(self, cooking_session):
        with pytest.raises(ValueError):
            mark_parallel_complete(cooking_session, 99)

    def test_next_step_marks_current_completed(self, cooking_session):
        next_step(cooking_session)
        assert cooking_session.recipe.is_completed(0)

    def test_mark_parallel_complete_event(self, cooking_session):
        ev = mark_parallel_complete(cooking_session, 3)
        assert ev["type"] == "step_completed"
        assert ev["stepIndex"] == 2
        assert ev["parallel"] is True


class TestDispatcherParallel:
    def test_mark_parallel_complete_dispatchable(self, cooking_session):
        result = dispatch(cooking_session, "mark_parallel_complete", {"step": 2})
        assert "error" not in result
        assert cooking_session.recipe.is_completed(1)


