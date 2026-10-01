"""FastAPI router exposing the function-calling flows over HTTP.

Mounts as ``/api/voice-agent/...`` so the React frontend can drive the
same two-agent design that the LiveKit worker implements.

Endpoints:

* ``POST /sessions``                           — create a new session
  ``{"mode": "intake" | "cook", "recipe"?: {...}}``.
* ``GET  /sessions/{sid}``                    — fetch session state +
  recent events.
* ``DELETE /sessions/{sid}``                  — drop a session.
* ``POST /sessions/{sid}/turn``               — run one user utterance
  through the function-calling loop.
* ``GET  /tools?mode=intake|cook``            — list the function-
  calling schemas (useful for the frontend to render "what Ruchi can
  do").
* ``POST /sessions/{sid}/save_answer``        — explicit shortcut
  used by the intake frontend's "save" button.

When ``GOOGLE_API_KEY`` is not set, every endpoint transparently uses
the rule-based mock runner so the UI works out-of-the-box for
development, matching the rest of ``backend/app.py``.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import transcript
from . import recording
from . import server_voice
from .dispatcher import known_tool_names
from .mock_runner import run_mock_turn
from .runner import GeminiFunctionCallingRunner, RunnerConfig, SessionManager, Turn
from .state import RecipeContext
from .tool_schemas import gemini_tools
from .prompts_loader import load_instructions

logger = logging.getLogger("ruchi.voice_agent_functions.routes")

# Durations Ruchi might state in a step: English + Telugu units.
_DURATION_RE = re.compile(
    r"(\d+)\s*(నిమిషాలు|నిమిషం|నిమిషాల|గంటలు|గంట|సెకన్లు|సెకన్ల|సెకను"
    r"|minutes?|mins?|hours?|hrs?|seconds?|secs?)",
    re.IGNORECASE,
)


def _duration_seconds(text: str) -> int | None:
    """Parse the first duration in ``text`` to seconds, or ``None``."""
    m = _DURATION_RE.search(text or "")
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    if any(u in unit for u in ("గంట", "hour", "hr")):
        return n * 3600
    if any(u in unit for u in ("సెక", "sec")):
        return n
    return n * 60


# Signals that a reply is an actual cooking instruction (not just a
# question or a mention of time). Telugu imperatives almost always end
# in "ండి"; English uses common cooking verbs.
_STEP_HINT_RE = re.compile(
    r"(ండి|వేయించ|ఉడికించ|నానబెట్ట|కలప|మగ్గించ|తరిగ|పోసి|మూత|"
    r"\b(cook|boil|fry|simmer|soak|stir|heat|roast|saute|sauté|add|mix|"
    r"pressure cook|bring to a boil|lower the flame)\b)",
    re.IGNORECASE,
)


def _looks_like_step(text: str) -> bool:
    return bool(_STEP_HINT_RE.search(text or ""))


def _ensure_step_timer(session, turn: "Turn") -> None:
    """Guarantee a timer when Ruchi's reply states a duration.

    The model doesn't reliably call ``start_timer`` even when it says the
    time out loud, so if its reply is an actual cooking step that mentions
    a duration, no timer is running, and it didn't already call the tool,
    start one server-side.
    """
    if session.mode != "cook" or session.active_timer is not None:
        return
    if any(tc.get("name") == "start_timer" for tc in turn.tool_calls):
        return
    text = turn.assistant_text
    # Only auto-time real step instructions, never questions or a
    # passing mention of the total cooking time.
    if "?" in text or "？" in text:
        return
    if not _looks_like_step(text):
        return
    secs = _duration_seconds(text)
    if not secs:
        return
    label = "step timer"
    if session.recipe and session.recipe.step_text():
        label = session.recipe.step_text()[:40]
    from .dispatcher import dispatch  # local import to avoid cycles

    res = dispatch(session, "start_timer", {"seconds": secs, "label": label})
    if "error" not in res:
        turn.tool_calls.append(
            {"name": "start_timer", "args": {"seconds": secs, "label": label}}
        )
        turn.tool_results.append(res.get("result"))
        logger.info("auto-started step timer for %ss (%s)", secs, label)


router = APIRouter(prefix="/api/voice-agent", tags=["voice-agent-functions"])

_sessions = SessionManager()
_runner: GeminiFunctionCallingRunner | None = None


def _get_runner() -> GeminiFunctionCallingRunner:
    """Lazy-build the singleton Gemini runner (mock runner if no key)."""
    global _runner
    if _runner is None:
        _runner = GeminiFunctionCallingRunner(
            RunnerConfig(
                api_key=os.getenv("GOOGLE_API_KEY", ""),
                max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "512")),
                request_timeout=float(os.getenv("GEMINI_REQUEST_TIMEOUT", "4")),
                attempts_per_model=int(os.getenv("GEMINI_ATTEMPTS", "4")),
                hedge=int(os.getenv("GEMINI_HEDGE", "1")),
                total_timeout=float(os.getenv("GEMINI_TOTAL_TIMEOUT", "12")),
                turn_timeout=float(os.getenv("GEMINI_TURN_TIMEOUT", "16")),
                thinking_budget=int(os.getenv("GEMINI_THINKING_BUDGET", "128")),
            )
        )
    return _runner


def lookup_session(session_id: str):
    """Return a live session (memory or DB) or ``None``."""
    try:
        return _sessions.get(session_id)
    except KeyError:
        return None


# ----------------------------------------------------------------------
# Request / response models
# ----------------------------------------------------------------------
class CreateSessionRequest(BaseModel):
    mode: Literal["intake", "cook"]
    session_id: str | None = None
    recipe: dict[str, Any] | None = None
    # Per-browser identity for long-term memory (defaults to DEFAULT_USER_ID).
    user_id: str | None = None


class SessionStateResponse(BaseModel):
    session_id: str
    mode: str
    current_field: str
    answers: dict[str, str]
    paused: bool
    active_timer: dict[str, Any] | None = None
    active_issue: dict[str, Any] | None = None
    recipe: dict[str, Any] | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)


class TurnRequest(BaseModel):
    text: str


class TurnResponse(BaseModel):
    session_id: str
    user_text: str
    assistant_text: str
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    mock: bool = False
    error: str | None = None


class SaveAnswerRequest(BaseModel):
    field: str
    value: str


class CallToolRequest(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


@router.post("/sessions/{session_id}/call")
async def call_tool(session_id: str, req: CallToolRequest):
    """Invoke a tool directly, bypassing the LLM.

    Used by the demo's function dashboard so pressing ``start_timer``
    fires exactly ``start_timer`` with the given args — no reliance on
    the model interpreting a synthetic utterance.
    """
    try:
        session = _sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    from .dispatcher import dispatch
    result = dispatch(session, req.name, req.args)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    transcript.log_note(
        session_id,
        f"tool fired directly: `{req.name}` {req.args or ''}",
    )
    return {
        "ok": True,
        "name": req.name,
        "result": result["result"],
        "events": list(session.events),
        "session": _session_to_response(session),
    }


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------
@router.post("/sessions", response_model=SessionStateResponse)
async def create_session(req: CreateSessionRequest):
    if req.mode not in ("intake", "cook"):
        raise HTTPException(status_code=400, detail=f"unknown mode: {req.mode!r}")
    # Cook mode may start with no recipe: Ruchi loads it from the catalog
    # by voice (load_recipe) once the user names a dish.
    if req.mode == "cook" and req.recipe and not req.recipe.get("steps"):
        raise HTTPException(
            status_code=400,
            detail="recipe.steps[] is required when a recipe is provided.",
        )
    session = _sessions.create(
        mode=req.mode,
        session_id=req.session_id,
        recipe=req.recipe,
        user_id=req.user_id or os.getenv("DEFAULT_USER_ID", "default"),
    )
    return _session_to_response(session)


@router.get("/sessions/{session_id}", response_model=SessionStateResponse)
async def get_session(session_id: str):
    try:
        session = _sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return _session_to_response(session)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    # Build the session MP3 before forgetting the session.
    recording.finalize(session_id)
    _sessions.drop(session_id)
    return {"ok": True}


class SetRecipeRequest(BaseModel):
    recipe: dict[str, Any]


@router.put("/sessions/{session_id}/recipe", response_model=SessionStateResponse)
async def set_recipe(session_id: str, req: SetRecipeRequest):
    """Replace the recipe context mid-session.

    Used when the user names a dish ("I want to cook dosa") — the
    frontend swaps the recipe panel and calls this so the step
    navigation tools operate on the new steps.
    """
    try:
        session = _sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    recipe = req.recipe or {}
    if not recipe.get("steps"):
        raise HTTPException(status_code=400, detail="recipe.steps[] is required")
    session.recipe = RecipeContext.from_metadata(recipe)
    session.events.append(
        {
            "type": "recipe_changed",
            "recipeName": session.recipe.recipe_name,
            "steps": list(session.recipe.steps),
            "stepIndex": session.recipe.step_index,
        }
    )
    return _session_to_response(session)


@router.post("/sessions/{session_id}/turn", response_model=TurnResponse)
async def run_turn(session_id: str, req: TurnRequest):
    try:
        session = _sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    use_mock = not os.getenv("GOOGLE_API_KEY")
    if use_mock:
        from .mock_runner import MockTurn  # local import to avoid cycles
        mt: MockTurn = await run_mock_turn(session, req.text)
        transcript.log_turn(
            session_id=session_id,
            mode=session.mode,
            user_text=mt.user_text,
            assistant_text=mt.assistant_text,
            tool_calls=mt.tool_calls,
            events=list(session.events),
            provider="mock",
        )
        _finalize_if_finished(session_id, mt.tool_calls)
        _persist_turn(session, mt.user_text, mt.assistant_text, mt.tool_calls)
        return TurnResponse(
            session_id=session_id,
            user_text=mt.user_text,
            assistant_text=mt.assistant_text,
            tool_calls=mt.tool_calls,
            tool_results=mt.tool_results,
            events=list(session.events),
            mock=True,
        )

    runner = _get_runner()
    turn: Turn = await runner.run_turn(session, req.text)
    # Never leave the user in silence: if the model failed/timed out and
    # produced no speech, fall back to a short line so Ruchi still replies.
    if turn.error and not turn.assistant_text:
        turn.assistant_text = (
            "క్షమించండి, ఇప్పుడు కొంచెం ఆలస్యం అవుతోంది. మళ్ళీ చెప్పండి."
        )
    _ensure_step_timer(session, turn)
    transcript.log_turn(
        session_id=session_id,
        mode=session.mode,
        user_text=turn.user_text,
        assistant_text=turn.assistant_text,
        tool_calls=turn.tool_calls,
        events=list(session.events),
        provider=os.getenv("GEMINI_MODEL", "gemini"),
    )
    _finalize_if_finished(session_id, turn.tool_calls)
    _persist_turn(session, turn.user_text, turn.assistant_text, turn.tool_calls)
    return TurnResponse(
        session_id=session_id,
        user_text=turn.user_text,
        assistant_text=turn.assistant_text,
        tool_calls=turn.tool_calls,
        tool_results=turn.tool_results,
        events=list(session.events),
        mock=False,
        error=turn.error,
    )


# ----------------------------------------------------------------------
# Memory (long-term, per user_id)
# ----------------------------------------------------------------------
class MemoryAddRequest(BaseModel):
    memory: str
    memory_type: str = "general"
    importance: float = 0.5


@router.get("/memory/{user_id}")
async def list_memory(user_id: str):
    from .memory import manager as memory_manager

    return {
        "enabled": memory_manager.memory_enabled(),
        "memories": [m.to_dict() for m in memory_manager.list_memories(user_id)],
    }


@router.get("/memory/{user_id}/search")
async def search_memory(user_id: str, q: str):
    from .memory import manager as memory_manager

    return {"memories": [m.to_dict() for m in memory_manager.search_memories(user_id, q)]}


@router.post("/memory/{user_id}")
async def add_memory(user_id: str, req: MemoryAddRequest):
    from .memory import manager as memory_manager

    ok = memory_manager.upsert_memory(
        user_id, req.memory, req.memory_type, req.importance
    )
    return {"ok": ok}


@router.delete("/memory/{user_id}")
async def delete_memory(user_id: str):
    from .memory import manager as memory_manager

    return {"deleted": memory_manager.delete_memories(user_id)}


@router.post("/sessions/{session_id}/save_answer")
async def explicit_save_answer(session_id: str, req: SaveAnswerRequest):
    """Direct frontend-driven call (e.g. tapping a 'save' button after
    typing an answer in the intake form). Identical contract to the
    ``save_answer`` function tool — exposed as its own endpoint so the
    UI doesn't need to round-trip through the LLM just to advance."""
    try:
        session = _sessions.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    if session.mode != "intake":
        raise HTTPException(
            status_code=400,
            detail="explicit /save_answer only valid in intake mode",
        )
    from .dispatcher import dispatch
    result = dispatch(session, "save_answer", {"field": req.field, "value": req.value})
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return {"ok": True, "result": result["result"], "session": _session_to_response(session)}


@router.get("/tools")
async def list_tools(mode: Literal["intake", "cook"]):
    """Return the Gemini function-calling schemas for the given mode,
    plus the names of the Python handlers they'll dispatch to."""
    return {
        "mode": mode,
        "tools": gemini_tools(mode),
        "tool_names": sorted(known_tool_names(mode)),
    }


@router.get("/prompts")
async def list_prompts():
    """Return the raw system-prompt text for both modes (handy for
    debugging what Ruchi actually sees)."""
    return {
        "base": load_instructions("cooking"),
        "intake": load_instructions("intake"),
        "cooking_companion": load_instructions("cooking_companion"),
    }


# ----------------------------------------------------------------------
# Session audio recording (mic + Ruchi TTS -> one MP3)
# ----------------------------------------------------------------------
@router.get("/sessions/{session_id}/recording")
async def get_session_recording(session_id: str):
    """Build (if needed) and return the session's MP3 recording.

    404 until at least one audio segment (mic or TTS) has been captured
    for the session. Safe to call repeatedly — the MP3 is only rebuilt
    when new segments have arrived.
    """
    from fastapi.responses import FileResponse

    path = recording.finalize(session_id)
    if path is None or not path.exists():
        raise HTTPException(
            status_code=404,
            detail="no recording for this session yet",
        )
    return FileResponse(path, media_type="audio/mpeg", filename=path.name)


@router.get("/recordings")
async def list_recordings():
    """List every finalized MP3 in the recordings directory."""
    return {
        "dir": str(recording.recordings_dir()),
        "enabled": recording.enabled(),
        "recordings": recording.list_recordings(),
    }


# ----------------------------------------------------------------------
# Transcript — the full user ↔ Ruchi conversation in text form
# ----------------------------------------------------------------------
@router.get("/transcript")
async def get_transcript():
    """Return the whole conversation transcript as Markdown text."""
    from fastapi.responses import PlainTextResponse
    text = transcript.read_transcript()
    if not text:
        text = "# Ruchi — Conversation Transcript\n\n_(no conversation yet)_\n"
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")


@router.get("/transcript/files")
async def get_transcript_files():
    """List the transcript files on disk."""
    return {
        "dir": str(transcript.transcript_dir()),
        "files": transcript.transcript_files(),
    }


@router.delete("/transcript")
async def delete_transcript():
    """Clear all transcript files."""
    transcript.clear_transcript()
    return {"ok": True}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _finalize_if_finished(session_id: str, tool_calls: list[dict[str, Any]]) -> None:
    """Build the session MP3 as soon as a 'finish' tool fires, so a
    complete recording exists without waiting for the download button."""
    if any(tc.get("name") in ("finish_cooking", "finish_interview") for tc in tool_calls):
        recording.finalize(session_id)


def _persist_turn(session, user_text: str, assistant_text: str, tool_calls) -> None:
    """Persist the turn to Postgres + schedule background memory extraction."""
    try:
        from .memory import context as memory_context

        memory_context.record_turn(
            user_id=session.user_id,
            session_id=session.session_id,
            user_text=user_text,
            assistant_text=assistant_text or "",
            tool_calls=tool_calls or [],
            state=session.snapshot(),
            recipe_id=session.recipe.recipe_id if session.recipe else None,
        )
    except Exception as exc:  # noqa: BLE001 - persistence never breaks the turn
        logger.debug("turn persistence skipped: %s", exc)


def _session_to_response(session) -> SessionStateResponse:
    timer = None
    if session.active_timer is not None:
        timer = {
            "seconds": session.active_timer.seconds,
            "label": session.active_timer.label,
        }
    issue = None
    if session.active_issue is not None:
        issue = {
            "issue": session.active_issue.issue,
            "details": session.active_issue.details,
        }
    recipe = None
    if session.recipe is not None:
        recipe = {
            "recipeName": session.recipe.recipe_name,
            "steps": list(session.recipe.steps),
            "stepIndex": session.recipe.step_index,
        }
    return SessionStateResponse(
        session_id=session.session_id,
        mode=session.mode,
        current_field=session.current_field,
        answers=dict(session.answers),
        paused=session.paused,
        active_timer=timer,
        active_issue=issue,
        recipe=recipe,
        events=list(session.events),
    )


__all__ = ["router"]
