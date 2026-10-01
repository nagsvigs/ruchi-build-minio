"""Cooking-companion tools.

These are the 10 tools the cooking-companion agent can call, matching
the spec from ruchi-voice-agent PR #1:

* Step navigation: ``next_step``, ``previous_step``, ``goto_step``
  (see STEP_NAVIGATION.md)
* Timers: ``start_timer``, ``cancel_timer`` (see TIMER.md)
* Pause/resume: ``pause``, ``resume`` (see PAUSE_RESUME.md)
* Fix my dish: ``report_issue``, ``resolve_issue`` (see FIX_MY_DISH.md)
* End: ``finish_cooking`` (see COOKING_AGENT.md)

Every tool:

1. Validates its arguments (raising ``ValueError`` for the dispatcher
   to surface as a Gemini ``error`` response).
2. Mutates the session state.
3. Publishes a ``VoiceSession.events`` entry with a stable
   ``type`` field the connected app listens for.
4. Returns the event dict (the runner may speak a canned follow-up
   based on it).

Handlers never classify user intent themselves — they only run once
the LLM has already decided to invoke a tool. (See COOKING_AGENT.md:
"the model decides when to call each one. The Python handler code
never guesses intent itself.")
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from .state import IssueState, RecipeContext, TimerState, VoiceSession

logger = logging.getLogger("ruchi.cooking_tools")


# Issue categories — the fixed set the app exposes as quick-tap chips
# (FIX_MY_DISH.md).
ISSUE_CATEGORIES: tuple[str, ...] = (
    "spicy",
    "salty",
    "watery",
    "burnt",
    "bland",
    "other",
)


# Canned lead-in per issue category, spoken verbatim before the
# (freeform) fix suggestion.
ISSUE_LEAD_INS: dict[str, str] = {
    "spicy":  "Ah, too spicy — here's what helps:",
    "salty":  "Too salty, got it — try this:",
    "watery": "Too watery — here's a fix:",
    "burnt":  "Let's rescue it —",
    "bland":  "Needs more flavor — try this:",
    "other":  "Let's sort that out —",
}


# Canned step-navigation lead-ins (STEP_NAVIGATION.md).
NEXT_STEP_LEAD_IN = "Okay, next step."
PREVIOUS_STEP_LEAD_IN = "Sure, here's the step before."


# Canned timer phrases (TIMER.md).
TIMER_SET_PHRASE = "Timer set for {duration}."
TIMER_CANCELLED_PHRASE = "Timer cancelled."
TIMER_EXPIRY_PHRASE = "Time's up — check whether your {label} is ready."
NO_TIMER_RUNNING_PHRASE = "There's no timer running right now."


# Canned pause/resume phrases (PAUSE_RESUME.md).
PAUSE_PHRASE = "Okay, I'll wait."
RESUME_LEAD_IN = "Continuing —"
RESUME_TAIL = "let's keep going."   # used for resolve_issue too


# Step-navigation out-of-range conversational fallbacks
# (STEP_NAVIGATION.md).
BEFORE_FIRST_STEP_PHRASE = "That's already the first step."
PAST_LAST_STEP_PHRASE = (
    "That's the last step — want me to wrap up?"
)


# Cooking-finished closing line.
FINISH_COOKING_PHRASE = "Enjoy your meal!"


# ============================================================
# Step navigation
# ============================================================
def _require_recipe(session: VoiceSession) -> None:
    if session.recipe is None:
        raise ValueError(
            "No recipe context set on session; the app must provide "
            "{recipeName, steps, stepIndex} in the room metadata before "
            "cooking tools can fire."
        )


def next_step(session: VoiceSession) -> dict[str, Any]:
    """Advance to the next *not-yet-completed* step.

    Steps completed early as a parallel "meanwhile" task are skipped so
    Ruchi never asks the user to repeat them.
    """
    _require_recipe(session)
    recipe = session.recipe
    assert recipe is not None
    # The step the user was just on is now done.
    recipe.mark_completed(recipe.step_index)
    recipe.parallel_step_index = None
    recipe.parallel_announced = False

    nxt = recipe.next_open_index(recipe.step_index + 1)
    if nxt is None:
        return {
            "type": "no_op",
            "reason": "already_at_last_step",
            "phrase": PAST_LAST_STEP_PHRASE,
        }
    event = session.goto_step(nxt)
    session.events.append(event)
    return {
        **event,
        "lead_in": NEXT_STEP_LEAD_IN,
        "phrase": f"{NEXT_STEP_LEAD_IN} {event['stepText']}",
    }


def mark_parallel_complete(session: VoiceSession, step: int) -> dict[str, Any]:
    """Mark a future step as already done (completed in parallel).

    The model calls this when the user confirms a "meanwhile" task is
    finished, so the step is skipped later instead of repeated.
    """
    _require_recipe(session)
    recipe = session.recipe
    assert recipe is not None
    if not isinstance(step, int) or step < 1 or step > len(recipe.steps):
        raise ValueError(f"step must be in 1..{len(recipe.steps)}, got {step!r}")
    idx = step - 1
    recipe.mark_completed(idx)
    if recipe.parallel_step_index == idx:
        recipe.parallel_step_index = None
    event = {
        "type": "step_completed",
        "stepIndex": idx,
        "stepText": recipe.steps[idx],
        "parallel": True,
    }
    session.events.append(event)
    return event


def previous_step(session: VoiceSession) -> dict[str, Any]:
    """Go back one step."""
    _require_recipe(session)
    assert session.recipe is not None
    if session.recipe.step_index <= 0:
        return {
            "type": "no_op",
            "reason": "already_at_first_step",
            "phrase": BEFORE_FIRST_STEP_PHRASE,
        }
    event = session.goto_step(session.recipe.step_index - 1)
    session.events.append(event)
    return {
        **event,
        "lead_in": PREVIOUS_STEP_LEAD_IN,
        "phrase": f"{PREVIOUS_STEP_LEAD_IN} {event['stepText']}",
    }


def goto_step(session: VoiceSession, step: int) -> dict[str, Any]:
    """Jump directly to step ``step`` (1-indexed when spoken)."""
    _require_recipe(session)
    assert session.recipe is not None
    if not isinstance(step, int) or step < 1:
        raise ValueError(f"step must be a positive int, got {step!r}")
    target = step - 1
    if target < 0 or target >= len(session.recipe.steps):
        if target >= len(session.recipe.steps):
            return {
                "type": "no_op",
                "reason": "past_last_step",
                "phrase": PAST_LAST_STEP_PHRASE,
            }
        return {
            "type": "no_op",
            "reason": "before_first_step",
            "phrase": BEFORE_FIRST_STEP_PHRASE,
        }
    event = session.goto_step(target)
    session.events.append(event)
    return {
        **event,
        "lead_in": f"Jumping to step {step}.",
        "phrase": f"Jumping to step {step}. {event['stepText']}",
    }


# ============================================================
# Recipe catalog
# ============================================================
def load_recipe(session: VoiceSession, name: str) -> dict[str, Any]:
    """Load a recipe from the bundled catalog by spoken dish name.

    The model passes the dish name (it translates Telugu → English first).
    Returns the loaded recipe, or the candidate list when the name is
    ambiguous, or a not-found marker so Ruchi can ask for another dish.
    """
    from . import recipe_catalog  # local import keeps startup light

    query = (name or "").strip()
    if not query:
        raise ValueError("name is required")
    res = recipe_catalog.resolve(query)
    status = res["status"]
    if status == "none":
        # Not in the bundled dataset — try to fetch it from the web.
        try:
            from . import online_recipe

            online = online_recipe.fetch_recipe(query)
        except Exception as exc:  # noqa: BLE001
            logger.warning("online recipe fetch failed: %s", exc)
            online = None
        if online:
            session.recipe = RecipeContext.from_metadata(online)
            event = {
                "type": "recipe_loaded",
                "id": None,
                "recipeName": session.recipe.recipe_name,
                "steps": list(session.recipe.steps),
                "stepIndex": 0,
                "source": "online",
            }
            session.events.append(event)
            return event
        return {
            "type": "recipe_not_found",
            "query": query,
            "phrase": f"I couldn't find {query} in my recipes.",
        }
    if status == "ambiguous":
        candidates = res["candidates"]
        return {
            "type": "recipe_ambiguous",
            "query": query,
            "candidates": [{"id": e.id, "name": e.name} for e in candidates],
            "phrase": "Which one did you mean: "
            + ", ".join(e.name for e in candidates)
            + "?",
        }
    entry = res["entry"]
    session.recipe = RecipeContext.from_metadata(entry.recipe)
    session.recipe.recipe_id = entry.id
    event = {
        "type": "recipe_loaded",
        "id": entry.id,
        "recipeName": entry.name,
        "steps": list(session.recipe.steps),
        "stepIndex": 0,
    }
    session.events.append(event)
    return event


def fetch_recipe_online(session: VoiceSession, name: str) -> dict[str, Any]:
    """Fetch a recipe from the web and load it into the session."""
    from . import online_recipe  # local import keeps startup light

    query = (name or "").strip()
    if not query:
        raise ValueError("name is required")
    recipe = online_recipe.fetch_recipe(query)
    if not recipe:
        return {
            "type": "recipe_not_found",
            "query": query,
            "phrase": f"I couldn't find {query} online either.",
        }
    session.recipe = RecipeContext.from_metadata(recipe)
    event = {
        "type": "recipe_loaded",
        "id": None,
        "recipeName": session.recipe.recipe_name,
        "steps": list(session.recipe.steps),
        "stepIndex": 0,
        "source": "online",
    }
    session.events.append(event)
    return event


def set_servings(session: VoiceSession, servings: int) -> dict[str, Any]:
    """Scale every ingredient quantity to ``servings`` people."""
    _require_recipe(session)
    recipe = session.recipe
    assert recipe is not None
    if not isinstance(servings, int) or servings <= 0:
        raise ValueError(f"servings must be a positive int, got {servings!r}")
    if recipe.structured is not None:
        recipe.structured.scale_to(servings)
    ingredients = (
        [i.spoken() for i in recipe.structured.ingredients]
        if recipe.structured is not None
        else []
    )
    event = {
        "type": "servings_set",
        "servings": servings,
        "ingredients": ingredients,
    }
    session.events.append(event)
    return event


def suggest_recipes(session: VoiceSession, description: str) -> dict[str, Any]:
    """Suggest recipes semantically, e.g. 'something with paneer and rice'."""
    from .memory import recipe_index  # local import keeps startup light

    query = (description or "").strip()
    if not query:
        raise ValueError("description is required")
    matches = recipe_index.suggest(query, top_k=5)
    if not matches:
        return {
            "type": "no_recipe_suggestions",
            "query": query,
            "phrase": f"I couldn't find recipes for {query}.",
        }
    event = {
        "type": "recipe_suggestions",
        "query": query,
        "candidates": matches,
    }
    session.events.append(event)
    return event


# ============================================================
# Timers
# ============================================================
def _format_duration(seconds: int) -> str:
    """Format ``seconds`` into the natural spoken forms the docs call
    for (TIMER.md)."""
    if seconds < 60:
        return f"{seconds} seconds"
    if seconds % 60 == 0:
        n = seconds // 60
        return f"{n} minute" if n == 1 else f"{n} minutes"
    minutes = seconds // 60
    rem = seconds % 60
    return f"{minutes} minutes {rem} seconds"


def start_timer(session: VoiceSession, seconds: int, label: str) -> dict[str, Any]:
    """Start a background timer for ``seconds`` with a short ``label``.

    Handler behavior (TIMER.md):
    1. Start a server-side timer as an asyncio task (it survives the
       tool call returning).
    2. Publish ``timer_started``.
    3. Canned line: ``"Timer set for <duration>."``.

    On expiry (handler-driven via the background task, not the model):
    1. Publish ``timer_expired``.
    2. The runner should proactively push the canned expiry line into
       the audio stream.
    """
    if not isinstance(seconds, int) or seconds <= 0:
        raise ValueError(f"seconds must be a positive int, got {seconds!r}")
    label = (label or "your timer").strip() or "your timer"

    # Cancel any pre-existing timer so we never have two running.
    if session.active_timer is not None:
        _safe_cancel(session.active_timer.task)

    async def _run_timer() -> None:
        try:
            await asyncio.sleep(seconds)
            session.active_timer = None  # clear so a new timer can start
            session.events.append(
                {"type": "timer_expired", "label": label}
            )
        except asyncio.CancelledError:
            return

    # Schedule the background coroutine. When there's no running event
    # loop (e.g. unit tests, or a sync code path), fall back to a
    # plain threading.Timer so the tool is still usable and the task
    # is still cancelable.
    task = _schedule(_run_timer, seconds)
    session.active_timer = TimerState(seconds=seconds, label=label, task=task)

    duration_phrase = _format_duration(seconds)
    event = {
        "type": "timer_started",
        "seconds": seconds,
        "label": label,
        "duration_phrase": duration_phrase,
        "phrase": TIMER_SET_PHRASE.format(duration=duration_phrase),
    }
    session.events.append(event)
    return event


def cancel_timer(session: VoiceSession) -> dict[str, Any]:
    """Cancel the currently running timer, if any."""
    if session.active_timer is None:
        return {
            "type": "no_op",
            "reason": "no_timer_running",
            "phrase": NO_TIMER_RUNNING_PHRASE,
        }
    _safe_cancel(session.active_timer.task)
    cancelled_label = session.active_timer.label
    session.active_timer = None
    event = {
        "type": "timer_cancelled",
        "label": cancelled_label,
        "phrase": TIMER_CANCELLED_PHRASE,
    }
    session.events.append(event)
    return event


def _safe_cancel(task) -> None:
    """Cancel an asyncio.Task OR a threading.Timer/handle without raising."""
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if isinstance(task, asyncio.Task) and running_loop is task.get_loop():
        task.cancel()
        return
    # threading.Timer / threading.Timer-like handle
    if hasattr(task, "cancel"):
        try:
            task.cancel()
        except Exception:
            pass


def _schedule(coro_factory, delay_seconds: float):
    """Run ``coro_factory()`` after ``delay_seconds``.

    Returns either an ``asyncio.Task`` (when there's a running loop)
    or a ``threading.Timer`` (when called from sync code, so unit
    tests and ad-hoc scripts work without an event loop).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No event loop — fall back to a thread timer. We need to
        # schedule a tiny async runner on the main thread; simplest
        # robust approach is to spawn one with asyncio.run in a
        # background thread.
        import threading

        def _runner() -> None:
            try:
                asyncio.run(coro_factory())
            except Exception:
                pass

        t = threading.Timer(delay_seconds, _runner)
        t.daemon = True
        t.start()
        return t

    return asyncio.create_task(coro_factory())


# ============================================================
# Pause / resume
# ============================================================
def pause(session: VoiceSession) -> dict[str, Any]:
    """Stop talking until ``resume`` is called.

    Handler behavior (PAUSE_RESUME.md):
    1. Set ``paused = True`` — runner stops generating prompts until
       ``resume`` is called.
    2. Publish ``paused``.
    3. Canned line: ``"Okay, I'll wait."``
    """
    session.paused = True
    event = {
        "type": "paused",
        "phrase": PAUSE_PHRASE,
    }
    session.events.append(event)
    return event


def resume(session: VoiceSession) -> dict[str, Any]:
    """Clear the paused state and replay the current step.

    Handler behavior (PAUSE_RESUME.md):
    1. Clear paused state.
    2. Publish ``resumed``.
    3. Canned lead-in ``"Continuing —"`` then the current step's
       instruction text (read from room metadata — never invented).
    """
    _require_recipe(session)
    assert session.recipe is not None
    session.paused = False
    step_text = session.recipe.step_text()
    event = {
        "type": "resumed",
        "stepIndex": session.recipe.step_index,
        "lead_in": RESUME_LEAD_IN,
        "phrase": f"{RESUME_LEAD_IN} {step_text}" if step_text else RESUME_LEAD_IN.rstrip("—").rstrip(),
    }
    session.events.append(event)
    return event


# ============================================================
# Fix my dish
# ============================================================
def report_issue(session: VoiceSession, issue: str, details: str) -> dict[str, Any]:
    """Record a user-reported problem with the dish.

    Handler behavior (FIX_MY_DISH.md):
    1. Store ``IssueState`` on the session.
    2. Publish ``issue_reported``.
    3. Return the canned lead-in for the category so the runner can
       speak it before the *freeform* (model-generated) fix suggestion.
    """
    if issue not in ISSUE_CATEGORIES:
        raise ValueError(
            f"issue must be one of {list(ISSUE_CATEGORIES)}, got {issue!r}"
        )
    session.active_issue = IssueState(issue=issue, details=(details or "").strip())
    event = {
        "type": "issue_reported",
        "issue": issue,
        "details": session.active_issue.details,
        "lead_in": ISSUE_LEAD_INS[issue],
    }
    session.events.append(event)
    return event


def resolve_issue(session: VoiceSession) -> dict[str, Any]:
    """Mark the current issue as fixed and resume normal cooking flow."""
    session.active_issue = None
    event = {
        "type": "issue_resolved",
        "phrase": f"{RESUME_TAIL.capitalize()}",   # "Let's keep going."
    }
    session.events.append(event)
    return event


# ============================================================
# Finish cooking
# ============================================================
def finish_cooking(session: VoiceSession) -> dict[str, Any]:
    """End the cooking session.

    Handler behavior (COOKING_AGENT.md):
    1. Publish ``cooking_finished``.
    2. Closing line: ``"Enjoy your meal!"``
    3. Cancel any running timer.
    """
    if session.active_timer is not None:
        _safe_cancel(session.active_timer.task)
        session.active_timer = None
    event = {
        "type": "cooking_finished",
        "phrase": FINISH_COOKING_PHRASE,
    }
    session.events.append(event)
    return event


__all__ = [
    # constants
    "ISSUE_CATEGORIES",
    "ISSUE_LEAD_INS",
    "NEXT_STEP_LEAD_IN",
    "PREVIOUS_STEP_LEAD_IN",
    "TIMER_SET_PHRASE",
    "TIMER_CANCELLED_PHRASE",
    "TIMER_EXPIRY_PHRASE",
    "NO_TIMER_RUNNING_PHRASE",
    "PAUSE_PHRASE",
    "RESUME_LEAD_IN",
    "BEFORE_FIRST_STEP_PHRASE",
    "PAST_LAST_STEP_PHRASE",
    "FINISH_COOKING_PHRASE",
    # step navigation
    "next_step",
    "previous_step",
    "goto_step",
    "mark_parallel_complete",
    # recipe catalog
    "load_recipe",
    "fetch_recipe_online",
    "set_servings",
    "suggest_recipes",
    # timers
    "start_timer",
    "cancel_timer",
    # pause / resume
    "pause",
    "resume",
    # fix my dish
    "report_issue",
    "resolve_issue",
    # end
    "finish_cooking",
]
