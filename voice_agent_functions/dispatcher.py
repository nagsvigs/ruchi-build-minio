"""Routes Gemini ``functionCall`` payloads to the matching Python handler.

Each tool name listed in :mod:`tool_schemas` maps to exactly one
function in :mod:`intake_tools` or :mod:`cooking_tools`. The dispatcher:

1. Looks up the handler by name.
2. Calls it with the session + args from the Gemini payload.
3. Returns a ``{"result": <event-dict>}`` shape (or ``{"error": "..."}``)
   that mirrors what Gemini expects back as a ``functionResponse``.

This keeps tool execution deterministic and unit-testable independently
of the LLM call.
"""
from __future__ import annotations

from typing import Any, Callable

from .cooking_tools import (
    cancel_timer,
    fetch_recipe_online,
    finish_cooking,
    goto_step,
    load_recipe,
    mark_parallel_complete,
    next_step,
    pause,
    previous_step,
    report_issue,
    resolve_issue,
    resume,
    set_servings,
    start_timer,
    suggest_recipes,
)
from .intake_tools import finish_interview, save_answer
from .state import VoiceSession


# name -> handler(session, **kwargs)
_HANDLERS: dict[str, Callable[..., dict[str, Any]]] = {
    # intake
    "save_answer": lambda s, **kw: save_answer(s, field=kw["field"], value=kw["value"]),
    "finish_interview": lambda s, **_: finish_interview(s),
    # cook — recipe catalog
    "load_recipe": lambda s, **kw: load_recipe(s, name=kw["name"]),
    "fetch_recipe_online": lambda s, **kw: fetch_recipe_online(s, name=kw["name"]),
    "set_servings": lambda s, **kw: set_servings(s, servings=kw["servings"]),
    "suggest_recipes": lambda s, **kw: suggest_recipes(s, description=kw["description"]),
    # cooking — step nav
    "next_step": lambda s, **_: next_step(s),
    "previous_step": lambda s, **_: previous_step(s),
    "goto_step": lambda s, **kw: goto_step(s, step=kw["step"]),
    "mark_parallel_complete": lambda s, **kw: mark_parallel_complete(s, step=kw["step"]),
    # cooking — timers
    "start_timer": lambda s, **kw: start_timer(s, seconds=kw["seconds"], label=kw["label"]),
    "cancel_timer": lambda s, **_: cancel_timer(s),
    # cooking — pause/resume
    "pause": lambda s, **_: pause(s),
    "resume": lambda s, **_: resume(s),
    # cooking — fix my dish
    "report_issue": lambda s, **kw: report_issue(s, issue=kw["issue"], details=kw["details"]),
    "resolve_issue": lambda s, **_: resolve_issue(s),
    # cooking — end
    "finish_cooking": lambda s, **_: finish_cooking(s),
}


def known_tool_names(mode: str) -> set[str]:
    """Names of tools valid for ``mode`` — used to validate before dispatch."""
    if mode == "intake":
        return {"save_answer", "finish_interview"}
    if mode == "cook":
        return {
            "load_recipe", "fetch_recipe_online", "set_servings", "suggest_recipes",
            "next_step", "previous_step", "goto_step", "mark_parallel_complete",
            "start_timer", "cancel_timer",
            "pause", "resume",
            "report_issue", "resolve_issue",
            "finish_cooking",
        }
    raise ValueError(f"Unknown mode: {mode!r}")


def dispatch(
    session: VoiceSession,
    name: str,
    args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke the named tool on ``session`` with ``args``.

    Returns a dict in one of two shapes:

    * ``{"result": <event-dict>, "name": <name>}`` on success.
    * ``{"error": "<message>", "name": <name>}`` on failure (validation
      errors, unknown tool, wrong mode, ...).

    Failures are *returned*, not raised, so the Gemini turn can include
    them as a ``functionResponse`` and the model can react to them in
    the next turn (e.g. re-ask the user, correct itself).
    """
    args = args or {}
    if name not in known_tool_names(session.mode):
        return {
            "error": (
                f"Tool {name!r} is not available in mode={session.mode!r}. "
                f"Valid tools: {sorted(known_tool_names(session.mode))}."
            ),
            "name": name,
        }
    handler = _HANDLERS[name]
    try:
        event = handler(session, **args)
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}", "name": name}
    except ValueError as exc:
        return {"error": str(exc), "name": name}
    return {"result": event, "name": name}


def dispatch_function_call(
    session: VoiceSession,
    function_call: dict[str, Any],
) -> dict[str, Any]:
    """Convenience wrapper around :func:`dispatch` that accepts the raw
    ``functionCall`` dict Gemini returns inside ``parts[]``.

    Schema::
        {"name": "next_step", "args": {}}
        {"name": "goto_step", "args": {"step": 4}}
    """
    name = function_call.get("name", "")
    args = function_call.get("args") or {}
    return dispatch(session, name, args)


__all__ = ["dispatch", "dispatch_function_call", "known_tool_names"]
