"""Gemini ``tools[].functionDeclarations`` for both agent modes.

Each entry follows the
`Gemini function-calling schema <https://ai.google.dev/gemini-api/docs/function-calling>`_:

* ``name`` — tool name (must match the Python handler in
  ``intake_tools.py`` / ``cooking_tools.py``).
* ``description`` — what the tool does and *when* to call it, as
  written in the PR docs.
* ``parameters`` — JSON Schema describing the arguments.

The descriptions are deliberately written the same way the docs
spell out the *trigger conditions* — that's the part the LLM uses
to decide whether to invoke the tool.
"""
from __future__ import annotations

from typing import Any


# ----------------------------------------------------------------------
# Intake mode — RECIPE_INTAKE_AGENT.md
# ----------------------------------------------------------------------
INTAKE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "save_answer",
        "description": (
            "Call ONLY when you're confident the user's answer to the CURRENT "
            "intake field is complete and clear. This is the 'advance' decision "
            "— calling this tool is what moves the interview forward. If you're "
            "not confident, do NOT call any tool; just ask the user to clarify "
            "(using one of the fixed re-ask phrases)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "enum": ["name", "time", "servings", "level",
                             "ingredients", "utensils", "tips", "steps"],
                    "description": (
                        "The intake field being saved. Must equal the session's "
                        "current field; the handler will reject mismatches."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": (
                        "The user's answer for this field. For 'tips' an empty "
                        "string is valid when the user says they have nothing "
                        "to add."
                    ),
                },
            },
            "required": ["field", "value"],
        },
    },
    {
        "name": "finish_interview",
        "description": (
            "Call once the 'steps' field (the last intake field) has been saved. "
            "Publishes the final interview_complete event with every collected "
            "answer and ends the structured part of the conversation."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
]


# ----------------------------------------------------------------------
# Cooking-companion mode — COOKING_AGENT.md / STEP_NAVIGATION.md / TIMER.md /
# PAUSE_RESUME.md / FIX_MY_DISH.md
# ----------------------------------------------------------------------
COOKING_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "load_recipe",
        "description": (
            "Call as soon as the user names the dish they want to cook, so the "
            "recipe is loaded from the catalog. Pass the dish name in English "
            "(translate Telugu dish names first). If the name is ambiguous the "
            "tool returns the candidate recipes so you can ask which one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The dish name, e.g. 'Pulihora', 'Idli', 'Sambar'.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "set_servings",
        "description": (
            "Call once the user says how many people they are cooking for, so "
            "every ingredient quantity is scaled to that number. Call it BEFORE "
            "listing ingredients or giving steps."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "servings": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of people being cooked for.",
                },
            },
            "required": ["servings"],
        },
    },
    {
        "name": "fetch_recipe_online",
        "description": (
            "Call when the dish is NOT in the local catalog — load_recipe "
            "returned not-found, or its candidate list does not contain the "
            "dish the user wants. Fetches the recipe from the web and loads it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The dish name to fetch from the web, in English.",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "suggest_recipes",
        "description": (
            "Call when the user asks what they can cook with certain "
            "ingredients or for ideas — 'what can I make with paneer and rice?', "
            "'suggest something for breakfast'. Returns matching recipes from "
            "the catalog to offer the user."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": (
                        "What the user wants, in English — e.g. 'paneer and rice', "
                        "'quick breakfast', 'something with lentils'."
                    ),
                },
            },
            "required": ["description"],
        },
    },
    {
        "name": "next_step",
        "description": (
            "Call when the user indicates they're ready to move forward — "
            "'okay', 'next', 'done', 'what's next', 'I finished that'. "
            "Advances stepIndex by 1."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "previous_step",
        "description": (
            "Call when the user wants to go back — 'wait, go back', 'what was "
            "the last step again', 'go back a step'."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "goto_step",
        "description": (
            "Call when the user names a specific step number directly — "
            "'go to step 4', 'skip to the end', 'take me to step 2'. Steps "
            "are 1-indexed in user speech; the handler converts to 0-indexed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "The 1-indexed step number the user asked for.",
                },
            },
            "required": ["step"],
        },
    },
    {
        "name": "mark_parallel_complete",
        "description": (
            "Call when the user confirms a 'meanwhile' parallel task is done "
            "(e.g. the water is already boiling). Marks that future step as "
            "completed so Ruchi never asks for it again. Pass the 1-indexed "
            "step number of the parallel task that was just finished."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "The 1-indexed step number that was completed early.",
                },
            },
            "required": ["step"],
        },
    },
    {
        "name": "start_timer",
        "description": (
            "Call for EVERY cooking step that has a time, so Ruchi can wait "
            "exactly that long before moving on. Examples: 'soak the rice for "
            "30 minutes', 'cook onions for 5 mins', 'simmer for 10 minutes'. "
            "Also call it when the user explicitly asks to set a timer. Do not "
            "call it for a duration that isn't tied to a cooking step. Never "
            "just say you set a timer — actually call this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "How long the timer should run, in seconds.",
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Short human description of what the timer is for "
                        "(e.g. 'the rice', 'on medium flame'). Pull from "
                        "context if the user gave one, otherwise something "
                        "generic like 'your timer' is fine."
                    ),
                },
            },
            "required": ["seconds", "label"],
        },
    },
    {
        "name": "cancel_timer",
        "description": (
            "Call when the user asks to stop/cancel a running timer — "
            "'cancel that timer', 'never mind the timer'. If no timer is "
            "running the handler returns a 'no timer running' response."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "pause",
        "description": (
            "Call ONLY when the user is directly instructing Ruchi to stop "
            "talking and wait — NOT when 'pause'/'wait'/'stop' comes up while "
            "they're talking about the cooking itself (e.g. 'should I pause "
            "the stirring?' is a technique question, not a command)."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "resume",
        "description": (
            "Call when a paused user tells Ruchi to continue — 'okay, go "
            "ahead', 'continue', 'I'm back, keep going', 'alright, what's "
            "next'."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "report_issue",
        "description": (
            "Call when the user describes a problem with the dish — 'it's too "
            "spicy', 'this is burning', etc. Picks one of the 6 fixed issue "
            "categories; use 'other' when none of the first five match. "
            "DO NOT call this for general cooking questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "issue": {
                    "type": "string",
                    "enum": ["spicy", "salty", "watery", "burnt", "bland", "other"],
                    "description": "One of the 6 fixed issue category ids.",
                },
                "details": {
                    "type": "string",
                    "description": (
                        "Short free-text capture of what the user actually "
                        "said, especially important when issue == 'other'."
                    ),
                },
            },
            "required": ["issue", "details"],
        },
    },
    {
        "name": "resolve_issue",
        "description": (
            "Call once the user confirms a previously reported issue is "
            "fixed (or they've moved on) and wants to get back to normal "
            "cooking flow."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "finish_cooking",
        "description": (
            "Call when the cooking is complete — after the last step is "
            "done and the user confirms they're finished — to end the "
            "session. Do NOT call it mid-recipe."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
]


# ----------------------------------------------------------------------
# Aggregates — what gets sent to Gemini
# ----------------------------------------------------------------------
def gemini_tools(mode: str) -> list[dict[str, Any]]:
    """Return the ``tools`` payload to send to Gemini for ``mode``
    (``"intake"`` or ``"cook"``)."""
    declarations = (
        INTAKE_TOOL_SCHEMAS if mode == "intake" else COOKING_TOOL_SCHEMAS
    )
    return [{"functionDeclarations": declarations}]


__all__ = [
    "INTAKE_TOOL_SCHEMAS",
    "COOKING_TOOL_SCHEMAS",
    "gemini_tools",
]
