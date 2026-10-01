"""Per-session state for the intake + cooking-companion flows.

Each ``VoiceSession`` tracks:

* ``mode``: which agent personality is active (``intake`` or ``cook``).
* ``answers``: collected answers while in intake mode.
* ``current_field``: which intake field is being asked next.
* ``recipe``: ``{recipeName, steps, stepIndex}`` carried by the room
  metadata in cook mode.
* ``paused``: whether the cooking agent is currently waiting.
* ``active_timer``: ``None`` or ``{seconds, label, task}``.
* ``active_issue``: ``None`` or ``{issue, details}``.
* ``events``: ordered list of every state change published by tools
  (``field_saved``, ``step_changed``, ``timer_started``, ...) so the
  connected frontend can render live progress.

The dataclass is intentionally simple and picklable so the same
session can be held in memory by a FastAPI process, persisted to
Redis later, or replayed by tests.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:  # pragma: no cover
    from .recipe_engine import Recipe, RecipeStep

Mode = Literal["intake", "cook"]


# Order of fields the intake agent asks for, from RECIPE_INTAKE_AGENT.md.
INTAKE_FIELDS: tuple[str, ...] = (
    "name",
    "time",
    "servings",
    "level",
    "ingredients",
    "utensils",
    "tips",   # optional — empty string is a valid answer
    "steps",  # last — calls finish_interview after saving
)


@dataclass
class TimerState:
    seconds: int
    label: str
    task: asyncio.Task


@dataclass
class IssueState:
    issue: str  # one of spicy | salty | watery | burnt | bland | other
    details: str


@dataclass
class RecipeContext:
    recipe_name: str
    steps: list[str]
    step_index: int = 0
    # Catalog id when the recipe came from the bundled dataset.
    recipe_id: Optional[str] = None
    # Structured "recipe truth" (heat, quantities, utensils, durations,
    # parallel opportunities) built by recipe_engine.normalize_recipe.
    structured: Optional["Recipe"] = None
    # Step indices already done — including steps completed early as a
    # parallel "meanwhile" task, so they are never repeated.
    completed_steps: set[int] = field(default_factory=set)
    # A future step the user is currently doing in parallel (if any).
    parallel_step_index: Optional[int] = None
    # Whether the parallel suggestion for the current step was already
    # surfaced to the model (so it is offered only once).
    parallel_announced: bool = False

    def step_text(self) -> str:
        if 0 <= self.step_index < len(self.steps):
            return self.steps[self.step_index]
        return ""

    def structured_step(self, index: Optional[int] = None) -> Optional["RecipeStep"]:
        if self.structured is None:
            return None
        return self.structured.step(self.step_index if index is None else index)

    def is_completed(self, index: int) -> bool:
        return index in self.completed_steps

    def mark_completed(self, index: int) -> None:
        if 0 <= index < len(self.steps):
            self.completed_steps.add(index)

    def next_open_index(self, start: int) -> Optional[int]:
        """First index >= ``start`` not already completed, or ``None``."""
        for i in range(max(0, start), len(self.steps)):
            if i not in self.completed_steps:
                return i
        return None

    @classmethod
    def from_metadata(cls, data: dict[str, Any]) -> "RecipeContext":
        from .recipe_engine import normalize_recipe

        steps = list(data.get("steps") or [])
        return cls(
            recipe_name=str(data.get("recipeName", "")),
            steps=steps,
            step_index=int(data.get("stepIndex", 0)),
            structured=normalize_recipe(data),
        )


@dataclass
class VoiceSession:
    session_id: str
    mode: Mode
    user_id: str = "default"
    answers: dict[str, str] = field(default_factory=dict)
    current_field: str = "name"
    recipe: Optional[RecipeContext] = None
    paused: bool = False
    active_timer: Optional[TimerState] = None
    active_issue: Optional[IssueState] = None
    events: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Persistence snapshot (Postgres)
    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """A compact, JSON-serializable state for the DB."""
        recipe = None
        if self.recipe is not None:
            recipe = {
                "recipeName": self.recipe.recipe_name,
                "steps": list(self.recipe.steps),
                "stepIndex": self.recipe.step_index,
                "recipeId": self.recipe.recipe_id,
                "completedSteps": sorted(self.recipe.completed_steps),
                "parallelStepIndex": self.recipe.parallel_step_index,
                "parallelAnnounced": self.recipe.parallel_announced,
            }
        return {
            "answers": dict(self.answers),
            "currentField": self.current_field,
            "paused": self.paused,
            "recipe": recipe,
        }

    @classmethod
    def from_snapshot(
        cls, session_id: str, mode: Mode, user_id: str, snap: dict[str, Any] | None
    ) -> "VoiceSession":
        snap = snap or {}
        session = cls(session_id=session_id, mode=mode, user_id=user_id or "default")
        session.answers = dict(snap.get("answers") or {})
        session.current_field = snap.get("currentField") or "name"
        session.paused = bool(snap.get("paused"))
        recipe = snap.get("recipe")
        if recipe and recipe.get("steps"):
            ctx = RecipeContext(
                recipe_name=recipe.get("recipeName", ""),
                steps=list(recipe["steps"]),
                step_index=int(recipe.get("stepIndex", 0)),
                recipe_id=recipe.get("recipeId"),
            )
            ctx.completed_steps = set(recipe.get("completedSteps") or [])
            ctx.parallel_step_index = recipe.get("parallelStepIndex")
            ctx.parallel_announced = bool(recipe.get("parallelAnnounced"))
            session.recipe = ctx
        return session

    # ------------------------------------------------------------------
    # Intake helpers
    # ------------------------------------------------------------------
    def intake_next_field(self, just_saved: str) -> Optional[str]:
        """Return the field name that comes after ``just_saved`` in
        the canonical intake order, or ``None`` if it was the last one."""
        try:
            idx = INTAKE_FIELDS.index(just_saved)
        except ValueError:
            return None
        if idx + 1 >= len(INTAKE_FIELDS):
            return None
        return INTAKE_FIELDS[idx + 1]

    def intake_field_question(self, field_name: str) -> str:
        """Exact question text for each field (verbatim from the docs)."""
        return {
            "name":       "What's the name of your recipe?",
            "time":       "About how long does it take to cook?",
            "servings":   "How many people does it serve?",
            "level":      "How difficult is it — easy, medium, or hard?",
            "ingredients": "List the ingredients, with approximate quantities.",
            "utensils":   "What utensils do you need? Tell me what you remember.",
            "tips":       "Any quick tips you want to add?",
            "steps":      "Now walk me through all the steps.",
        }.get(field_name, "")

    # ------------------------------------------------------------------
    # Cooking helpers
    # ------------------------------------------------------------------
    def goto_step(self, new_index: int) -> dict[str, Any]:
        assert self.recipe is not None, "recipe context not set"
        clamped = max(0, min(new_index, len(self.recipe.steps) - 1))
        self.recipe.step_index = clamped
        self.recipe.parallel_step_index = None
        self.recipe.parallel_announced = False
        return {
            "type": "step_changed",
            "stepIndex": clamped,
            "stepText": self.recipe.step_text(),
        }


__all__ = [
    "Mode",
    "INTAKE_FIELDS",
    "TimerState",
    "IssueState",
    "RecipeContext",
    "VoiceSession",
]
