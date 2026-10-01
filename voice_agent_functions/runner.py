"""Wires the function-calling tools to the Gemini REST API.

Implements the **standard Gemini tool-use loop**:

1. Send the conversation + ``tools`` to ``generateContent``.
2. If the response contains ``functionCall`` parts, dispatch each one
   via :func:`dispatcher.dispatch_function_call`.
3. Append the resulting ``functionResponse`` parts and loop until
   the model returns a plain text reply.

Both agent modes (``intake`` and ``cook``) share the same loop — only
the system prompt and the ``tools`` payload change.

The runner is intentionally transport-agnostic — it doesn't talk to
LiveKit or any audio plugin. It takes/returns plain text turns, so
the existing FastAPI ``/api/chat`` (or a websocket, or a CLI test) can
drive it.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from .dispatcher import dispatch_function_call, known_tool_names
from .prompts_loader import load_instructions
from . import recipe_engine
from .state import RecipeContext, VoiceSession
from .tool_schemas import gemini_tools

logger = logging.getLogger("ruchi.voice_agent_functions")


def _clean_model(name: str | None) -> str:
    """Strip whitespace (a common .env mistake) from a model name."""
    return (name or "").strip()


# Model fallback chain. The first entry is whatever GEMINI_MODEL is set
# to (loaded from .env); the rest are known-good fallbacks in case the
# configured model has been retired. Duplicates are dropped so a retired
# model configured in .env isn't requested twice.
def _build_model_chain() -> tuple[str, ...]:
    candidates = (
        _clean_model(os.getenv("GEMINI_MODEL")) or "gemini-3.5-flash-lite",
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.1-flash-lite",
        "gemini-flash-latest",
    )
    seen: list[str] = []
    for m in candidates:
        if m and m not in seen:
            seen.append(m)
    # Cap the chain so a slow/retired model can't cascade into a long hang.
    return tuple(seen[:2])


DEFAULT_MODEL_CHAIN: tuple[str, ...] = _build_model_chain()


@dataclass
class Turn:
    """One model turn (user input + assistant reply + tool calls/responses)."""

    user_text: str
    assistant_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def _function_response(
    call_id: str | None,
    name: str,
    *,
    result: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Build a ``functionResponse`` part for Gemini.

    * ``id`` must echo the ``functionCall.id`` so the model can match
      the response to the call.
    * The response body is ``{"result": ...}`` on success or
      ``{"error": ...}`` on failure.
    """
    fr: dict[str, Any] = {"name": name}
    if call_id:
        fr["id"] = call_id
    if error is not None:
        fr["response"] = {"error": error}
    else:
        fr["response"] = {"result": result or {}}
    return {"functionResponse": fr}


@dataclass
class RunnerConfig:
    api_key: str = ""
    model_chain: tuple[str, ...] = DEFAULT_MODEL_CHAIN
    max_tool_iters: int = 5
    temperature: float = 0.7
    max_output_tokens: int = 512
    # Fail fast: the API intermittently hangs, so abort a slow attempt
    # quickly and retry. Each attempt is capped, then retried several
    # times before moving to the next model.
    request_timeout: float = 4.0
    attempts_per_model: int = 3
    # Race this many copies of each request; first to answer wins.
    # Keep at 1 on rate-limited keys (hedging doubles quota usage).
    hedge: int = 1
    # Hard ceiling for a single Gemini call (across all retries/models).
    total_timeout: float = 12.0
    # Hard ceiling for a whole turn (may include two Gemini calls + tools).
    turn_timeout: float = 16.0
    # Cap "thinking" so latency stays predictable. -1 omits the field.
    thinking_budget: int = 128
    # Keep only the last N messages of history (bounds prompt growth).
    max_history_messages: int = 16


# ============================================================
# Session manager
# ============================================================
class SessionManager:
    """Store of :class:`VoiceSession` objects, write-through to Postgres.

    Sessions are kept in memory for speed and persisted to the DB (when
    memory is enabled) so they survive a restart. ``get`` reconstructs a
    session from the DB if it isn't in memory.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, VoiceSession] = {}

    def create(
        self,
        mode: Literal["intake", "cook"],
        session_id: str | None = None,
        recipe: dict[str, Any] | None = None,
        user_id: str = "default",
    ) -> VoiceSession:
        sid = session_id or uuid.uuid4().hex
        recipe_ctx = (
            RecipeContext.from_metadata(recipe) if recipe else None
        )
        session = VoiceSession(
            session_id=sid,
            mode=mode,
            user_id=user_id or "default",
            recipe=recipe_ctx,
        )
        self._sessions[sid] = session
        self._persist_create(session)
        return session

    def get(self, session_id: str) -> VoiceSession:
        if session_id in self._sessions:
            return self._sessions[session_id]
        restored = self._restore(session_id)
        if restored is not None:
            self._sessions[session_id] = restored
            return restored
        raise KeyError(f"Unknown session_id: {session_id!r}")

    def drop(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session and session.active_timer is not None:
            session.active_timer.task.cancel()
        self._persist_end(session_id)

    # ---------------------------------------------------------------- db
    @staticmethod
    def _persist_create(session: VoiceSession) -> None:
        try:
            from .memory import manager as memory_manager

            if not memory_manager.memory_enabled():
                return
            recipe_id = session.recipe.recipe_id if session.recipe else None
            memory_manager.create_session(
                session.user_id,
                session.session_id,
                mode=session.mode,
                recipe_id=recipe_id,
                state=session.snapshot(),
            )
        except Exception as exc:  # noqa: BLE001 - never break the loop
            logger.debug("session persist skipped: %s", exc)

    @staticmethod
    def _persist_end(session_id: str) -> None:
        try:
            from .memory import manager as memory_manager

            if memory_manager.memory_enabled():
                memory_manager.end_session(session_id)
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def _restore(session_id: str) -> VoiceSession | None:
        try:
            from .memory import manager as memory_manager

            if not memory_manager.memory_enabled():
                return None
            row = memory_manager.get_session(session_id)
        except Exception:  # noqa: BLE001
            return None
        if not row:
            return None
        session = VoiceSession.from_snapshot(
            session_id, row.get("mode") or "cook", row.get("user_id") or "default",
            row.get("state"),
        )
        # Rebuild the structured recipe from the catalog when possible so
        # heat/quantities/utensils survive a restart.
        snap_recipe = (row.get("state") or {}).get("recipe") or {}
        recipe_id = row.get("recipe_id") or snap_recipe.get("recipeId")
        if recipe_id:
            try:
                from . import recipe_catalog

                entry = recipe_catalog.get_catalog().get(recipe_id)
                if entry is not None:
                    ctx = RecipeContext.from_metadata(entry.recipe)
                    ctx.step_index = int(snap_recipe.get("stepIndex", 0))
                    ctx.recipe_id = recipe_id
                    ctx.completed_steps = set(snap_recipe.get("completedSteps") or [])
                    ctx.parallel_step_index = snap_recipe.get("parallelStepIndex")
                    ctx.parallel_announced = bool(snap_recipe.get("parallelAnnounced"))
                    session.recipe = ctx
            except Exception:  # noqa: BLE001
                pass
        return session


# ============================================================
# Gemini tool-use loop
# ============================================================
class GeminiFunctionCallingRunner:
    """Runs a Gemini conversation with our 12 function-calling tools."""

    def __init__(self, config: RunnerConfig | None = None) -> None:
        self.config = config or RunnerConfig(api_key=os.getenv("GOOGLE_API_KEY", ""))
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.request_timeout, connect=4.0),
            transport=httpx.AsyncHTTPTransport(retries=2),
        )
        self._history: dict[str, list[dict[str, Any]]] = {}
        # Last model that actually answered. Tried first on every later
        # call so a retired model at the head of the chain costs one 404
        # total instead of one 404 per request.
        self._working_model: str | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    # ---------------------------------------------------------- prompts
    def _system_prompt(
        self, session: VoiceSession, memory_block: str = "", situation_hint: str = ""
    ) -> str:
        if session.mode == "intake":
            base = load_instructions("intake")
        else:
            # Cook mode uses the cooking prompt as the single source of truth.
            # (We deliberately do NOT append the old "not a script" overlay —
            # it contradicted the required stage-by-stage cooking flow.)
            base = f"{load_instructions('cooking')}\n\n---\n\n{self._session_context(session)}"
        if situation_hint:
            base = f"{base}\n\n---\n\n{situation_hint}"
        if memory_block:
            base = f"{base}\n\n---\n\n{memory_block}"
        return base

    @staticmethod
    def _situation_hint(session: VoiceSession, history: list[dict[str, Any]]) -> str:
        """Deterministic situation directive for recipe-less cook sessions.

        The model sometimes skips the mandatory clarification questions, so
        when no recipe is loaded we detect the intent from the latest user
        message and tell it exactly which single question to ask.
        """
        import re

        if session.mode != "cook":
            return ""
        recipe = session.recipe
        if recipe is not None and recipe.steps:
            return ""
        user_text = ""
        for m in reversed(history or []):
            if m.get("role") != "user":
                continue
            parts = m.get("parts") or []
            texts = [
                p.get("text", "") for p in parts
                if isinstance(p, dict) and "text" in p
            ]
            user_text = " ".join(texts).strip()
            if user_text:
                break
        if not user_text:
            return ""
        t = user_text
        health_vague = re.search(
            r"\b(healthy|healthier|health|diet\b|ఆరోగ్య)\b", t, re.I
        )
        health_specific = re.search(
            r"protein|vitamin|low[\s-]oil|low[\s-]fat|iron|calcium|"
            r"fibr[eo]|diabet|sugar[\s-]?free|weight|ప్రోటీన్|విటమిన్|"
            r"తక్కువ నూనె|డైట్", t, re.I
        )
        discovery = re.search(
            r"don'?t know|what (should|can|do) (i|we) (cook|make)|"
            r"\bsuggest\b|ideas?|options?|what to (cook|make|do)|"
            r"ఏం వండాలో తెలియదు|ఏం చేయాలి|సజెస్ట్", t, re.I
        )
        if health_vague and not health_specific:
            return (
                "SITUATION NOW: the user wants healthy food but gave NO specific "
                "health goal. Your reply MUST be a single question asking what "
                "'healthy' means — protein-rich? light? less oil? vitamins? "
                "Ask it in Telugu. Do NOT name any dish and do NOT call "
                "suggest_recipes in this reply."
            )
        if discovery and not health_specific:
            return (
                "SITUATION NOW: the user does not know what to cook. Your reply "
                "MUST ask ONLY what ingredients they have (in Telugu). Do NOT "
                "name any dish and do NOT call suggest_recipes in this reply."
            )
        return ""

    @staticmethod
    def _session_context(session: VoiceSession) -> str:
        """A compact, app-owned snapshot of where the user is.

        Injected into the system prompt every turn so the model follows
        the recipe truth (exact quantities, heat levels, English utensils,
        parallel tasks, completed steps) instead of guessing or inventing.
        """
        lines = ["SESSION CONTEXT (from the app; never read this block aloud):"]
        recipe = session.recipe
        if recipe is None or not recipe.steps:
            lines.append("- No recipe is loaded in the app yet.")
            lines.append("- You are still at Stage 1: ask what to cook.")
            lines.append(
                "- As soon as the user names a dish, call load_recipe(name) "
                "(translate Telugu dish names to English first). If it is not "
                "found or ambiguous, follow the tool result."
            )
            return "\n".join(lines)

        total = len(recipe.steps)
        name = recipe.recipe_name or "the current dish"
        lines.append(f'- Recipe loaded in the app: "{name}" ({total} steps).')

        structured = getattr(recipe, "structured", None)
        if structured is not None:
            base = structured.base_servings or 1
            target = structured.target_servings
            if target:
                lines.append(
                    f"- Cooking for {target} people (quantities already scaled "
                    f"from a base of {base})."
                )
            else:
                lines.append(
                    f"- Base servings: {base}. Once the user gives the headcount, "
                    f"call set_servings(n) so quantities scale."
                )

        # Exact recipe-level ingredient + utensil lists (for Stage 4).
        if structured is not None and structured.ingredients:
            lines.append(
                "- Exact ingredients (always speak the quantity; never drop it): "
                + recipe_engine.spoken_ingredient_list(structured.ingredients)
            )
        if structured is not None and structured.utensils:
            eng = [recipe_engine.normalize_utensil(u) for u in structured.utensils]
            lines.append("- Utensils (ENGLISH ONLY, never regional names): " + ", ".join(eng))

        # Steps already completed — including early parallel ones.
        if recipe.completed_steps:
            done = ", ".join(str(i + 1) for i in sorted(recipe.completed_steps))
            lines.append(f"- Steps already completed (NEVER ask for these again): {done}")

        step_text = recipe.step_text()
        if not step_text:
            lines.append("- Current step: not started.")
            lines.append("- Use this only to stay consistent. Do not read this block aloud.")
            return "\n".join(lines)

        lines.append(f'- Current step ({recipe.step_index + 1}/{total}): "{step_text}"')
        st = recipe.structured_step() if structured is not None else None
        if st is not None:
            if st.duration_seconds:
                lines.append(f"  duration: {max(1, round(st.duration_seconds / 60))} min")
            if st.heat_phrase:
                lines.append(f'  heat (MUST be spoken in Telugu): "{st.heat_phrase}"')
            if st.ingredients:
                lines.append(
                    "  exact ingredients to use now: "
                    + recipe_engine.spoken_ingredient_list(st.ingredients)
                )
            if st.utensils:
                lines.append("  utensils in use (English): " + st.spoken_utensils())
            if st.requires_attention:
                lines.append(
                    "  requires attention: YES — do not suggest a meanwhile task for this step."
                )

        # Parallel "meanwhile" task — offered once per step, then tracked.
        # Only once cooking is actually underway (not during Stages 1-4).
        cooking_active = (
            session.active_timer is not None
            or recipe.step_index > 0
            or bool(recipe.completed_steps)
        )
        if structured is not None and cooking_active:
            if not recipe.parallel_announced:
                recipe.parallel_announced = True
                idx = recipe_engine.find_parallel_step(
                    structured, recipe.step_index, recipe.completed_steps
                )
                if idx is not None:
                    recipe.parallel_step_index = idx
                    p = structured.step(idx)
                    if p is not None:
                        extra = f'; heat: "{p.heat_phrase}"' if p.heat_phrase else ""
                        if p.ingredients:
                            extra += "; ingredients: " + p.spoken_ingredients()
                        lines.append(
                            f'- MEANWHILE task (offer ONCE, safely in parallel): '
                            f'step {idx + 1} "{p.text}"{extra}'
                        )
            elif recipe.parallel_step_index is not None:
                p = structured.step(recipe.parallel_step_index)
                if p is not None:
                    lines.append(
                        f'- Ongoing parallel task: step {recipe.parallel_step_index + 1} '
                        f'"{p.text}" — do not re-explain it; when the user confirms it is '
                        f"done, call mark_parallel_complete(step={recipe.parallel_step_index + 1})."
                    )

        lines.append(
            "- Use this only to stay consistent. Do not read this block aloud until "
            "you are in Stage 5 of the COOKING FLOW."
        )
        return "\n".join(lines)

    # ---------------------------------------------------------- gemini
    def _build_payload(
        self,
        session: VoiceSession,
        history: list[dict[str, Any]],
        user_text: str | None,
        include_tools: bool = True,
        memory_block: str = "",
    ) -> dict[str, Any]:
        contents = list(history)
        if user_text is not None:
            contents.append({"role": "user", "parts": [{"text": user_text}]})

        gen: dict[str, Any] = {
            "temperature": self.config.temperature,
            "maxOutputTokens": self.config.max_output_tokens,
        }
        if self.config.thinking_budget >= 0:
            gen["thinkingConfig"] = {"thinkingBudget": self.config.thinking_budget}
        payload: dict[str, Any] = {
            "system_instruction": {
                "parts": [
                    {
                        "text": self._system_prompt(
                            session,
                            memory_block,
                            self._situation_hint(session, contents),
                        )
                    }
                ]
            },
            "contents": contents,
            "generationConfig": gen,
        }
        # The follow-up (spoken reply) call doesn't need the tools payload —
        # it's much smaller and faster, and the server-side auto-timer
        # covers any missed start_timer.
        if include_tools:
            payload["tools"] = gemini_tools(session.mode)
            payload["tool_config"] = {"function_calling_config": {"mode": "AUTO"}}
        return payload

    def _model_order(self) -> list[str]:
        """Order the model chain, preferring the last known-good model."""
        chain = list(self.config.model_chain)
        if self._working_model and self._working_model in chain:
            chain.remove(self._working_model)
            chain.insert(0, self._working_model)
        return chain

    def _trim_history(self, history: list[dict[str, Any]]) -> None:
        """Bound the conversation history so prompts don't grow forever.

        Keeps the most recent messages but never starts in the middle of a
        tool exchange (a functionResponse with no preceding functionCall).
        """
        n = max(2, self.config.max_history_messages)
        if len(history) <= n:
            return
        trimmed = history[-n:]
        while trimmed and not (
            trimmed[0].get("role") == "user"
            and any("text" in p for p in trimmed[0].get("parts", []))
        ):
            trimmed.pop(0)
        history[:] = trimmed

    async def _post_once(
        self, url: str, payload: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str]:
        """One request. Returns ``(data, "")`` on success or ``(None, err)``.

        Raises for client errors (400/401/403) that retrying won't fix.
        """
        try:
            r = await self._client.post(url, json=payload)
        except httpx.HTTPError as exc:
            return None, f"network error: {exc!r}"
        if r.status_code in (400, 401, 403):
            raise RuntimeError(f"Gemini {r.status_code}: {r.text[:400]}")
        if r.status_code == 429:
            # Rate limited — retrying immediately only burns more quota.
            return None, "__rate_limited__"
        if r.status_code == 404:
            return None, "__model_missing__"
        if r.status_code >= 500:
            return None, f"Gemini {r.status_code}: {r.text[:200]}"
        try:
            data = r.json()
        except ValueError:
            return None, f"bad json: {r.text[:200]}"
        if data.get("candidates"):
            return data, ""
        return None, f"empty Gemini response: {str(data)[:200]}"

    async def _hedged_post(
        self, url: str, payload: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str]:
        """Fire ``hedge`` copies of the request and return the first win.

        The API has high latency variance, so racing two requests cuts the
        tail dramatically; the slower copy is cancelled.
        """
        hedge = max(1, self.config.hedge)
        tasks = [asyncio.create_task(self._post_once(url, payload)) for _ in range(hedge)]
        last_err = ""
        try:
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for t in done:
                    data, err = t.result()
                    if data is not None:
                        return data, ""
                    last_err = err
                    if err == "__model_missing__":
                        return None, err
        finally:
            for t in tasks:
                if not t.done():
                    t.cancel()
        return None, last_err

    async def _call_gemini(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        last_err = ""
        deadline = time.monotonic() + self.config.total_timeout
        for model in self._model_order():
            url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                f"models/{model}:generateContent?key={self.config.api_key}"
            )
            for attempt in range(max(1, self.config.attempts_per_model)):
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"Gemini timed out after {self.config.total_timeout}s: {last_err}"
                    )
                data, err = await self._hedged_post(url, payload)
                if data is not None:
                    self._working_model = model
                    return data
                last_err = f"{model}: {err}"
                if err in ("__model_missing__", "__rate_limited__"):
                    break  # don't burn quota; try next model / give up
                await asyncio.sleep(0.1)
        raise RuntimeError(f"All Gemini models failed: {last_err}")

    # ---------------------------------------------------------- main loop
    async def run_turn(
        self,
        session: VoiceSession,
        user_text: str,
    ) -> Turn:
        """Process one user utterance through the function-calling loop.

        Returns a :class:`Turn` describing the assistant reply and any
        tools that fired. ``session.events`` is appended to as a
        side-effect.
        """
        history = self._history.setdefault(session.session_id, [])
        turn = Turn(user_text=user_text)

        history.append({"role": "user", "parts": [{"text": user_text}]})
        self._trim_history(history)

        # Long-term memory retrieval (local embedding + DB) — best-effort.
        memory_block = ""
        try:
            from .memory import context as memory_context

            memory_block = memory_context.retrieve_memory_block(
                session.user_id, user_text
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("memory retrieval skipped: %s", exc)

        turn_started = time.monotonic()
        include_tools = True
        for _ in range(self.config.max_tool_iters):
            if time.monotonic() - turn_started > self.config.turn_timeout:
                turn.error = f"turn timed out after {self.config.turn_timeout}s"
                logger.warning("turn %s timed out", session.session_id)
                break
            payload = self._build_payload(
                session, history, user_text=None, include_tools=include_tools,
                memory_block=memory_block,
            )
            try:
                data = await self._call_gemini(payload)
            except Exception as exc:
                turn.error = str(exc)
                logger.exception("Gemini call failed")
                break

            parts = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )

            function_calls = [p["functionCall"] for p in parts if "functionCall" in p]
            # Exclude "thought" parts — Gemini's private chain-of-thought.
            # Speaking or storing these would leak Ruchi's reasoning.
            text_chunks = [
                p["text"] for p in parts if "text" in p and not p.get("thought")
            ]
            text = "".join(text_chunks).strip()
            # Keep any text the model produced, even on turns that also fire
            # tools — otherwise a tool call with a spoken line can end up
            # with an empty assistant_text (Ruchi goes silent).
            if text:
                turn.assistant_text = (
                    f"{turn.assistant_text} {text}".strip()
                    if turn.assistant_text
                    else text
                )

            if function_calls:
                # Append the model turn with its functionCall parts
                # (including thoughtSignature, which Gemini requires us
                # to echo back verbatim).
                history.append({"role": "model", "parts": parts})
                # Dispatch each call, keeping the functionCall id so the
                # functionResponse can be matched to it.
                response_parts: list[dict[str, Any]] = []
                for fc in function_calls:
                    name = fc.get("name", "")
                    call_id = fc.get("id")
                    if name not in known_tool_names(session.mode):
                        # Defensive — the schema should prevent this, but
                        # if Gemini ever returns an out-of-scope tool we
                        # want a clean error back rather than a crash.
                        response_parts.append(
                            _function_response(
                                call_id, name,
                                error=f"tool {name!r} not in scope for mode={session.mode!r}",
                            )
                        )
                        turn.tool_calls.append({"name": name, "args": fc.get("args", {})})
                        continue
                    result = dispatch_function_call(session, fc)
                    response_parts.append(_function_response(call_id, name, result=result))
                    turn.tool_calls.append({"name": name, "args": fc.get("args", {})})
                    turn.tool_results.append(result)

                # IMPORTANT: this Gemini API rejects role="function";
                # functionResponse parts go in a role="user" turn.
                history.append({"role": "user", "parts": response_parts})
                # Loop back to let the model turn the tool outputs into
                # a final text reply. That follow-up doesn't need the tools
                # payload (smaller + faster).
                include_tools = False
                continue

            if not function_calls and not text:
                # Model returned only private thought — retry rather than
                # speaking an empty reply.
                logger.warning("model returned only thought parts; retrying")
                continue

            # Plain text reply — we're done.
            history.append({"role": "model", "parts": [{"text": text}]})
            return turn

        # Fell out of the loop without a final text — surface what we have.
        turn.error = turn.error or "model loop ended without a text reply"
        return turn


# ============================================================
# Convenience: one-shot helper
# ============================================================
async def run_once(
    mode: Literal["intake", "cook"],
    user_text: str,
    *,
    session: VoiceSession | None = None,
    recipe: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> tuple[VoiceSession, Turn]:
    """Run a single user turn end-to-end and return ``(session, turn)``.

    Useful for tests and CLI one-shots. Use a :class:`SessionManager` +
    a long-lived :class:`GeminiFunctionCallingRunner` for production.
    """
    cfg = RunnerConfig(api_key=api_key or os.getenv("GOOGLE_API_KEY", ""))
    runner = GeminiFunctionCallingRunner(cfg)
    try:
        if session is None:
            session = VoiceSession(
                session_id=uuid.uuid4().hex,
                mode=mode,
                recipe=RecipeContext.from_metadata(recipe) if recipe else None,
            )
        turn = await runner.run_turn(session, user_text)
        return session, turn
    finally:
        await runner.aclose()


__all__ = [
    "Turn",
    "RunnerConfig",
    "SessionManager",
    "GeminiFunctionCallingRunner",
    "run_once",
]


# Silence an unused-import warning from the type-check side of the loop
# above when asyncio isn't used elsewhere in this module.
_ = asyncio
