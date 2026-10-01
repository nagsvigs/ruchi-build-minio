"""Extract durable memories from a turn using Gemini (httpx REST).

Runs in the background, off the voice critical path. The extractor turns
multilingual (Telugu/English/code-mixed) input into short **English**
memory strings so downstream embedding search is consistent.

Best-effort: never raises. A rule-based prefilter skips clearly
procedural turns ("okay", "next", "done", "సరే") so we don't spend the
rate-limited Gemini quota on them.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger("ruchi.memory.extractor")

_API = "https://generativelanguage.googleapis.com/v1beta/models"

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "should_store": {"type": "boolean"},
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "memory": {"type": "string"},
                    "memory_type": {"type": "string"},
                    "importance": {"type": "number"},
                },
                "required": ["memory"],
            },
        },
        "reason": {"type": "string"},
    },
    "required": ["should_store", "memories"],
}

EXTRACTION_PROMPT = """\
You extract durable memories about a USER from their latest turn with a
Telugu-speaking cooking voice assistant.

Store only durable facts that will matter in FUTURE sessions: food
preferences, allergies, dietary/religious restrictions, dislikes,
household members, kitchen equipment, recurring schedules, skill level.
Do NOT store procedural/transient chat (a current step, "yes", "next",
"how much salt", "wait", "done", "okay", timers).

The user may speak Telugu, English, or mix both. Always translate the
MEANING into a concise English memory string (do not transliterate).

Return STRICT JSON:
{"should_store": bool,
 "memories": [{"memory": str, "memory_type": str, "importance": number}],
 "reason": str}
memory_type is one of: preference, allergy, restriction, dislike,
equipment, household, schedule, skill, general.
importance 0..1 (0.9-1.0 for allergies/religious restrictions/hard
dislikes; 0.7-0.9 strong preferences; 0.5-0.7 mild). At most 3 memories.
"""

# Turns that are clearly procedural — skip the LLM entirely.
_TRIVIAL = re.compile(
    r"^\s*(ok|okay|okey|yes|yeah|no|nope|next|done|continue|wait|stop|"
    r"thanks|thank you|hi|hello|hey|సరే|అవును|కాదు|అయిపోయింది|తర్వాత|"
    r"ఆగండి|చాలు|ధన్యవాదాలు)\s*[.!?]*\s*$",
    re.IGNORECASE,
)


@dataclass
class ExtractionResult:
    should_store: bool = False
    memories: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def is_noop(self) -> bool:
        return not self.should_store or not self.memories


def _enabled() -> bool:
    return (
        os.getenv("MEMORY_ENABLED", "1").lower() not in ("0", "false", "no", "off")
        and os.getenv("MEMORY_EXTRACT", "1").lower() not in ("0", "false", "no", "off")
        and bool(os.getenv("GOOGLE_API_KEY"))
    )


def _trivial(text: str) -> bool:
    t = (text or "").strip()
    if len(t.split()) <= 2:
        return True
    return bool(_TRIVIAL.match(t))


async def extract_memories(
    user_text: str, assistant_text: str | None = None
) -> ExtractionResult:
    if not _enabled() or _trivial(user_text):
        return ExtractionResult()
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    url = f"{_API}/{model}:generateContent?key={os.getenv('GOOGLE_API_KEY', '')}"
    payload = {
        "system_instruction": {"parts": [{"text": EXTRACTION_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"USER TURN: {user_text}\n"
                            f"ASSISTANT (context): {(assistant_text or '')[:400]}"
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 400,
            "responseMimeType": "application/json",
            "responseSchema": _RESPONSE_SCHEMA,
        },
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=4.0)) as client:
            r = await client.post(url, json=payload)
        if r.status_code >= 400:
            return ExtractionResult(error=f"Gemini {r.status_code}: {r.text[:200]}")
        parts = (
            r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        )
        text = "".join(p.get("text", "") for p in parts).strip()
        data = json.loads(text) if text else {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("extract_memories failed: %s", exc)
        return ExtractionResult(error=str(exc))

    memories = []
    for m in (data.get("memories") or [])[:3]:
        mem = str(m.get("memory") or "").strip()
        if not mem:
            continue
        try:
            importance = float(m.get("importance", 0.5))
        except (TypeError, ValueError):
            importance = 0.5
        memories.append(
            {
                "memory": mem,
                "memory_type": str(m.get("memory_type") or "general").strip() or "general",
                "importance": max(0.0, min(1.0, importance)),
            }
        )
    return ExtractionResult(
        should_store=bool(data.get("should_store")) and bool(memories),
        memories=memories,
    )


__all__ = ["ExtractionResult", "extract_memories"]
