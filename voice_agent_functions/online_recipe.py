"""Fetch a recipe from the web when it isn't in the local catalog.

Used by ``load_recipe`` as a fallback for dishes outside the bundled
dataset. Flow:

1. Search DuckDuckGo (HTML) for "<dish> recipe".
2. Fetch the top few result pages and strip them to text.
3. Ask Gemini to turn that text into a structured recipe (schema-enforced).
4. If the web yields nothing, fall back to asking Gemini to write the
   recipe from its own knowledge.

Returns a recipe dict compatible with ``RecipeContext.from_metadata``
(``recipeName, servings, ingredients[], utensils[], steps[], stepDetails``)
or ``None``. All failures are swallowed — the voice loop keeps running.

Enable/disable with ``ONLINE_RECIPE=0``.
"""
from __future__ import annotations

import html
import json
import logging
import os
import re
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

import httpx

logger = logging.getLogger("ruchi.online_recipe")

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120 Safari/537.36"
)
_GEMINI = "https://generativelanguage.googleapis.com/v1beta/models"
_MAX_PAGES = 3
_MAX_TEXT = 8000

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "recipeName": {"type": "string"},
        "servings": {"type": "integer"},
        "utensils": {"type": "array", "items": {"type": "string"}},
        "ingredients": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit": {"type": "string"},
                    "prep": {"type": "string"},
                },
                "required": ["name"],
            },
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "instruction": {"type": "string"},
                    "heat_level": {
                        "type": "string",
                        "enum": ["low", "medium-low", "medium", "medium-high", "high"],
                    },
                    "duration_minutes": {"type": "integer"},
                    "utensils": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["instruction"],
            },
        },
    },
    "required": ["recipeName", "ingredients", "steps"],
}

_PROMPT = """\
You are a recipe formatter. From the provided web text (or, if the text is
empty, from your own reliable knowledge), produce ONE structured recipe for
the requested dish.

Rules:
- recipeName: the dish name in English.
- servings: a sensible base number of servings.
- ingredients: every ingredient with a numeric `quantity`, a `unit`
  (g, ml, cup, tablespoon, teaspoon, pinch, whole, clove, ...), and an
  optional `prep` (finely chopped, grated, ...). Use practical amounts.
- utensils: English utensil names only (Pan, Pot, Bowl, Knife, ...).
- steps: ordered cooking steps. For each step give the instruction, the
  heat_level (low/medium-low/medium/medium-high/high) when it uses a stove,
  and a duration_minutes when the step has a wait.
- Be accurate and practical. Do not invent unsafe steps.
"""


def enabled() -> bool:
    return os.getenv("ONLINE_RECIPE", "1").lower() not in ("0", "false", "no", "off")


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?</\1>", " ", text)
    text = _TAG_RE.sub(" ", text)
    return _WS_RE.sub(" ", html.unescape(text)).strip()


def _search_urls(query: str, limit: int = _MAX_PAGES) -> list[str]:
    try:
        r = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": _UA},
            timeout=12,
            follow_redirects=True,
        )
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.warning("recipe search failed: %s", exc)
        return []
    urls: list[str] = []
    for href in re.findall(r'href="([^"]+)"', r.text):
        if "uddg=" in href:
            qs = parse_qs(urlparse(href).query)
            target = unquote(qs.get("uddg", [""])[0])
        elif href.startswith("http"):
            target = href
        else:
            continue
        if target and "duckduckgo.com" not in target and target not in urls:
            urls.append(target)
        if len(urls) >= limit:
            break
    return urls


def _fetch_text(url: str) -> str:
    try:
        r = httpx.get(
            url,
            headers={"User-Agent": _UA},
            timeout=12,
            follow_redirects=True,
        )
        if r.status_code >= 400 or "text/html" not in r.headers.get("content-type", ""):
            return ""
        return _strip_html(r.text)[: _MAX_TEXT // _MAX_PAGES]
    except Exception:  # noqa: BLE001
        return ""


def _gemini_extract(name: str, web_text: str) -> Optional[dict[str, Any]]:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    url = f"{_GEMINI}/{model}:generateContent?key={key}"
    payload = {
        "system_instruction": {"parts": [{"text": _PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"Dish: {name}\n\n"
                            f"WEB TEXT:\n{web_text or '(none — use your own knowledge)'}"
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
            "responseSchema": _STEP_SCHEMA,
            "thinkingConfig": {"thinkingBudget": 128},
        },
    }
    try:
        r = httpx.post(url, json=payload, timeout=40)
        if r.status_code >= 400:
            logger.warning("gemini extract failed: %s %s", r.status_code, r.text[:200])
            return None
        parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts).strip()
        return json.loads(text) if text else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("gemini extract error: %s", exc)
        return None


def _to_recipe_dict(data: dict[str, Any], name: str) -> Optional[dict[str, Any]]:
    steps = data.get("steps") or []
    instructions = [str(s.get("instruction", "")).strip() for s in steps if s.get("instruction")]
    if not instructions:
        return None
    details: dict[str, Any] = {}
    for i, s in enumerate(steps):
        if not s.get("instruction"):
            continue
        details[str(len(details))] = {
            "heat_level": s.get("heat_level"),
            "duration_minutes": s.get("duration_minutes"),
            "utensils": s.get("utensils") or [],
        }
    return {
        "recipeName": data.get("recipeName") or name,
        "servings": int(data.get("servings") or 4),
        "ingredients": data.get("ingredients") or [],
        "utensils": data.get("utensils") or [],
        "steps": instructions,
        "stepDetails": details,
        "source": "online",
    }


def fetch_recipe(name: str) -> Optional[dict[str, Any]]:
    """Fetch a structured recipe for ``name`` from the web. Blocking."""
    if not enabled() or not name.strip():
        return None
    query = f"{name.strip()} recipe ingredients steps"
    urls = _search_urls(query)
    web_text = ""
    for u in urls:
        t = _fetch_text(u)
        if t:
            web_text += "\n\n" + t
        if len(web_text) >= _MAX_TEXT:
            break
    web_text = web_text[:_MAX_TEXT]
    data = _gemini_extract(name.strip(), web_text)
    if not data:
        return None
    recipe = _to_recipe_dict(data, name.strip())
    if recipe:
        logger.info("online recipe for %r: %d steps", name, len(recipe["steps"]))
    return recipe


__all__ = ["enabled", "fetch_recipe"]
