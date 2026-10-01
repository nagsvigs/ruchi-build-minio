"""Deterministic recipe intelligence — the "recipe truth" layer.

The LLM's job is to turn structured recipe data into natural spoken
Telugu/English. This module owns the parts that must never be left to
chance:

* **Heat levels** → the Telugu phrase Ruchi must speak (``heat_phrase``).
* **Utensils** → normalized to English (``normalize_utensil``).
* **Ingredient quantities** → parsed and formatted exactly, never dropped
  (``Ingredient.spoken`` / ``format_quantity``).
* **Durations** → parsed from step text (``parse_duration_seconds``).
* **Parallel work** → safe, independent later steps that can happen during
  a long waiting step (``find_parallel_step``).
* **Slow list speech** → comma/ellipsis separated lists for TTS.

Everything here is pure and unit-testable; the runner injects the results
into the model prompt so the model never has to remember quantities, heat
levels, or utensil names across turns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

# ======================================================================
# Heat levels
# ======================================================================
HEAT_LEVELS: tuple[str, ...] = ("low", "medium-low", "medium", "medium-high", "high")

# The exact Telugu phrases Ruchi must speak for each heat level.
HEAT_PHRASE_TE: dict[str, str] = {
    "low": "లో ఫ్లేమ్‌లో",
    "medium-low": "మీడియం-లో ఫ్లేమ్‌లో",
    "medium": "మీడియం ఫ్లేమ్‌లో",
    "medium-high": "మీడియం-హై ఫ్లేమ్‌లో",
    "high": "హై ఫ్లేమ్‌లో",
}


def heat_phrase(level: Optional[str]) -> Optional[str]:
    """Return the Telugu flame phrase for ``level`` (or ``None``).

    Also accepts an already-Telugu phrase and returns it unchanged, so
    datasets that store the spoken phrase directly pass through.
    """
    if not level:
        return None
    s = str(level).strip()
    lowered = s.lower()
    if lowered in HEAT_PHRASE_TE:
        return HEAT_PHRASE_TE[lowered]
    if s in HEAT_PHRASE_BY_TE:
        return s
    return None


# Reverse map: spoken Telugu phrase → canonical level. Used to ingest
# datasets that already provide the spoken phrase.
HEAT_PHRASE_BY_TE: dict[str, str] = {v: k for k, v in HEAT_PHRASE_TE.items()}


def level_from_phrase(phrase: Optional[str]) -> Optional[str]:
    """Return the canonical level for a Telugu heat phrase, or ``None``."""
    if not phrase:
        return None
    return HEAT_PHRASE_BY_TE.get(str(phrase).strip())


# ======================================================================
# Utensils — English only
# ======================================================================
# Regional (Telugu/Hindi, romanized and script) and common variants →
# the single English name Ruchi is allowed to speak.
_UTENSIL_MAP: dict[str, str] = {
    "kadai": "Pan",
    "kadhai": "Pan",
    "kadai pan": "Pan",
    "karahi": "Wok",
    "wok": "Wok",
    "tawa": "Griddle",
    "tava": "Griddle",
    "flat pan": "Griddle",
    "griddle": "Griddle",
    "handi": "Pot",
    "pot": "Pot",
    "pan": "Pan",
    "frying pan": "Pan",
    "saucepan": "Pan",
    "bowl": "Bowl",
    "mixing bowl": "Bowl",
    "spoon": "Spoon",
    "ladle": "Ladle",
    "knife": "Knife",
    "cutting board": "Cutting board",
    "chopping board": "Cutting board",
    "pressure cooker": "Pressure cooker",
    "cooker": "Pressure cooker",
    "strainer": "Strainer",
    "sieve": "Strainer",
    "plate": "Plate",
    "cup": "Cup",
    "grater": "Grater",
    "blender": "Blender",
    "mixer": "Mixer",
    "mixer grinder": "Mixer",
    "rolling pin": "Rolling pin",
    "tray": "Tray",
    "spatula": "Spatula",
    "tongs": "Tongs",
    "whisk": "Whisk",
    # Telugu script
    "కడాయి": "Pan",
    "పెనం": "Griddle",
    "గిన్నె": "Pot",
    "బౌల్": "Bowl",
    "చెంచా": "Spoon",
    "గరిటె": "Ladle",
    "కత్తి": "Knife",
    "ప్లేటు": "Plate",
    "కప్పు": "Cup",
    "స్ట్రెయినర్": "Strainer",
    "కుక్కర్": "Pressure cooker",
    "ప్రెషర్ కుక్కర్": "Pressure cooker",
    "తరిగే బోర్డు": "Cutting board",
}


def normalize_utensil(name: str) -> str:
    """Map a regional/variant utensil name to its English name.

    Unknown names are returned title-cased so they still read as English.
    """
    if not name:
        return name
    key = re.sub(r"\s+", " ", str(name).strip().lower())
    if key in _UTENSIL_MAP:
        return _UTENSIL_MAP[key]
    # substring match: "big kadai" → Pan, "steel bowl" → Bowl
    for needle, english in _UTENSIL_MAP.items():
        if needle in key:
            return english
    return str(name).strip().title()


# ======================================================================
# Quantities & ingredients
# ======================================================================
_FRACTION_WORDS: dict[float, str] = {
    0.5: "1/2",
    0.25: "1/4",
    0.75: "3/4",
    1 / 3: "1/3",
    2 / 3: "2/3",
    0.2: "1/5",
}


def format_quantity(qty: Any) -> str:
    """Format a numeric quantity as a clear spoken form.

    ``0.5 -> "1/2"``, ``1 -> "1"``, ``1.5 -> "1 1/2"``. Strings pass
    through unchanged (so ``"to taste"`` survives).
    """
    if qty is None:
        return ""
    if isinstance(qty, str):
        return qty.strip()
    try:
        q = float(qty)
    except (TypeError, ValueError):
        return str(qty)
    if abs(q - round(q)) < 1e-9:
        return str(int(round(q)))
    whole = int(q)
    frac = q - whole
    for value, word in _FRACTION_WORDS.items():
        if abs(frac - value) < 0.02:
            return f"{whole} {word}".strip() if whole else word
    return f"{q:g}"


_UNIT_ALIASES: dict[str, str] = {
    "tbsp": "tablespoon",
    "tbsps": "tablespoons",
    "tablespoon": "tablespoon",
    "tablespoons": "tablespoons",
    "tsp": "teaspoon",
    "tsps": "teaspoons",
    "teaspoon": "teaspoon",
    "teaspoons": "teaspoons",
    "cup": "cup",
    "cups": "cups",
    "gram": "grams",
    "grams": "grams",
    "g": "grams",
    "kg": "kg",
    "kilogram": "kg",
    "kilograms": "kg",
    "ml": "ml",
    "l": "litre",
    "litre": "litre",
    "liter": "litre",
    "litres": "litres",
    "liters": "litres",
    "pinch": "pinch",
    "pinches": "pinches",
    "clove": "clove",
    "cloves": "cloves",
    "slice": "slices",
    "slices": "slices",
    "piece": "pieces",
    "pieces": "pieces",
    "bunch": "bunch",
    "bunches": "bunches",
    "whole": "whole",
    "leaf": "leaves",
    "leaves": "leaves",
    "sheet": "sheets",
    "sheets": "sheets",
}


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    if not unit:
        return None
    key = str(unit).strip().lower().rstrip(".")
    return _UNIT_ALIASES.get(key, str(unit).strip())


@dataclass
class Ingredient:
    """A single ingredient with an exact quantity, unit and preparation."""

    name: str
    quantity: Any = None          # number, fraction string, or "to taste"
    unit: Optional[str] = None
    prep: Optional[str] = None    # "finely chopped", "grated", ...
    raw: str = ""

    def spoken(self) -> str:
        """Exact spoken form: quantity + unit + prep + name.

        ``0.5 tablespoon oil`` → ``"1/2 tablespoon oil"``;
        ``2 finely chopped onions`` → ``"2 finely chopped onions"``.
        """
        parts: list[str] = []
        qty = format_quantity(self.quantity)
        if qty:
            parts.append(qty)
        unit = normalize_unit(self.unit) if self.unit else None
        # "whole" adds nothing ("1 dried red chilli", not "1 whole ...").
        if unit and unit.lower() != "whole":
            parts.append(unit)
        prep = (self.prep or "").strip()
        # Short adjective preps go before the name ("2 finely chopped
        # onions"); long clauses go after ("250 grams cooked rice, cooled
        # to room temperature").
        if prep and len(prep.split()) <= 3:
            parts.append(prep)
            parts.append(self.name)
            return " ".join(p for p in parts if p).strip()
        parts.append(self.name)
        base = " ".join(p for p in parts if p).strip()
        return f"{base}, {prep}" if prep else base

    @classmethod
    def from_any(cls, value: Any) -> "Ingredient":
        """Build from a dict ``{name, quantity, unit, prep}`` or a string
        like ``"1/2 tablespoon oil"``."""
        if isinstance(value, dict):
            return cls(
                name=str(value.get("name") or value.get("ingredient") or "").strip(),
                quantity=value.get("quantity"),
                unit=value.get("unit"),
                prep=value.get("prep") or value.get("preparation"),
                raw=str(value),
            )
        return parse_ingredient(str(value))


# quantity token: "1/2", "0.5", "2"
_QTY = r"(?:\d+\s*/\s*\d+|\d+(?:\.\d+)?)"
_UNITS = (
    r"tablespoons?|tbsps?|teaspoons?|tsps?|cups?|grams?|kilograms?|kgs?|kg|"
    r"ml|litres?|liters?|l|pinch(?:es)?|cloves?|slices?|pieces?|bunch(?:es)?"
)
_PREP_WORDS = (
    r"finely chopped|roughly chopped|chopped|sliced|grated|minced|crushed|"
    r"diced|peeled|beaten|ground|roasted|boiled|cooked|soaked|"
    r"finely|thinly|medium|large|small"
)
_ING_RE = re.compile(
    rf"^\s*(?P<qty>{_QTY})?\s*(?P<unit>{_UNITS})?\s*(?P<prep>(?:{_PREP_WORDS})\s+)?(?P<name>.+?)\s*$",
    re.IGNORECASE,
)


def parse_ingredient(text: str) -> Ingredient:
    """Best-effort parse of a free-text ingredient line.

    Handles ``"1/2 tablespoon oil"``, ``"2 finely chopped onions"``,
    ``"Oil — 1/2 tablespoon"`` and ``"salt to taste"``.
    """
    raw = (text or "").strip()
    if not raw:
        return Ingredient(name="", raw=raw)
    # "Name — 1/2 tablespoon" / "Name: 1/2 tablespoon"
    m = re.match(r"^(?P<name>[^—:\-]+?)\s*[—:\-]\s*(?P<rest>.+)$", raw)
    if m and re.search(rf"{_QTY}", m.group("rest")):
        text = f"{m.group('rest').strip()} {m.group('name').strip()}"
    else:
        text = raw
    m = _ING_RE.match(text)
    if not m:
        return Ingredient(name=raw, raw=raw)
    qty = m.group("qty")
    if qty:
        if "/" in qty:
            num, den = qty.split("/")
            try:
                qty = float(num.strip()) / float(den.strip())
            except (ValueError, ZeroDivisionError):
                qty = qty.strip()
        else:
            try:
                qty = float(qty)
            except ValueError:
                pass
    name = (m.group("name") or "").strip()
    return Ingredient(
        name=name,
        quantity=qty,
        unit=(m.group("unit") or "").strip() or None,
        prep=(m.group("prep") or "").strip() or None,
        raw=raw,
    )


def parse_ingredients(values: Iterable[Any]) -> list[Ingredient]:
    out: list[Ingredient] = []
    for v in values or []:
        ing = Ingredient.from_any(v)
        if ing.name:
            out.append(ing)
    return out


def spoken_ingredient_list(ingredients: Iterable[Ingredient]) -> str:
    """Comma/and separated list of exact ingredients."""
    items = [i.spoken() for i in ingredients if i.name]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + ", and " + items[-1]


# ======================================================================
# Durations
# ======================================================================
_DUR_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?|\d+\s*/\s*\d+)\s*"
    r"(?P<unit>hours?|hrs?|minutes?|mins?|seconds?|secs?|"
    r"గంటలు|గంట|నిమిషాలు|నిమిషం|సెకన్లు|సెకను)",
    re.IGNORECASE,
)


def parse_duration_seconds(text: str) -> Optional[int]:
    """Return the first duration in ``text`` as seconds, or ``None``."""
    m = _DUR_RE.search(text or "")
    if not m:
        return None
    raw = m.group("num").replace(" ", "")
    try:
        if "/" in raw:
            num, den = raw.split("/")
            value = float(num) / float(den)
        else:
            value = float(raw)
    except (ValueError, ZeroDivisionError):
        return None
    unit = m.group("unit").lower()
    if any(u in unit for u in ("గంట", "hour", "hr")):
        return int(value * 3600)
    if any(u in unit for u in ("సెక", "sec")):
        return int(value)
    return int(value * 60)


# ======================================================================
# Heat inference
# ======================================================================
_HEAT_EXPLICIT: list[tuple[re.Pattern, str]] = [
    (re.compile(r"medium[\s-]*(?:to[\s-]*)?high|medium-high|మీడియం[-\s]*హై", re.I), "medium-high"),
    (re.compile(r"medium[\s-]*(?:to[\s-]*)?low|medium-low|మీడియం[-\s]*లో", re.I), "medium-low"),
    (re.compile(r"\bmedium\b|మీడియం", re.I), "medium"),
    (re.compile(r"\bhigh\b|హై\s*ఫ్లేమ్|full flame", re.I), "high"),
    (re.compile(r"\blow\b|లో\s*ఫ్లేమ్|simmer|slow cook", re.I), "low"),
]

# (regex, heat) technique heuristics, checked in order.
_HEAT_TECHNIQUE: list[tuple[re.Pattern, str]] = [
    (re.compile(r"to a boil|\bboil\b|rolling boil", re.I), "high"),
    (re.compile(r"pressure cook", re.I), "high"),
    (re.compile(r"deep[\s-]*fry", re.I), "medium-high"),
    (re.compile(r"simmer|slow[\s-]*cook|reduce the flame|low flame", re.I), "low"),
    (re.compile(r"saut[eé]|golden brown|splutter|temper|tadka|roast", re.I), "medium"),
    (re.compile(r"stir[\s-]*fry|fry", re.I), "medium"),
]


def infer_heat_level(text: str) -> Optional[str]:
    """Infer the flame level for a step from explicit words or technique."""
    t = text or ""
    for pattern, level in _HEAT_EXPLICIT:
        if pattern.search(t):
            return level
    for pattern, level in _HEAT_TECHNIQUE:
        if pattern.search(t):
            return level
    return None


def step_mentions_stove(text: str) -> bool:
    """True when a step involves a stove/burner (needs a heat level)."""
    return bool(
        re.search(
            r"stove|burner|flame|pan|pot|kadai|kadhai|wok|tawa|griddle|cooker|"
            r"fry|saut[eé]|boil|simmer|roast|heat|temper|splutter|tadka|"
            r"వేయించ|ఉడికించ|మగ్గించ|నానబెట్ట|మూత|మంట",
            text or "",
            re.IGNORECASE,
        )
    )


# ======================================================================
# Attention sensitivity
# ======================================================================
_ATTENTION_RE = re.compile(
    r"splutter|temper|tadka|mustard seeds|cumin seeds|caramel|deep[\s-]*fry|"
    r"stir[\s-]*fry|quickly|constantly stir|burn|fragrant|"
    r"వేగిపో|కాలిపో|దోరగా",
    re.IGNORECASE,
)


def requires_attention(text: str) -> bool:
    """Steps that need the user's eyes (tempering, fast frying, ...)."""
    return bool(_ATTENTION_RE.search(text or ""))


# ======================================================================
# Structured steps & recipes
# ======================================================================
@dataclass
class RecipeStep:
    index: int
    text: str
    duration_seconds: Optional[int] = None
    heat_level: Optional[str] = None
    requires_attention: bool = False
    # ``None`` = unknown, ``False`` = dataset says do not parallelize.
    parallelizable: Optional[bool] = None
    ingredients: list[Ingredient] = field(default_factory=list)
    utensils: list[str] = field(default_factory=list)
    dependencies: list[int] = field(default_factory=list)
    # A later step index that can safely run during this step's wait.
    parallel_with: Optional[int] = None

    @property
    def heat_phrase(self) -> Optional[str]:
        return heat_phrase(self.heat_level)

    def spoken_ingredients(self) -> str:
        return spoken_ingredient_list(self.ingredients)

    def spoken_utensils(self) -> str:
        names = [normalize_utensil(u) for u in self.utensils if u]
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        return ", ".join(names[:-1]) + ", and " + names[-1]


@dataclass
class Recipe:
    name: str
    steps: list[RecipeStep]
    ingredients: list[Ingredient] = field(default_factory=list)
    utensils: list[str] = field(default_factory=list)
    base_servings: int = 1
    target_servings: Optional[int] = None

    def step(self, index: int) -> Optional[RecipeStep]:
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def _all_ingredients(self) -> list[Ingredient]:
        """Unique ingredient objects (step ingredients may alias the
        recipe-level ones)."""
        seen: set[int] = set()
        out: list[Ingredient] = []
        for ing in list(self.ingredients) + [
            i for st in self.steps for i in st.ingredients
        ]:
            if id(ing) not in seen:
                seen.add(id(ing))
                out.append(ing)
        return out

    def scale_to(self, servings: int) -> None:
        """Scale every ingredient quantity to ``servings`` people."""
        if not servings or servings <= 0:
            return
        base = self.base_servings or 1
        # Undo a previous scale so repeated calls stay correct.
        if self.target_servings:
            _scale_ingredients(self._all_ingredients(), 1 / (self.target_servings / base))
        _scale_ingredients(self._all_ingredients(), servings / base)
        self.target_servings = servings


def _scale_ingredients(ingredients: list[Ingredient], factor: float) -> None:
    for ing in ingredients:
        q = ing.quantity
        if isinstance(q, (int, float)):
            scaled = q * factor
            ing.quantity = round(scaled, 1) if scaled < 10 else round(scaled)


_UTENSIL_WORDS = re.compile(
    r"\b(pan|pot|bowl|spoon|ladle|knife|cutting board|pressure cooker|cooker|"
    r"strainer|plate|cup|grater|blender|mixer|wok|tawa|griddle|kadai|kadhai|"
    r"tray|spatula|whisk|rolling pin)\b",
    re.IGNORECASE,
)


def _extract_utensils(text: str) -> list[str]:
    found: list[str] = []
    for m in _UTENSIL_WORDS.finditer(text or ""):
        eng = normalize_utensil(m.group(0))
        if eng not in found:
            found.append(eng)
    return found


def _normalize_step(index: int, raw: Any, recipe_ingredients: list[Ingredient]) -> RecipeStep:
    """Turn a string or dict step into a :class:`RecipeStep`."""
    if isinstance(raw, dict):
        text = str(raw.get("instruction") or raw.get("text") or raw.get("step") or "").strip()
        step = RecipeStep(index=index, text=text)
        if raw.get("duration_minutes") is not None:
            step.duration_seconds = int(float(raw["duration_minutes"]) * 60)
        # Heat: prefer the spoken Telugu phrase (dataset), then an explicit
        # level, then the internal phrase.
        spoken = raw.get("heat_level_spoken_telugu") or raw.get("heat_phrase")
        level = level_from_phrase(spoken) or level_from_phrase(raw.get("heat_level_internal"))
        explicit = raw.get("heat_level") or raw.get("heat")
        if explicit and str(explicit).strip().lower() in HEAT_LEVELS:
            level = str(explicit).strip().lower()
        step.heat_level = level
        if raw.get("requires_attention") is not None or raw.get("requires_close_attention") is not None:
            step.requires_attention = bool(
                raw.get("requires_attention", raw.get("requires_close_attention"))
            )
        if raw.get("can_parallelize") is not None:
            step.parallelizable = bool(raw.get("can_parallelize"))
        if raw.get("ingredients"):
            step.ingredients = parse_ingredients(raw["ingredients"])
        if raw.get("utensils"):
            step.utensils = [normalize_utensil(u) for u in raw["utensils"]]
        if raw.get("dependencies"):
            step.dependencies = [int(d) for d in raw["dependencies"]]
    else:
        text = str(raw).strip()
        step = RecipeStep(index=index, text=text)

    if step.duration_seconds is None:
        step.duration_seconds = parse_duration_seconds(step.text)
    if step.heat_level is None:
        step.heat_level = infer_heat_level(step.text)
    if not step.requires_attention:
        step.requires_attention = requires_attention(step.text)
    if not step.utensils:
        step.utensils = _extract_utensils(step.text)
    # Attach recipe-level ingredients that are named in this step text.
    if recipe_ingredients:
        for ing in recipe_ingredients:
            if not ing.name or ing in step.ingredients:
                continue
            if re.search(rf"\b{re.escape(ing.name)}\b", step.text, re.IGNORECASE):
                step.ingredients.append(ing)
    return step


def normalize_recipe(data: dict[str, Any] | None) -> Optional[Recipe]:
    """Normalize a raw recipe dict into a :class:`Recipe`.

    Accepts the existing ``{recipeName, steps:[str], stepIndex}`` shape as
    well as an optional structured form with ``ingredients``, ``utensils``
    and per-step objects.
    """
    if not data:
        return None
    raw_steps = list(data.get("steps") or [])
    if not raw_steps:
        return None
    recipe_ingredients = parse_ingredients(data.get("ingredients") or [])
    step_details = data.get("stepDetails") or {}
    steps: list[RecipeStep] = []
    for i, raw in enumerate(raw_steps):
        detail = raw
        if isinstance(raw, str) and step_details:
            extra = step_details.get(str(i)) or step_details.get(i)
            if isinstance(extra, dict):
                detail = {"instruction": raw, **extra}
        steps.append(_normalize_step(i, detail, recipe_ingredients))
    utensils: list[str] = []
    for u in data.get("utensils") or []:
        eng = normalize_utensil(u)
        if eng not in utensils:
            utensils.append(eng)
    for st in steps:
        for u in st.utensils:
            if u not in utensils:
                utensils.append(u)
    return Recipe(
        name=str(data.get("recipeName") or data.get("name") or "the dish"),
        steps=steps,
        ingredients=recipe_ingredients,
        utensils=utensils,
        base_servings=int(data.get("servings") or data.get("base_servings") or 1),
    )


# ======================================================================
# Parallel-task detection
# ======================================================================
_PARALLEL_KEYWORDS = re.compile(
    r"boil|water|soak|chop|prepare|grind|blend|mix|roast|make the|"
    r"కోయండి|నానబెట్ట|మరిగించ|సిద్ధం",
    re.IGNORECASE,
)
_ANAPHORA = re.compile(
    r"\b(add to|into the same|same pan|same pot|those|these|it into|"
    r"మిశ్రమంలో|అదే|వాటిలో)\b",
    re.IGNORECASE,
)
_MIN_PARALLEL_WAIT_S = 300  # 5 minutes


def find_parallel_step(
    recipe: Optional[Recipe],
    current_index: int,
    completed: Optional[set[int]] = None,
) -> Optional[int]:
    """Return a later step that can safely run during ``current_index``.

    Conservative on purpose — a candidate must:
    * not be completed already,
    * be independent: it must name a *different* utensil than the current
      step (so it needs its own pot/bowl/board), and not point back at the
      current step ("add to the onions"),
    * be a prep/boil-style task (chop, boil water, grind, mix, ...),
    and the current step must be a long, non-attention step.
    """
    if recipe is None:
        return None
    completed = completed or set()
    current = recipe.step(current_index)
    if current is None:
        return None
    if (current.duration_seconds or 0) < _MIN_PARALLEL_WAIT_S:
        return None
    if current.requires_attention:
        return None
    if current.parallelizable is False:
        return None
    current_utensils = set(current.utensils)
    for candidate in recipe.steps[current_index + 1:]:
        if candidate.index in completed:
            continue
        if candidate.index in current.dependencies:
            continue
        if _ANAPHORA.search(candidate.text):
            continue
        # Must clearly use its own (different) utensil — this is what makes
        # it safe to run in parallel.
        if not candidate.utensils:
            continue
        if set(candidate.utensils) & current_utensils:
            continue
        if not _PARALLEL_KEYWORDS.search(candidate.text):
            continue
        return candidate.index
    return None


def annotate_parallel_steps(recipe: Optional[Recipe], completed: Optional[set[int]] = None) -> None:
    """Set ``parallel_with`` on each step (best-effort, idempotent)."""
    if recipe is None:
        return
    completed = completed or set()
    for step in recipe.steps:
        step.parallel_with = find_parallel_step(recipe, step.index, completed)


__all__ = [
    "HEAT_LEVELS",
    "HEAT_PHRASE_TE",
    "HEAT_PHRASE_BY_TE",
    "heat_phrase",
    "level_from_phrase",
    "normalize_utensil",
    "format_quantity",
    "normalize_unit",
    "Ingredient",
    "parse_ingredient",
    "parse_ingredients",
    "spoken_ingredient_list",
    "parse_duration_seconds",
    "infer_heat_level",
    "step_mentions_stove",
    "requires_attention",
    "RecipeStep",
    "Recipe",
    "normalize_recipe",
    "find_parallel_step",
    "annotate_parallel_steps",
]
