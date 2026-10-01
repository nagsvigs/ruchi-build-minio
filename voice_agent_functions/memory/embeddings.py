"""Local text embeddings via ``fastembed`` (ONNX, no API key).

The default model is the 384-dim multilingual
``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``, which
handles Telugu + English queries against English memory strings.

Both helpers never raise: on any failure they log and return ``None`` /
``[]`` so the voice loop keeps running without memory.

Set ``EMBEDDING_QUERY_PREFIX`` / ``EMBEDDING_PASSAGE_PREFIX`` if you
switch to an E5-style model that expects ``query:`` / ``passage:``.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Iterable, Sequence

logger = logging.getLogger("ruchi.memory.embeddings")

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DIMENSIONS = 384

_model = None
_lock = threading.Lock()


def model_name() -> str:
    return os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)


def configured_dimensions() -> int:
    raw = os.getenv("EMBEDDING_DIMENSIONS")
    try:
        value = int(raw) if raw else DEFAULT_DIMENSIONS
        return value if value > 0 else DEFAULT_DIMENSIONS
    except (TypeError, ValueError):
        return DEFAULT_DIMENSIONS


def _query_prefix() -> str:
    return os.getenv("EMBEDDING_QUERY_PREFIX", "")


def _passage_prefix() -> str:
    return os.getenv("EMBEDDING_PASSAGE_PREFIX", "")


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _lock:
        if _model is None:
            from fastembed import TextEmbedding

            logger.info("loading embedding model %s", model_name())
            _model = TextEmbedding(
                model_name=model_name(),
                cache_dir=os.getenv("EMBEDDING_CACHE_DIR", "data/fastembed"),
            )
    return _model


def warmup() -> bool:
    """Load the model eagerly (used at startup). Returns success."""
    try:
        _get_model()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("embedding warmup failed: %s", exc)
        return False


def embed_for_storage(texts: Sequence[str] | Iterable[str]) -> list[list[float]]:
    """Embed documents for storage. Returns ``[]`` on failure."""
    items = [f"{_passage_prefix()}{t}".strip() for t in texts]
    if not items:
        return []
    try:
        model = _get_model()
        return [[float(x) for x in vec] for vec in model.embed(items)]
    except Exception as exc:  # noqa: BLE001
        logger.warning("embed_for_storage failed: %s", exc)
        return []


def embed_for_query(text: str) -> list[float] | None:
    """Embed a search query. Returns ``None`` on failure."""
    if not text or not text.strip():
        return None
    try:
        model = _get_model()
        vec = next(iter(model.embed([f"{_query_prefix()}{text}".strip()])))
        return [float(x) for x in vec]
    except Exception as exc:  # noqa: BLE001
        logger.warning("embed_for_query failed: %s", exc)
        return None


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_DIMENSIONS",
    "model_name",
    "configured_dimensions",
    "warmup",
    "embed_for_storage",
    "embed_for_query",
]
