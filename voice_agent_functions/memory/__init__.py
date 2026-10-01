"""Ruchi's persistent memory layer.

Postgres + pgvector for:
* sessions + conversation history (survives restarts),
* long-term memories (semantic search + dedup),
* recipe RAG (semantic recipe lookup).

Embeddings run locally via ``fastembed`` (no API key). Everything is
defensive: if the DB or embeddings are unavailable the voice loop keeps
working with no memory.
"""
from .database import (
    DatabaseError,
    bootstrap_schema,
    close_pool,
    connection,
    get_pool,
    vector_dimensions,
)

__all__ = [
    "DatabaseError",
    "bootstrap_schema",
    "close_pool",
    "connection",
    "get_pool",
    "vector_dimensions",
]
