"""Chunking, embedding, and pgvector ranking for filing/news retrieval —
owned by Track B (PRD §7.2).

Work items (see implementation plan, Track B):
1. Chunking: 500-token chunks, 50-token overlap, over filing text.
2. Embedding: wrap Voyage AI (voyage-3 / voyage-finance-2). Ship a
   FakeEmbedder (deterministic hash-based vectors) alongside the real one so
   ranking logic is unit-testable without a Voyage key.
3. Ranking: score = cosine_similarity * recency_decay(source_date), top 3-5
   chunks. Candidates come from Database.search_chunks (cosine ANN via
   pgvector); this module applies the recency-decay re-rank on top.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from equity_ensemble.schemas.models import RetrievalChunk

CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
TOP_K_MIN, TOP_K_MAX = 3, 5


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class VoyageEmbedder:
    def __init__(self, api_key: str | None = None, model: str = "voyage-finance-2") -> None:
        raise NotImplementedError("Track B: wrap voyageai client")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Track B: call Voyage embeddings API")


class FakeEmbedder:
    """Deterministic hash-based embedder for tests — no network, no key."""

    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError("Track B: deterministic hash(text) -> unit vector of `dimension`")


def chunk_text(
    *, ticker: str, source_type: str, source_date: date, section: str | None, text: str
) -> list[str]:
    """Split `text` into ~CHUNK_TOKENS-token chunks with CHUNK_OVERLAP_TOKENS
    overlap. Track B: implement + unit test against a sample 10-K fixture."""
    raise NotImplementedError("Track B: implement chunking")


def recency_decay(
    source_date: date, *, as_of: date | None = None, half_life_days: int = 90
) -> float:
    """Exponential decay weight in (0, 1], newer sources score closer to 1.
    Track B: implement and unit-test the ordering it produces."""
    raise NotImplementedError("Track B: implement recency decay")


def rank_chunks(
    candidates: list[tuple[RetrievalChunk, float]], *, as_of: date | None = None
) -> list[RetrievalChunk]:
    """`candidates` are (chunk, cosine_similarity) pairs from
    Database.search_chunks. Returns the top 3-5 by
    cosine_similarity * recency_decay(source_date) (PRD §7.2)."""
    raise NotImplementedError("Track B: implement ranking + top-k selection")
