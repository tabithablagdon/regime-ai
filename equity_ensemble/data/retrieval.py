"""Chunking, embedding, and recency-weighted ranking for filing/news
retrieval — Track B (PRD §7.2).

`Database.search_chunks` (persistence/db.py) returns ANN candidates by
cosine distance; this module re-ranks those candidates by
`cosine_similarity * recency_decay(source_date)` in Python, using each
chunk's own stored embedding — no change to the Database protocol needed.
"""

from __future__ import annotations

import hashlib
import os
from datetime import date
from typing import Protocol

import numpy as np

from equity_ensemble.schemas.models import RetrievalChunk

CHUNK_TOKENS = 500
CHUNK_OVERLAP_TOKENS = 50
TOP_K_MAX = 5
RECENCY_HALF_LIFE_DAYS = 90


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class VoyageEmbedder:
    """Real embedder, wrapping Voyage AI (PRD §5)."""

    def __init__(self, *, api_key: str | None = None, model: str = "voyage-finance-2") -> None:
        import voyageai  # local import: keeps voyageai optional for pure unit tests

        self._client = voyageai.AsyncClient(api_key=api_key or os.environ["VOYAGE_API_KEY"])
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        result = await self._client.embed(texts, model=self._model, input_type="document")
        return result.embeddings


class FakeEmbedder:
    """Deterministic, content-sensitive pseudo-embedder for tests — no
    network call, no Voyage key. Identical text always yields the identical
    vector; different text yields a (whp) different one, so ranking-logic
    tests exercise real cosine-similarity behavior."""

    def __init__(self, dimension: int = 1024) -> None:
        self.dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        vec = np.random.default_rng(seed).normal(size=self.dimension)
        return (vec / np.linalg.norm(vec)).tolist()


def chunk_text(
    text: str, *, chunk_tokens: int = CHUNK_TOKENS, overlap_tokens: int = CHUNK_OVERLAP_TOKENS
) -> list[str]:
    """Split `text` into ~`chunk_tokens`-token chunks with `overlap_tokens`
    overlap. Whitespace-delimited words stand in for tokens — a reasonable
    MVP approximation that avoids pulling in a full tokenizer dependency."""
    words = text.split()
    if not words:
        return []

    step = max(chunk_tokens - overlap_tokens, 1)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + chunk_tokens]))
        if start + chunk_tokens >= len(words):
            break
        start += step
    return chunks


async def build_chunks(
    *,
    ticker: str,
    source_type: str,
    source_date: date,
    section: str | None,
    text: str,
    embedder: Embedder,
) -> list[RetrievalChunk]:
    """Chunk `text` and embed every chunk, returning ready-to-persist
    `RetrievalChunk` rows."""
    texts = chunk_text(text)
    if not texts:
        return []
    embeddings = await embedder.embed(texts)
    return [
        RetrievalChunk(
            chunk_id=f"{ticker}:{source_type}:{source_date.isoformat()}:{i}",
            ticker=ticker,
            source_type=source_type,
            source_date=source_date,
            section=section,
            text=chunk,
            embedding=embedding,
        )
        for i, (chunk, embedding) in enumerate(zip(texts, embeddings, strict=True))
    ]


def recency_decay(
    source_date: date, *, as_of: date | None = None, half_life_days: int = RECENCY_HALF_LIFE_DAYS
) -> float:
    """Exponential decay weight in (0, 1]; newer sources score closer to 1."""
    as_of = as_of or date.today()
    age_days = max((as_of - source_date).days, 0)
    return 0.5 ** (age_days / half_life_days)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a_arr) * np.linalg.norm(b_arr))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)


def rank_chunks(
    query_embedding: list[float], candidates: list[RetrievalChunk], *, as_of: date | None = None
) -> list[RetrievalChunk]:
    """`score = cosine_similarity x recency_decay(source_date)`, top chunks
    first, capped at `TOP_K_MAX` (PRD §7.2). Deliberately simpler than the
    north star's MMR diversity re-ranking (revisit if duplicate/redundant
    chunks show up in practice, per the PRD's own note)."""
    scored = [
        (
            chunk,
            cosine_similarity(query_embedding, chunk.embedding)
            * recency_decay(chunk.source_date, as_of=as_of),
        )
        for chunk in candidates
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [chunk for chunk, _ in scored[:TOP_K_MAX]]
