from __future__ import annotations

from datetime import date, timedelta

import pytest

from equity_ensemble.data.retrieval import (
    RECENCY_HALF_LIFE_DAYS,
    TOP_K_MAX,
    FakeEmbedder,
    build_chunks,
    chunk_text,
    cosine_similarity,
    rank_chunks,
    recency_decay,
)
from equity_ensemble.schemas.models import RetrievalChunk


def _chunk(chunk_id: str, embedding: list[float], source_date: date) -> RetrievalChunk:
    return RetrievalChunk(
        chunk_id=chunk_id,
        ticker="AAPL",
        source_type="10-K",
        source_date=source_date,
        section=None,
        text=f"text for {chunk_id}",
        embedding=embedding,
    )


class TestChunkText:
    def test_empty_text_yields_no_chunks(self):
        assert chunk_text("") == []

    def test_short_text_yields_one_chunk(self):
        text = " ".join(["word"] * 10)
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_is_split_with_overlap(self):
        words = [f"w{i}" for i in range(1200)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_tokens=500, overlap_tokens=50)
        assert len(chunks) > 1
        # consecutive chunks share the overlap region
        first_words = chunks[0].split()
        second_words = chunks[1].split()
        assert first_words[-50:] == second_words[:50]

    def test_last_chunk_reaches_the_end_of_text(self):
        words = [f"w{i}" for i in range(1200)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_tokens=500, overlap_tokens=50)
        assert chunks[-1].split()[-1] == "w1199"


class TestRecencyDecay:
    def test_today_is_full_weight(self):
        assert recency_decay(date.today(), as_of=date.today()) == pytest.approx(1.0)

    def test_half_life_days_ago_is_half_weight(self):
        as_of = date.today()
        source = as_of - timedelta(days=RECENCY_HALF_LIFE_DAYS)
        assert recency_decay(source, as_of=as_of) == pytest.approx(0.5, abs=1e-6)

    def test_future_date_treated_as_zero_age(self):
        as_of = date.today()
        source = as_of + timedelta(days=10)
        assert recency_decay(source, as_of=as_of) == pytest.approx(1.0)


class TestCosineSimilarity:
    def test_identical_vectors_are_similarity_one(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_are_similarity_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-9)

    def test_opposite_vectors_are_similarity_negative_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


class TestRankChunks:
    def test_orders_by_similarity_times_recency(self):
        today = date.today()
        query = [1.0, 0.0]
        best = _chunk("best", [1.0, 0.0], today)
        stale_but_similar = _chunk("stale", [1.0, 0.0], today - timedelta(days=365))
        dissimilar = _chunk("dissimilar", [0.0, 1.0], today)

        ranked = rank_chunks(query, [dissimilar, stale_but_similar, best])

        assert ranked[0].chunk_id == "best"
        assert ranked[-1].chunk_id == "dissimilar"

    def test_caps_at_top_k_max(self):
        today = date.today()
        candidates = [_chunk(f"c{i}", [1.0, 0.0], today) for i in range(TOP_K_MAX + 5)]
        ranked = rank_chunks([1.0, 0.0], candidates)
        assert len(ranked) == TOP_K_MAX

    def test_empty_candidates_returns_empty(self):
        assert rank_chunks([1.0, 0.0], []) == []


class TestFakeEmbedder:
    async def test_identical_text_yields_identical_vector(self):
        embedder = FakeEmbedder(dimension=64)
        [a] = await embedder.embed(["hello world"])
        [b] = await embedder.embed(["hello world"])
        assert a == b

    async def test_different_text_yields_different_vector(self):
        embedder = FakeEmbedder(dimension=64)
        [a] = await embedder.embed(["hello world"])
        [b] = await embedder.embed(["goodbye world"])
        assert a != b

    async def test_vectors_are_unit_norm(self):
        import numpy as np

        embedder = FakeEmbedder(dimension=64)
        [v] = await embedder.embed(["hello world"])
        assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)


class TestBuildChunksMaxChunks:
    async def test_max_chunks_truncates_before_embedding(self):
        text = " ".join(f"w{i}" for i in range(3000))  # many chunks worth
        embedder = FakeEmbedder(dimension=8)

        chunks = await build_chunks(
            ticker="AAPL",
            source_type="10-Q",
            source_date=date(2026, 1, 1),
            section=None,
            text=text,
            embedder=embedder,
            max_chunks=3,
        )

        assert len(chunks) == 3

    async def test_no_cap_embeds_every_chunk(self):
        text = " ".join(f"w{i}" for i in range(1200))
        embedder = FakeEmbedder(dimension=8)

        chunks = await build_chunks(
            ticker="AAPL",
            source_type="10-Q",
            source_date=date(2026, 1, 1),
            section=None,
            text=text,
            embedder=embedder,
        )

        assert len(chunks) == len(chunk_text(text))

    async def test_cap_larger_than_available_chunks_is_a_no_op(self):
        text = " ".join(f"w{i}" for i in range(10))
        embedder = FakeEmbedder(dimension=8)

        chunks = await build_chunks(
            ticker="AAPL",
            source_type="10-Q",
            source_date=date(2026, 1, 1),
            section=None,
            text=text,
            embedder=embedder,
            max_chunks=100,
        )

        assert len(chunks) == 1
