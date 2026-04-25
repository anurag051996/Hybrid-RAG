"""
tests/test_retrieval.py
────────────────────────
Unit tests for BM25, VectorStore, HybridRetriever, and RRF.
Uses mocks to avoid requiring real embeddings or a running ChromaDB.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import SearchResult
from src.retrieval.bm25_retriever import BM25Retriever
from src.reranking.rrf_reranker import RRFReranker
from src.retrieval.hybrid_retriever import HybridRetriever


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_chunks(n: int = 5) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"doc::chunk_{i}",
            text=f"This is sample chunk number {i} about retrieval augmented generation.",
            metadata={"source": f"doc_{i}.txt", "chunk_index": i},
        )
        for i in range(n)
    ]


def make_results(texts: list[str]) -> list[SearchResult]:
    return [
        SearchResult(chunk_id=f"id_{i}", text=t, score=1.0 / (i + 1), metadata={})
        for i, t in enumerate(texts)
    ]


# ── BM25 Tests ────────────────────────────────────────────────────────────────

class TestBM25Retriever:
    def test_build_and_search(self):
        chunks = make_chunks(10)
        retriever = BM25Retriever()
        retriever.build(chunks)

        assert retriever.is_ready

        results = retriever.search("retrieval augmented generation", top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_returns_ranked_results(self):
        chunks = [
            Chunk("a", "BM25 is a keyword ranking function", {"source": "a.txt"}),
            Chunk("b", "Neural networks learn dense embeddings", {"source": "b.txt"}),
            Chunk("c", "BM25 outperforms TF-IDF on sparse retrieval", {"source": "c.txt"}),
        ]
        retriever = BM25Retriever()
        retriever.build(chunks)

        results = retriever.search("BM25 keyword ranking", top_k=3)
        # BM25-relevant docs should rank higher
        assert results[0].text in [chunks[0].text, chunks[2].text]

    def test_empty_chunks_raises(self):
        retriever = BM25Retriever()
        with pytest.raises(ValueError, match="empty"):
            retriever.build([])

    def test_not_ready_raises_on_search(self):
        retriever = BM25Retriever()
        with pytest.raises(RuntimeError, match="not built"):
            retriever.search("query")

    def test_tokenizer_lowercases_and_strips_punctuation(self):
        retriever = BM25Retriever()
        tokens = retriever._tokenize("Hello, World! This is BM25.")
        assert "hello" in tokens
        assert "bm25" in tokens
        assert "," not in tokens
        assert "." not in tokens


# ── RRF Tests ─────────────────────────────────────────────────────────────────

class TestRRFReranker:
    def test_basic_fusion(self):
        list1 = make_results(["doc_A", "doc_B", "doc_C"])
        list2 = make_results(["doc_B", "doc_D", "doc_A"])

        reranker = RRFReranker(k=60)
        fused = reranker.fuse([list1, list2], top_k=3)

        assert len(fused) == 3
        # doc_A and doc_B appear in both lists → should score highest
        top_texts = [r.text for r in fused[:2]]
        assert "doc_A" in top_texts
        assert "doc_B" in top_texts

    def test_rrf_scores_are_positive(self):
        results = make_results(["x", "y", "z"])
        reranker = RRFReranker()
        fused = reranker.fuse([results], top_k=3)
        assert all(r.score > 0 for r in fused)

    def test_top_k_respected(self):
        results = make_results([f"doc_{i}" for i in range(20)])
        reranker = RRFReranker()
        fused = reranker.fuse([results], top_k=5)
        assert len(fused) == 5

    def test_deduplication(self):
        """Same doc appearing in both lists should appear only once in output."""
        shared = SearchResult("id_1", "shared document text", 0.9, {})
        list1 = [shared, SearchResult("id_2", "other doc A", 0.8, {})]
        list2 = [shared, SearchResult("id_3", "other doc B", 0.7, {})]

        reranker = RRFReranker()
        fused = reranker.fuse([list1, list2], top_k=10)

        texts = [r.text for r in fused]
        assert texts.count("shared document text") == 1

    def test_higher_k_dampens_top_ranks(self):
        results = make_results(["a", "b", "c"])
        low_k = RRFReranker(k=1).fuse([results], top_k=1)[0].score
        high_k = RRFReranker(k=1000).fuse([results], top_k=1)[0].score
        assert low_k > high_k


# ── HybridRetriever Tests ─────────────────────────────────────────────────────

class TestHybridRetriever:
    def _make_retriever(self, vector_results, bm25_results):
        mock_vs = MagicMock()
        mock_vs.search.return_value = vector_results

        mock_bm25 = MagicMock()
        mock_bm25.is_ready = True
        mock_bm25.search.return_value = bm25_results

        return HybridRetriever(mock_vs, mock_bm25)

    def test_returns_fused_results(self):
        vr = make_results(["vec_A", "vec_B", "vec_C"])
        br = make_results(["bm25_A", "vec_A", "bm25_B"])

        retriever = self._make_retriever(vr, br)
        results = retriever.retrieve("test query", top_k_final=4)

        assert len(results) <= 4
        # vec_A appears in both → should be in results
        texts = [r.text for r in results]
        assert "vec_A" in texts

    def test_falls_back_to_vector_when_bm25_not_ready(self):
        vr = make_results(["vec_A", "vec_B"])
        mock_vs = MagicMock()
        mock_vs.search.return_value = vr

        mock_bm25 = MagicMock()
        mock_bm25.is_ready = False

        retriever = HybridRetriever(mock_vs, mock_bm25)
        results = retriever.retrieve("query", top_k_final=2)

        assert results == vr[:2]
        mock_bm25.search.assert_not_called()
