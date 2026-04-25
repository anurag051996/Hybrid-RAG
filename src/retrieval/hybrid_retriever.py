"""
src/retrieval/hybrid_retriever.py
───────────────────────────────────
Orchestrates dense (vector) and sparse (BM25) retrievers,
then fuses results with Reciprocal Rank Fusion.
"""

import logging
from typing import List

from src.retrieval.vector_store import VectorStore, SearchResult
from src.retrieval.bm25_retriever import BM25Retriever
from src.reranking.rrf_reranker import RRFReranker
from config.settings import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid retriever = VectorSearch + BM25 + RRF fusion.

    Usage:
        retriever = HybridRetriever(vector_store, bm25_retriever)
        results = retriever.retrieve("What is RAG?")
    """

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        rrf_k: int | None = None,
    ):
        self.vector_store = vector_store
        self.bm25 = bm25_retriever
        self.reranker = RRFReranker(k=rrf_k or settings.rrf_k)

    def retrieve(
        self,
        query: str,
        top_k_vector: int | None = None,
        top_k_bm25: int | None = None,
        top_k_final: int | None = None,
    ) -> List[SearchResult]:
        """
        Run hybrid retrieval for a query.

        Steps:
            1. Dense vector search  → top_k_vector candidates
            2. BM25 sparse search   → top_k_bm25 candidates
            3. RRF fusion           → top_k_final re-ranked results

        Returns:
            Final ranked list of SearchResult objects.
        """
        top_k_vector = top_k_vector or settings.top_k_vector
        top_k_bm25 = top_k_bm25 or settings.top_k_bm25
        top_k_final = top_k_final or settings.top_k_final

        logger.info("Hybrid retrieval for query: '%s'", query[:80])

        # ── Step 1: Dense Search ──────────────────────────────────────────
        vector_results = self.vector_store.search(query, top_k=top_k_vector)
        logger.debug("Vector search returned %d results", len(vector_results))

        # ── Step 2: BM25 Sparse Search ────────────────────────────────────
        if not self.bm25.is_ready:
            logger.warning("BM25 index not ready — using vector results only.")
            return vector_results[:top_k_final]

        bm25_results = self.bm25.search(query, top_k=top_k_bm25)
        logger.debug("BM25 search returned %d results", len(bm25_results))

        # ── Step 3: RRF Fusion ────────────────────────────────────────────
        fused = self.reranker.fuse(
            ranked_lists=[vector_results, bm25_results],
            top_k=top_k_final,
        )
        logger.info("RRF fusion returned %d final results", len(fused))

        return fused
