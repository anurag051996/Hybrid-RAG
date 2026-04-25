"""
src/reranking/rrf_reranker.py
──────────────────────────────
Reciprocal Rank Fusion (RRF) for combining multiple ranked lists.

Reference:
    Cormack, Clarke & Buettcher (2009).
    "Reciprocal rank fusion outperforms condorcet and individual rank learning methods."
    SIGIR '09.

Formula:
    RRF(d) = Σ_r  1 / (k + rank_r(d))

    where k=60 is the standard smoothing constant,
    rank_r(d) is the 1-indexed rank of document d in ranker r.
"""

import logging
from typing import List, Dict

from src.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)


class RRFReranker:
    """
    Fuses N ranked lists into a single ranked list using RRF.

    Usage:
        reranker = RRFReranker(k=60)
        fused = reranker.fuse([vector_results, bm25_results], top_k=5)
    """

    def __init__(self, k: int = 60):
        """
        Args:
            k: RRF constant. Higher k → less reward for top-ranked docs.
               k=60 is the empirically optimal value in the original paper.
        """
        self.k = k

    def fuse(
        self,
        ranked_lists: List[List[SearchResult]],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """
        Merge multiple ranked lists and return the top_k by RRF score.

        Args:
            ranked_lists: Each list is assumed to be sorted best-first.
            top_k: Number of results to return.

        Returns:
            Re-ranked list of SearchResult with rrf_score in metadata.
        """
        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, SearchResult] = {}

        for ranked_list in ranked_lists:
            for rank, result in enumerate(ranked_list, start=1):
                # Use text as dedup key (chunk_ids may differ between retrievers)
                key = result.text[:200]
                rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (self.k + rank)
                if key not in doc_lookup:
                    doc_lookup[key] = result

        # Sort by descending RRF score
        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
        top_keys = sorted_keys[:top_k]

        fused_results = []
        for key in top_keys:
            result = doc_lookup[key]
            result.metadata["rrf_score"] = round(rrf_scores[key], 6)
            result.score = rrf_scores[key]
            fused_results.append(result)

        logger.debug(
            "RRF fused %d lists → %d unique docs → returning top %d",
            len(ranked_lists),
            len(rrf_scores),
            len(fused_results),
        )
        return fused_results
