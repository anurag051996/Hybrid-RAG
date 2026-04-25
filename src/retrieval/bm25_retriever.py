"""
src/retrieval/bm25_retriever.py
────────────────────────────────
BM25 sparse retriever using rank-bm25.

BM25 excels at exact keyword matching and is a strong
complement to dense vector search.
"""

import logging
import pickle
import string
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from config.settings import settings
from src.ingestion.chunker import Chunk
from src.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(".bm25_index.pkl")


class BM25Retriever:
    """
    In-memory BM25 index over text chunks.
    Persists to disk so re-ingestion is fast.

    Usage:
        retriever = BM25Retriever()
        retriever.build(chunks)
        results = retriever.search("query", top_k=10)
    """

    def __init__(self):
        self._chunks: List[Chunk] = []
        self._bm25: BM25Okapi | None = None

    # ── Index Building ────────────────────────────────────────────────────

    def build(self, chunks: List[Chunk]) -> None:
        """Tokenize chunks and build BM25 index."""
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunk list.")

        self._chunks = chunks
        tokenized = [self._tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built over %d chunks.", len(chunks))
        self._save()

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase + remove punctuation + split on whitespace."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text.split()

    # ── Search ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int | None = None) -> List[SearchResult]:
        """Return top_k chunks ranked by BM25 score."""
        if self._bm25 is None:
            raise RuntimeError("BM25 index is not built. Call build() first.")

        top_k = top_k or settings.top_k_bm25
        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)

        # Get indices sorted by descending score
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_indices = ranked_indices[:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            results.append(SearchResult(
                chunk_id=self._chunks[idx].chunk_id,
                text=self._chunks[idx].text,
                score=float(scores[idx]),
                metadata={**self._chunks[idx].metadata, "bm25_rank": rank + 1},
            ))
        return results

    # ── Persistence ───────────────────────────────────────────────────────

    def _save(self, path: Path = _CACHE_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "bm25": self._bm25}, f)
        logger.info("BM25 index saved to %s", path)

    def load(self, path: Path = _CACHE_PATH) -> bool:
        """Load a previously saved index. Returns True if successful."""
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._chunks = data["chunks"]
            self._bm25 = data["bm25"]
            logger.info("BM25 index loaded from %s (%d chunks)", path, len(self._chunks))
            return True
        except Exception as e:
            logger.error("Failed to load BM25 index: %s", e)
            return False

    @property
    def is_ready(self) -> bool:
        return self._bm25 is not None
