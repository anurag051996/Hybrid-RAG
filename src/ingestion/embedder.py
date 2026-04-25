"""
src/ingestion/embedder.py
──────────────────────────
Generates dense vector embeddings using sentence-transformers.
"""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from config.settings import settings

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps a HuggingFace sentence-transformer model.

    Usage:
        embedder = Embedder()
        vectors = embedder.embed(["text one", "text two"])
    """

    def __init__(self, model_name: str | None = None):
        model_name = model_name or settings.embedding_model
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %d", self.dimension)

    def embed(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> List[List[float]]:
        """
        Embed a list of texts.

        Returns:
            List of float vectors, one per input text.
        """
        if not texts:
            return []
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity via dot product
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        return self.embed([query])[0]
