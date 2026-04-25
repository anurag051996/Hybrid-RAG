"""
src/retrieval/vector_store.py
──────────────────────────────
ChromaDB-backed vector store for dense retrieval.
"""

import logging
from dataclasses import dataclass, field
from typing import List

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from src.ingestion.chunker import Chunk
from src.ingestion.embedder import Embedder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single retrieval result with score."""
    chunk_id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """
    Persistent ChromaDB collection for dense vector retrieval.

    Usage:
        store = VectorStore(embedder)
        store.add_chunks(chunks)
        results = store.search("query text", top_k=10)
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB collection '%s' ready — %d existing docs",
            settings.chroma_collection,
            self.collection.count(),
        )

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 100) -> None:
        """Embed and upsert chunks into ChromaDB."""
        if not chunks:
            logger.warning("No chunks to add.")
            return

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            ids = [c.chunk_id for c in batch]
            metadatas = [c.metadata for c in batch]
            embeddings = self.embedder.embed(texts, show_progress=True)

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            logger.info("Upserted batch %d/%d", i // batch_size + 1, -(-len(chunks) // batch_size))

        logger.info("VectorStore now contains %d documents.", self.collection.count())

    def search(self, query: str, top_k: int | None = None) -> List[SearchResult]:
        """Dense retrieval: returns top_k results ranked by cosine similarity."""
        top_k = top_k or settings.top_k_vector
        query_embedding = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance → similarity score
            score = 1.0 - dist
            search_results.append(SearchResult(
                chunk_id=meta.get("chunk_index", ""),
                text=doc,
                score=score,
                metadata=meta,
            ))

        return search_results

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self.client.delete_collection(settings.chroma_collection)
        self.collection = self.client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Collection '%s' has been reset.", settings.chroma_collection)
