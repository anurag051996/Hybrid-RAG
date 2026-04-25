"""
config/settings.py
──────────────────
Centralised configuration loaded from environment variables.
All values can be overridden via .env file.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # ── LLM ───────────────────────────────────────────
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # ── Embeddings ────────────────────────────────────
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )

    # ── Chunking ──────────────────────────────────────
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", 512)))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", 64)))

    # ── Retrieval ─────────────────────────────────────
    top_k_vector: int = field(default_factory=lambda: int(os.getenv("TOP_K_VECTOR", 10)))
    top_k_bm25: int = field(default_factory=lambda: int(os.getenv("TOP_K_BM25", 10)))
    top_k_final: int = field(default_factory=lambda: int(os.getenv("TOP_K_FINAL", 5)))
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", 60)))

    # ── Vector Store ──────────────────────────────────
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", ".chromadb")
    )
    chroma_collection: str = field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION", "hybrid_rag")
    )

    def validate(self) -> None:
        """Raise if required secrets are missing."""
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set in your .env file.")
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set in your .env file.")


# Singleton — import this everywhere
settings = Settings()
