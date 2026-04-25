"""
src/generation/rag_pipeline.py
────────────────────────────────
End-to-end RAG pipeline: retrieval → prompt → LLM → answer.
Supports both OpenAI and Anthropic as LLM backends.
"""

import logging
from typing import List

from config.settings import settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.vector_store import SearchResult

logger = logging.getLogger(__name__)

# ── Prompt Template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise and helpful assistant.
Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer, say so clearly.
Do not make up information."""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """
    Full RAG pipeline: hybrid retrieval + LLM generation.

    Usage:
        pipeline = RAGPipeline(retriever)
        response = pipeline.run("What is BM25?")
        print(response.answer)
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self._llm_client = None

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, question: str, top_k_final: int | None = None) -> "RAGResponse":
        """
        Execute the full RAG pipeline for a question.

        Returns:
            RAGResponse with answer, sources, and retrieved chunks.
        """
        settings.validate()

        # 1. Retrieve
        chunks = self.retriever.retrieve(question, top_k_final=top_k_final)
        if not chunks:
            return RAGResponse(
                question=question,
                answer="No relevant documents found in the knowledge base.",
                sources=[],
                chunks=[],
            )

        # 2. Build prompt
        context = self._build_context(chunks)
        prompt = USER_PROMPT_TEMPLATE.format(context=context, question=question)

        # 3. Generate
        answer = self._generate(prompt)

        # 4. Extract source references
        sources = list({c.metadata.get("source", "unknown") for c in chunks})

        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            chunks=chunks,
        )

    # ── Context Building ──────────────────────────────────────────────────

    def _build_context(self, chunks: List[SearchResult]) -> str:
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source", "unknown")
            parts.append(f"[{i}] (source: {source})\n{chunk.text}")
        return "\n\n".join(parts)

    # ── LLM Dispatch ─────────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        provider = settings.llm_provider.lower()
        if provider == "openai":
            return self._generate_openai(prompt)
        elif provider == "anthropic":
            return self._generate_anthropic(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: '{provider}'. Use 'openai' or 'anthropic'.")

    def _generate_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def _generate_anthropic(self, prompt: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        response = client.messages.create(
            model=settings.llm_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()


# ── Response Dataclass ────────────────────────────────────────────────────────

class RAGResponse:
    def __init__(
        self,
        question: str,
        answer: str,
        sources: List[str],
        chunks: List[SearchResult],
    ):
        self.question = question
        self.answer = answer
        self.sources = sources
        self.chunks = chunks

    def __repr__(self) -> str:
        return (
            f"RAGResponse(\n"
            f"  question={self.question!r}\n"
            f"  answer={self.answer[:100]!r}...\n"
            f"  sources={self.sources}\n"
            f")"
        )
