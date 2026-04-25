"""
tests/test_pipeline.py
───────────────────────
Integration-style tests for RAGPipeline.
LLM calls are mocked so no API key is needed.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.generation.rag_pipeline import RAGPipeline, RAGResponse
from src.retrieval.vector_store import SearchResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_search_results(n: int = 3) -> list[SearchResult]:
    return [
        SearchResult(
            chunk_id=f"chunk_{i}",
            text=f"Relevant passage {i}: RAG combines retrieval with generation.",
            score=1.0 / (i + 1),
            metadata={"source": f"doc_{i}.txt", "rrf_score": 0.02},
        )
        for i in range(n)
    ]


def make_pipeline(retrieval_results=None) -> RAGPipeline:
    """Build a RAGPipeline with a mocked retriever."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = retrieval_results or make_search_results()
    return RAGPipeline(mock_retriever)


# ── RAGPipeline Tests ─────────────────────────────────────────────────────────

class TestRAGPipeline:

    def test_returns_rag_response(self):
        pipeline = make_pipeline()
        with patch.object(pipeline, "_generate", return_value="RAG uses retrieval first."):
            response = pipeline.run("What is RAG?")

        assert isinstance(response, RAGResponse)
        assert response.question == "What is RAG?"
        assert "RAG" in response.answer

    def test_sources_extracted(self):
        pipeline = make_pipeline()
        with patch.object(pipeline, "_generate", return_value="answer"):
            response = pipeline.run("test question")

        assert len(response.sources) > 0
        assert all(isinstance(s, str) for s in response.sources)

    def test_empty_retrieval_returns_no_documents_message(self):
        pipeline = make_pipeline(retrieval_results=[])
        response = pipeline.run("What is the moon made of?")

        assert "No relevant documents" in response.answer
        assert response.chunks == []

    def test_context_includes_all_chunk_texts(self):
        chunks = make_search_results(3)
        pipeline = make_pipeline(retrieval_results=chunks)

        captured = {}
        def fake_generate(prompt):
            captured["prompt"] = prompt
            return "generated answer"

        with patch.object(pipeline, "_generate", side_effect=fake_generate):
            pipeline.run("question")

        for chunk in chunks:
            assert chunk.text in captured["prompt"]

    def test_openai_dispatch(self):
        pipeline = make_pipeline()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "OpenAI answer"

        with patch("src.generation.rag_pipeline.settings") as mock_settings:
            mock_settings.llm_provider = "openai"
            mock_settings.llm_model = "gpt-4o-mini"
            mock_settings.openai_api_key = "sk-fake"
            mock_settings.validate = lambda: None

            with patch("openai.OpenAI") as mock_openai_cls:
                mock_client = MagicMock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai_cls.return_value = mock_client

                answer = pipeline._generate_openai("test prompt")

        assert answer == "OpenAI answer"

    def test_anthropic_dispatch(self):
        pipeline = make_pipeline()
        mock_response = MagicMock()
        mock_response.content[0].text = "Anthropic answer"

        with patch("anthropic.Anthropic") as mock_anthropic_cls:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic_cls.return_value = mock_client

            with patch("src.generation.rag_pipeline.settings") as mock_settings:
                mock_settings.llm_provider = "anthropic"
                mock_settings.llm_model = "claude-sonnet-4-20250514"
                mock_settings.anthropic_api_key = "sk-ant-fake"

                answer = pipeline._generate_anthropic("test prompt")

        assert answer == "Anthropic answer"

    def test_unknown_provider_raises(self):
        pipeline = make_pipeline()
        with patch("src.generation.rag_pipeline.settings") as mock_settings:
            mock_settings.llm_provider = "cohere"
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                pipeline._generate("prompt")


# ── RAGResponse Tests ─────────────────────────────────────────────────────────

class TestRAGResponse:
    def test_repr_truncates_answer(self):
        response = RAGResponse(
            question="q",
            answer="a" * 200,
            sources=["doc.txt"],
            chunks=[],
        )
        r = repr(response)
        assert "..." in r
        assert len(r) < 500
