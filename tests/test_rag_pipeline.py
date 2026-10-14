"""
test_rag_pipeline.py – Tests for src/rag_pipeline.py.

Verifies end-to-end orchestration between embedder, vector store,
and fallback LLM manager using lightweight mocks so no GPU,
API keys, or file I/O are required.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from src.rag_pipeline import answer_question


class TestAnswerQuestion:
    """Tests for the answer_question function."""

    def test_raises_when_store_not_ready(self):
        """Should raise RuntimeError if store.is_ready is False."""
        mock_store = MagicMock()
        mock_store.is_ready = False
        mock_embedder = MagicMock()
        mock_manager = MagicMock()

        with pytest.raises(RuntimeError, match="No documents indexed yet"):
            answer_question(
                question="What is backpropagation?",
                embedder=mock_embedder,
                store=mock_store,
                manager=mock_manager,
            )

    def test_happy_path(self):
        """Normal flow: embeds query, searches store, calls manager, returns result."""
        mock_store = MagicMock()
        mock_store.is_ready = True
        fake_chunks = [
            {"chunk_text": "Backpropagation computes gradients.", "file_name": "nn.pdf", "page_num": 1, "score": 0.89}
        ]
        mock_store.search.return_value = fake_chunks

        fake_query_vector = np.zeros(384, dtype=np.float32)
        mock_embedder = MagicMock()
        mock_embedder.embed_one.return_value = fake_query_vector

        mock_manager = MagicMock()
        mock_manager.generate.return_value = (
            "Backpropagation computes gradients using the chain rule.",
            "gemini-flash",
        )

        answer, sources, provider = answer_question(
            question="What is backpropagation?",
            embedder=mock_embedder,
            store=mock_store,
            manager=mock_manager,
            top_k=3,
        )

        mock_embedder.embed_one.assert_called_once_with("What is backpropagation?")
        mock_store.search.assert_called_once_with(fake_query_vector, top_k=3)
        mock_manager.generate.assert_called_once_with("What is backpropagation?", fake_chunks)

        assert answer == "Backpropagation computes gradients using the chain rule."
        assert sources == fake_chunks
        assert provider == "gemini-flash"

    def test_empty_search_results_handled_gracefully(self):
        """If search returns no chunks, returns graceful message without calling manager."""
        mock_store = MagicMock()
        mock_store.is_ready = True
        mock_store.search.return_value = []

        mock_embedder = MagicMock()
        mock_embedder.embed_one.return_value = np.zeros(384, dtype=np.float32)

        mock_manager = MagicMock()

        answer, sources, provider = answer_question(
            question="Unknown topic?",
            embedder=mock_embedder,
            store=mock_store,
            manager=mock_manager,
        )

        assert "couldn't find any relevant passages" in answer.lower()
        assert sources == []
        assert provider == "retrieval-only"
        mock_manager.generate.assert_not_called()
