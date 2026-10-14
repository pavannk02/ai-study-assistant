"""
rag_pipeline.py – Orchestrates the full RAG flow: retrieve then generate.

This is the single function the Streamlit UI calls. It:
  1. Embeds the user's question.
  2. Searches the FAISS index for the top-k most relevant chunks.
  3. Calls FallbackManager.generate() with those chunks.
  4. Returns (answer, sources, provider_name) — all the UI needs.

Keeping this in its own file (not in app.py) means it can be unit-tested
and called from a CLI without any Streamlit imports.
"""

import logging
from typing import Any

from src.embedder import Embedder
from src.vector_store import VectorStore
from src.llm.fallback_manager import FallbackManager
from src.config import TOP_K

logger = logging.getLogger(__name__)

# Type alias for the answer tuple returned to the UI
RAGResult = tuple[str, list[dict[str, Any]], str]
# (answer_text, list_of_source_dicts, provider_name)


def answer_question(
    question: str,
    embedder: Embedder,
    store: VectorStore,
    manager: FallbackManager,
    top_k: int = TOP_K,
) -> RAGResult:
    """
    Run the full RAG pipeline for one question.

    Args:
        question: The user's question (must be non-empty — caller validates).
        embedder: Initialised Embedder instance (shared across requests).
        store:    Initialised VectorStore with at least one indexed PDF.
        manager:  Initialised FallbackManager.
        top_k:    Number of chunks to retrieve (from sidebar slider).

    Returns:
        Tuple of:
          - answer      (str): The LLM's response.
          - sources     (list[dict]): The retrieved chunks used as context,
                        each with chunk_text, file_name, page_num, score.
          - provider    (str): Which provider/model answered (or 'retrieval-only').

    Raises:
        RuntimeError: If the vector store is empty (no PDFs indexed yet).
    """
    if not store.is_ready:
        raise RuntimeError(
            "No documents indexed yet. Please upload at least one PDF first."
        )

    # ── Step 1: Embed the question ─────────────────────────────────────────────
    logger.info("Embedding question: '%s'", question[:80])
    query_vector = embedder.embed_one(question)

    # ── Step 2: Retrieve top-k chunks ──────────────────────────────────────────
    logger.info("Retrieving top-%d chunks from FAISS index.", top_k)
    sources = store.search(query_vector, top_k=top_k)

    if not sources:
        # Extremely unlikely (only if index is corrupted), but handle gracefully
        logger.warning("FAISS search returned no results.")
        return (
            "I couldn't find any relevant passages. Try uploading more documents.",
            [],
            "retrieval-only",
        )

    # ── Step 3: Generate answer via fallback manager ───────────────────────────
    logger.info("Sending %d chunks to LLM.", len(sources))
    answer, provider = manager.generate(question, sources)

    return answer, sources, provider
