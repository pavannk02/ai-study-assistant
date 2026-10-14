"""
base.py – Abstract base class for all LLM providers.

Every provider must implement:
  - name         : human-readable identifier used in logs and UI
  - is_configured(): returns False if credentials are missing (so the
                     fallback manager can skip silently)
  - generate()   : call the model and return a plain string answer

The shared build_prompt() method lives here so the prompt sent to EVERY
provider is identical — only the API call differs between providers.
"""

from abc import ABC, abstractmethod
from typing import Any


class LLMProvider(ABC):
    """Abstract base class that all LLM providers must inherit from."""

    # ── Identity ────────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name (e.g. 'Gemini', 'Groq')."""

    # ── Configuration check ────────────────────────────────────────────────────

    @abstractmethod
    def is_configured(self) -> bool:
        """
        Return True if the provider has the credentials it needs to run.

        For API-based providers: checks that the API key is non-empty.
        For Ollama: checks that the local server is reachable.
        """

    # ── Generation ─────────────────────────────────────────────────────────────

    @abstractmethod
    def generate(self, question: str, context_chunks: list[dict[str, Any]]) -> str:
        """
        Call the LLM and return the answer as a plain string.

        Args:
            question:       The user's question.
            context_chunks: The top-k retrieved chunks from the vector store.
                            Each dict has at least: chunk_text, file_name, page_num.

        Returns:
            The model's answer as a string.

        Raises:
            Any exception from the underlying SDK — the fallback manager
            is responsible for catching and classifying these.
        """

    # ── Shared prompt builder ──────────────────────────────────────────────────

    @staticmethod
    def build_prompt(question: str, context_chunks: list[dict[str, Any]]) -> str:
        """
        Build the grounded-answer prompt that is sent to EVERY provider.

        Using the same prompt everywhere means:
          - Answers are consistent regardless of which provider responds.
          - Prompt logic only needs to be changed in one place.

        The prompt instructs the model to:
          1. Answer ONLY from the provided context.
          2. Say "I couldn't find this in your documents." if the answer
             is not present — preventing hallucination.
          3. Cite the source file and page number in the answer.

        Args:
            question:       The user's question.
            context_chunks: List of chunk dicts with chunk_text, file_name, page_num.

        Returns:
            A formatted prompt string ready to send to any LLM.
        """
        # Format each chunk with its source reference
        context_parts: list[str] = []
        for i, chunk in enumerate(context_chunks, start=1):
            source = f"[Source {i}: {chunk['file_name']}, page {chunk['page_num']}]"
            context_parts.append(f"{source}\n{chunk['chunk_text']}")

        context_text = "\n\n".join(context_parts)

        prompt = (
            "You are a helpful study assistant. Answer the student's question "
            "using ONLY the context passages provided below. "
            "If the answer is not contained in the context, respond with exactly: "
            "\"I couldn't find this in your documents.\"\n"
            "Always cite the source file and page number when you use information "
            "from a passage.\n\n"
            "=== CONTEXT ===\n"
            f"{context_text}\n\n"
            "=== QUESTION ===\n"
            f"{question}\n\n"
            "=== ANSWER ==="
        )
        return prompt
