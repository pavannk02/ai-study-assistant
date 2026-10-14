"""
gemini_provider.py – LLM provider for Google Gemini via the google-genai SDK.

The google-genai SDK raises google.genai.errors.APIError for all API-level
failures.  We check the HTTP status code on the exception to decide whether
it is a rate-limit error (429) that the fallback manager should cooldown vs.
any other error that it should just skip and move on from.
"""

import logging
from typing import Any

import google.genai as genai
import google.genai.errors as genai_errors

from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """
    Calls the Gemini API using the official google-genai SDK.

    Reads GEMINI_API_KEY and GEMINI_MODEL from config (which reads .env).
    """

    def __init__(
        self,
        api_key: str = GEMINI_API_KEY,
        model: str = GEMINI_MODEL,
    ) -> None:
        self._api_key = api_key
        self._model = model

    # ── Identity ────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"Gemini ({self._model})"

    # ── Configuration check ────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """Return True only if a non-empty API key has been provided."""
        return bool(self._api_key and self._api_key.strip())

    # ── Generation ─────────────────────────────────────────────────────────────

    def generate(self, question: str, context_chunks: list[dict[str, Any]]) -> str:
        """
        Build a grounded prompt and send it to Gemini.

        Args:
            question:       The user's question.
            context_chunks: Top-k retrieved chunks from the vector store.

        Returns:
            The model's text response.

        Raises:
            google.genai.errors.APIError: Propagated so fallback_manager
                can inspect the status code and decide what to do.
        """
        prompt = self.build_prompt(question, context_chunks)

        logger.info("Calling Gemini model: %s", self._model)

        # Create a fresh client per call — lightweight, stateless
        client = genai.Client(api_key=self._api_key)

        try:
            response = client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
            answer = response.text or ""
            logger.info("Gemini responded (%d chars).", len(answer))
            return answer.strip()

        except genai_errors.APIError as exc:
            # Re-raise so the fallback manager can inspect exc.status_code
            logger.warning("Gemini API error: %s", exc)
            raise
