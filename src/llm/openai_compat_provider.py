"""
openai_compat_provider.py – A SINGLE provider class for ALL OpenAI-compatible APIs.

The openai SDK supports a custom base_url, so one class handles:
  - Groq       (https://api.groq.com/openai/v1)
  - Cerebras   (https://api.cerebras.ai/v1)
  - OpenRouter (https://openrouter.ai/api/v1)
  - Ollama     (http://localhost:11434/v1)    ← local, no internet

For Ollama, is_configured() also does a quick HTTP check to see if the
local server is actually running, so the fallback manager can skip it
gracefully when Ollama hasn't been started.
"""

import logging
from typing import Any

import httpx
from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError

from src.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# Well-known base URLs so the caller doesn't have to remember them
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL_DEFAULT = "http://localhost:11434/v1"


class OpenAICompatProvider(LLMProvider):
    """
    Calls any OpenAI-compatible API endpoint.

    Args:
        provider_name: Display name used in logs and the UI status panel.
        base_url:      The API base URL (e.g. GROQ_BASE_URL).
        api_key:       The provider's API key.  For Ollama, any non-empty
                       string works (Ollama ignores the value).
        model:         The model identifier string.
        is_local:      Set True for Ollama — enables the reachability check
                       instead of the API-key check.
    """

    def __init__(
        self,
        provider_name: str,
        base_url: str,
        api_key: str,
        model: str,
        is_local: bool = False,
    ) -> None:
        self._provider_name = provider_name
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._is_local = is_local

    # ── Identity ────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"{self._provider_name} ({self._model})"

    # ── Configuration check ────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """
        For cloud providers: check that a non-empty API key is set.
        For Ollama (is_local=True): check that the server is reachable
        via a quick HTTP GET to /api/tags.
        """
        if self._is_local:
            return self._is_ollama_reachable()
        return bool(self._api_key and self._api_key.strip())

    def _is_ollama_reachable(self) -> bool:
        """
        Try a 2-second GET to the Ollama tags endpoint.
        Returns False instead of raising if Ollama isn't running.
        """
        try:
            # Derive the base server URL from the OpenAI-compat URL
            # e.g. http://localhost:11434/v1 → http://localhost:11434
            server_url = self._base_url.rstrip("/v1").rstrip("/")
            httpx.get(f"{server_url}/api/tags", timeout=2.0)
            return True
        except Exception:
            return False

    # ── Generation ─────────────────────────────────────────────────────────────

    def generate(self, question: str, context_chunks: list[dict[str, Any]]) -> str:
        """
        Build a grounded prompt and send it to the OpenAI-compatible endpoint.

        Args:
            question:       The user's question.
            context_chunks: Top-k retrieved chunks from the vector store.

        Returns:
            The model's text response.

        Raises:
            openai.APIStatusError:     HTTP 4xx / 5xx — fallback manager
                                       inspects .status_code for 429.
            openai.APIConnectionError: Network failure (e.g. Ollama not running).
            openai.APITimeoutError:    Request timed out.
        """
        prompt = self.build_prompt(question, context_chunks)
        logger.info("Calling %s model: %s", self._provider_name, self._model)

        # Build a fresh client per call (stateless, safe for Streamlit reruns)
        client = OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )

        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
        )

        answer = response.choices[0].message.content or ""
        logger.info("%s responded (%d chars).", self._provider_name, len(answer))
        return answer.strip()
