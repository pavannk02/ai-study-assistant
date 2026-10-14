"""
fallback_manager.py – Tries LLM providers in order and handles failures gracefully.

Key behaviours
──────────────
1. Skips providers that are not configured (no API key / Ollama not running).
2. Catches rate-limit (429), quota, auth, timeout, and server errors.
3. Puts a rate-limited provider on cooldown (default 60 s) before trying again.
4. If EVERY provider fails, returns the raw retrieved passages so the user
   always gets something useful ("retrieval-only mode").
5. The provider list and cooldown duration are configurable via .env.

Error classification
────────────────────
- openai.APIStatusError:         check .status_code
  - 429 → rate-limit cooldown
  - 401/403 → auth error, skip (no cooldown)
  - 500/502/503 → server error, skip
- openai.APIConnectionError:     network failure, skip
- openai.APITimeoutError:        timeout, skip
- google.genai.errors.APIError:  check .status_code the same way
- Any other Exception:           log and skip
"""

import logging
import time
from typing import Any

from openai import APIStatusError as OpenAIStatusError
from openai import APIConnectionError as OpenAIConnectionError
from openai import APITimeoutError as OpenAITimeoutError

try:
    import google.genai.errors as genai_errors
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

from src.config import (
    GEMINI_API_KEY, GEMINI_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
    CEREBRAS_API_KEY, CEREBRAS_MODEL,
    OPENROUTER_API_KEY, OPENROUTER_MODEL,
    OLLAMA_BASE_URL, OLLAMA_API_KEY, OLLAMA_MODEL,
    LLM_PROVIDER_ORDER,
    RATE_LIMIT_COOLDOWN_SECONDS,
)
from src.llm.base import LLMProvider
from src.llm.gemini_provider import GeminiProvider
from src.llm.openai_compat_provider import (
    OpenAICompatProvider,
    GROQ_BASE_URL,
    CEREBRAS_BASE_URL,
    OPENROUTER_BASE_URL,
)

logger = logging.getLogger(__name__)

# Sentinel returned when every provider fails
RETRIEVAL_ONLY_ANSWER = (
    "⚠️ **No LLM provider was available** — showing retrieved passages only.\n\n"
    "You can still read the relevant excerpts from your documents in the Sources section below."
)


class FallbackManager:
    """
    Manages a prioritised list of LLM providers and tries them in order.

    Supports:
      - Runtime key injection (sidebar bring-your-own-key feature).
      - Offline mode (only Ollama is tried).
      - Per-provider cooldown tracking (rate-limit backoff).
      - Retrieval-only fallback when all providers fail.
    """

    def __init__(
        self,
        provider_order: list[str] | None = None,
        runtime_keys: dict[str, str] | None = None,
        offline_mode: bool = False,
        ollama_model: str | None = None,
    ) -> None:
        """
        Args:
            provider_order: Ordered list of provider names to try.
                            Defaults to LLM_PROVIDER_ORDER from config.
            runtime_keys:   Dict mapping provider name → API key entered in the
                            sidebar.  These override .env values for this session.
            offline_mode:   When True, only 'ollama' is tried.
            ollama_model:   Model name to use with Ollama (defaults to OLLAMA_MODEL).
        """
        self._order = provider_order or LLM_PROVIDER_ORDER
        self._runtime_keys = runtime_keys or {}
        self._offline_mode = offline_mode
        self._ollama_model = ollama_model

        # Maps provider name → unix timestamp when cooldown expires (0 = no cooldown)
        self._cooldown_until: dict[str, float] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self, question: str, context_chunks: list[dict[str, Any]]
    ) -> tuple[str, str]:
        """
        Try providers in order and return the first successful answer.

        Args:
            question:       The user's question.
            context_chunks: Top-k retrieved chunks from the vector store.

        Returns:
            Tuple of (answer: str, provider_name: str).
            If all fail, returns (RETRIEVAL_ONLY_ANSWER, "retrieval-only").
        """
        providers = self._build_provider_list()

        for provider in providers:
            pname = provider.name

            # 1. Skip if not configured
            if not provider.is_configured():
                logger.info("Skipping %s (not configured).", pname)
                continue

            # 2. Skip if on cooldown
            remaining = self._cooldown_remaining(pname)
            if remaining > 0:
                logger.info(
                    "Skipping %s (rate-limited, %ds remaining).", pname, int(remaining)
                )
                continue

            # 3. Try to generate
            try:
                answer = provider.generate(question, context_chunks)
                logger.info("Answer from %s.", pname)
                return answer, pname

            except Exception as exc:
                self._handle_error(pname, exc)

        # All providers failed — retrieval-only fallback
        logger.warning("All providers failed. Falling back to retrieval-only mode.")
        return RETRIEVAL_ONLY_ANSWER, "retrieval-only"

    def provider_statuses(self) -> list[dict[str, Any]]:
        """
        Return a status dict for each configured provider (for the UI panel).

        Each dict has: name, status ('ready'|'no_key'|'rate_limited'|'unreachable'),
        and optionally cooldown_remaining (seconds).
        """
        statuses = []
        for provider in self._build_provider_list():
            pname = provider.name
            remaining = self._cooldown_remaining(pname)

            if remaining > 0:
                status = "rate_limited"
            elif not provider.is_configured():
                # Distinguish between no-key and unreachable (Ollama)
                status = "unreachable" if "Ollama" in pname else "no_key"
            else:
                status = "ready"

            statuses.append(
                {
                    "name": pname,
                    "status": status,
                    "cooldown_remaining": max(0.0, remaining),
                }
            )
        return statuses

    # ── Provider list builder ──────────────────────────────────────────────────

    def _build_provider_list(self) -> list[LLMProvider]:
        """
        Construct provider objects from the ordered name list.

        Injects runtime keys (from the sidebar) over .env defaults.
        In offline mode, only Ollama is returned.
        """
        # Helper to pick runtime key or fall back to .env value
        def key(provider_name: str, env_key: str) -> str:
            return self._runtime_keys.get(provider_name, env_key)

        name_to_provider: dict[str, LLMProvider] = {
            "gemini": GeminiProvider(
                api_key=key("gemini", GEMINI_API_KEY),
                model=GEMINI_MODEL,
            ),
            "groq": OpenAICompatProvider(
                provider_name="Groq",
                base_url=GROQ_BASE_URL,
                api_key=key("groq", GROQ_API_KEY),
                model=GROQ_MODEL,
            ),
            "cerebras": OpenAICompatProvider(
                provider_name="Cerebras",
                base_url=CEREBRAS_BASE_URL,
                api_key=key("cerebras", CEREBRAS_API_KEY),
                model=CEREBRAS_MODEL,
            ),
            "openrouter": OpenAICompatProvider(
                provider_name="OpenRouter",
                base_url=OPENROUTER_BASE_URL,
                api_key=key("openrouter", OPENROUTER_API_KEY),
                model=OPENROUTER_MODEL,
            ),
            "ollama": OpenAICompatProvider(
                provider_name="Ollama",
                base_url=OLLAMA_BASE_URL,
                api_key=OLLAMA_API_KEY,
                model=self._ollama_model or OLLAMA_MODEL,
                is_local=True,
            ),
        }

        if self._offline_mode:
            return [name_to_provider["ollama"]]

        return [
            name_to_provider[name]
            for name in self._order
            if name in name_to_provider
        ]

    # ── Cooldown helpers ───────────────────────────────────────────────────────

    def _set_cooldown(self, provider_name: str, seconds: float) -> None:
        """Put a provider on cooldown for `seconds` seconds."""
        self._cooldown_until[provider_name] = time.monotonic() + seconds
        logger.warning(
            "%s put on cooldown for %ds.", provider_name, int(seconds)
        )

    def _cooldown_remaining(self, provider_name: str) -> float:
        """Return how many seconds of cooldown remain (0 if none)."""
        until = self._cooldown_until.get(provider_name, 0.0)
        return max(0.0, until - time.monotonic())

    # ── Error handler ──────────────────────────────────────────────────────────

    def _handle_error(self, provider_name: str, exc: Exception) -> None:
        """
        Classify the exception and apply a cooldown if appropriate.

        Rate-limit (429) → cooldown for RATE_LIMIT_COOLDOWN_SECONDS.
        Auth / server / connection / timeout → log and skip (no cooldown).
        """
        status_code = self._extract_status_code(exc)

        if status_code == 429:
            logger.warning(
                "%s: rate-limited (429). Cooldown %ds.",
                provider_name, RATE_LIMIT_COOLDOWN_SECONDS,
            )
            self._set_cooldown(provider_name, RATE_LIMIT_COOLDOWN_SECONDS)

        elif status_code in (401, 403):
            logger.warning("%s: auth error (%s). Skipping.", provider_name, status_code)

        elif status_code and status_code >= 500:
            logger.warning("%s: server error (%s). Skipping.", provider_name, status_code)

        elif isinstance(exc, (OpenAIConnectionError, OpenAITimeoutError)):
            logger.warning("%s: connection/timeout error. Skipping.", provider_name)

        else:
            logger.warning(
                "%s: unexpected error (%s: %s). Skipping.",
                provider_name, type(exc).__name__, exc,
            )

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        """
        Pull the HTTP status code out of an exception, regardless of which
        SDK raised it (openai or google-genai).
        """
        # openai SDK
        if isinstance(exc, OpenAIStatusError):
            return exc.status_code

        # google-genai SDK
        if _GENAI_AVAILABLE:
            try:
                # google.genai.errors.APIError stores the code differently
                # depending on SDK version; try both attribute names
                if hasattr(exc, "status_code"):
                    return int(exc.status_code)
                if hasattr(exc, "code"):
                    return int(exc.code)
            except (TypeError, ValueError):
                pass

        return None
