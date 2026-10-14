"""
test_fallback_manager.py – Tests for src/llm/fallback_manager.py.

All LLM providers are MOCKED — no real API calls are ever made.

The four key behaviours tested:
  (a) A 429 error on the first provider causes fallthrough to the next.
  (b) Unconfigured providers are silently skipped.
  (c) A rate-limit cooldown is applied after a 429 error.
  (d) Retrieval-only mode is used when all providers fail.
"""

import time
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from openai import APIStatusError as OpenAIStatusError

from src.llm.fallback_manager import FallbackManager, RETRIEVAL_ONLY_ANSWER
from src.llm.base import LLMProvider


# ── Helpers ───────────────────────────────────────────────────────────────────

DUMMY_CHUNKS = [
    {
        "chunk_text": "The mitochondria is the powerhouse of the cell.",
        "file_name": "biology.pdf",
        "page_num": 1,
        "score": 0.95,
    }
]
DUMMY_QUESTION = "What is the powerhouse of the cell?"


def make_mock_provider(
    name: str,
    is_configured: bool = True,
    answer: str = "Mock answer.",
    raises: Exception | None = None,
) -> MagicMock:
    """
    Create a mock LLMProvider.

    Args:
        name:          The provider's display name.
        is_configured: Whether is_configured() returns True.
        answer:        What generate() returns (if it doesn't raise).
        raises:        If set, generate() raises this exception instead.
    """
    provider = MagicMock(spec=LLMProvider)
    type(provider).name = PropertyMock(return_value=name)
    provider.is_configured.return_value = is_configured

    if raises:
        provider.generate.side_effect = raises
    else:
        provider.generate.return_value = answer

    return provider


def make_openai_429() -> OpenAIStatusError:
    """Create a real OpenAI 429 APIStatusError (needs a mock response)."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    return OpenAIStatusError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}},
    )


def make_openai_500() -> OpenAIStatusError:
    """Create a real OpenAI 500 APIStatusError."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    return OpenAIStatusError(
        message="Internal server error",
        response=mock_response,
        body={"error": {"message": "Internal server error"}},
    )


def make_openai_401() -> OpenAIStatusError:
    """Create a real OpenAI 401 APIStatusError."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    return OpenAIStatusError(
        message="Unauthorized",
        response=mock_response,
        body={"error": {"message": "Unauthorized"}},
    )


# ── (a) 429 falls through to the next provider ───────────────────────────────

class TestFallthrough:
    def test_429_on_first_falls_to_second(self):
        """
        When the first provider raises a 429, the manager should try
        the second provider and return its answer.
        """
        p1 = make_mock_provider("Provider1", raises=make_openai_429())
        p2 = make_mock_provider("Provider2", answer="Answer from P2")

        manager = FallbackManager(provider_order=["p1", "p2"])

        with patch.object(manager, "_build_provider_list", return_value=[p1, p2]):
            answer, provider_name = manager.generate(DUMMY_QUESTION, DUMMY_CHUNKS)

        assert answer == "Answer from P2"
        assert "Provider2" in provider_name
        # P1 must have been called once; P2 must have been called once
        p1.generate.assert_called_once()
        p2.generate.assert_called_once()

    def test_500_on_first_falls_to_second(self):
        """Server errors should also cause fallthrough (no cooldown)."""
        p1 = make_mock_provider("Provider1", raises=make_openai_500())
        p2 = make_mock_provider("Provider2", answer="Answer from P2")

        manager = FallbackManager()
        with patch.object(manager, "_build_provider_list", return_value=[p1, p2]):
            answer, _ = manager.generate(DUMMY_QUESTION, DUMMY_CHUNKS)

        assert answer == "Answer from P2"

    def test_401_on_first_falls_to_second(self):
        """Auth errors should cause fallthrough (no cooldown)."""
        p1 = make_mock_provider("Provider1", raises=make_openai_401())
        p2 = make_mock_provider("Provider2", answer="Good answer")

        manager = FallbackManager()
        with patch.object(manager, "_build_provider_list", return_value=[p1, p2]):
            answer, _ = manager.generate(DUMMY_QUESTION, DUMMY_CHUNKS)

        assert answer == "Good answer"

    def test_first_provider_used_when_it_succeeds(self):
        """If P1 succeeds, P2 must never be called."""
        p1 = make_mock_provider("Provider1", answer="P1 answer")
        p2 = make_mock_provider("Provider2", answer="P2 answer")

        manager = FallbackManager()
        with patch.object(manager, "_build_provider_list", return_value=[p1, p2]):
            answer, _ = manager.generate(DUMMY_QUESTION, DUMMY_CHUNKS)

        assert answer == "P1 answer"
        p2.generate.assert_not_called()


# ── (b) Unconfigured providers are skipped ───────────────────────────────────

class TestUnconfigured:
    def test_unconfigured_first_skips_to_configured_second(self):
        """
        A provider that returns is_configured() == False must be skipped
        without calling generate() at all.
        """
        p1 = make_mock_provider("NoKey", is_configured=False)
        p2 = make_mock_provider("HasKey", answer="Configured answer")

        manager = FallbackManager()
        with patch.object(manager, "_build_provider_list", return_value=[p1, p2]):
            answer, name = manager.generate(DUMMY_QUESTION, DUMMY_CHUNKS)

        p1.generate.assert_not_called()   # ← must be skipped entirely
        assert answer == "Configured answer"

    def test_all_unconfigured_returns_retrieval_only(self):
        """All providers unconfigured → retrieval-only fallback."""
        providers = [
            make_mock_provider("P1", is_configured=False),
            make_mock_provider("P2", is_configured=False),
        ]
        manager = FallbackManager()
        with patch.object(manager, "_build_provider_list", return_value=providers):
            answer, name = manager.generate(DUMMY_QUESTION, DUMMY_CHUNKS)

        assert answer == RETRIEVAL_ONLY_ANSWER
        assert name == "retrieval-only"


# ── (c) Rate-limit cooldown is applied ───────────────────────────────────────

class TestCooldown:
    def test_cooldown_applied_after_429(self):
        """
        After a 429, the provider should be on cooldown and skipped
        on the very next call.
        """
        p1 = make_mock_provider("RateLimited", raises=make_openai_429())
        p2 = make_mock_provider("Backup", answer="Backup answer")

        manager = FallbackManager()

        # First call: P1 raises 429, P2 answers
        with patch.object(manager, "_build_provider_list", return_value=[p1, p2]):
            manager.generate(DUMMY_QUESTION, DUMMY_CHUNKS)

        # Verify cooldown was set for P1's name
        remaining = manager._cooldown_remaining("RateLimited")
        assert remaining > 0, "Cooldown should be active after a 429"

    def test_cooldown_expires(self):
        """After the cooldown period, the provider should be usable again."""
        manager = FallbackManager()
        # Set an artificial very-short cooldown (0.1 s)
        manager._set_cooldown("TestProvider", 0.1)
        assert manager._cooldown_remaining("TestProvider") > 0

        time.sleep(0.15)
        assert manager._cooldown_remaining("TestProvider") == 0.0

    def test_provider_skipped_while_on_cooldown(self):
        """
        A provider that is on cooldown must not have generate() called
        even if it is configured.
        """
        p1 = make_mock_provider("CooledDown", is_configured=True, answer="P1 answer")
        p2 = make_mock_provider("Fresh", answer="P2 answer")

        manager = FallbackManager()
        # Manually set P1 on cooldown
        manager._set_cooldown("CooledDown", 60)

        with patch.object(manager, "_build_provider_list", return_value=[p1, p2]):
            answer, name = manager.generate(DUMMY_QUESTION, DUMMY_CHUNKS)

        p1.generate.assert_not_called()
        assert answer == "P2 answer"

    def test_no_cooldown_for_server_error(self):
        """500 errors must NOT set a cooldown (only 429 should)."""
        p1 = make_mock_provider("Server", raises=make_openai_500())
        manager = FallbackManager()

        with patch.object(manager, "_build_provider_list", return_value=[p1]):
            manager.generate(DUMMY_QUESTION, DUMMY_CHUNKS)

        assert manager._cooldown_remaining("Server") == 0.0


# ── (d) Retrieval-only mode when all providers fail ──────────────────────────

class TestRetrievalOnly:
    def test_all_fail_returns_retrieval_only_answer(self):
        """
        When every provider either raises or is unconfigured,
        the manager must return RETRIEVAL_ONLY_ANSWER.
        """
        providers = [
            make_mock_provider("P1", raises=make_openai_500()),
            make_mock_provider("P2", raises=make_openai_500()),
            make_mock_provider("P3", raises=Exception("Unknown error")),
        ]
        manager = FallbackManager()
        with patch.object(manager, "_build_provider_list", return_value=providers):
            answer, name = manager.generate(DUMMY_QUESTION, DUMMY_CHUNKS)

        assert answer == RETRIEVAL_ONLY_ANSWER
        assert name == "retrieval-only"

    def test_empty_provider_list_returns_retrieval_only(self):
        """An empty provider list must return retrieval-only gracefully."""
        manager = FallbackManager(provider_order=[])
        with patch.object(manager, "_build_provider_list", return_value=[]):
            answer, name = manager.generate(DUMMY_QUESTION, DUMMY_CHUNKS)

        assert answer == RETRIEVAL_ONLY_ANSWER
        assert name == "retrieval-only"


# ── Offline mode ──────────────────────────────────────────────────────────────

class TestOfflineMode:
    def test_offline_mode_only_uses_ollama(self):
        """
        In offline mode, _build_provider_list() should return only
        the Ollama provider regardless of provider_order.
        """
        manager = FallbackManager(
            provider_order=["gemini", "groq", "ollama"],
            offline_mode=True,
        )
        providers = manager._build_provider_list()
        # Only one provider, and it must be Ollama
        assert len(providers) == 1
        assert "Ollama" in providers[0].name


# ── Provider statuses ─────────────────────────────────────────────────────────

class TestProviderStatuses:
    def test_statuses_returned_for_all_providers(self):
        """provider_statuses() must return a list with a dict per provider."""
        p1 = make_mock_provider("Ready", is_configured=True)
        p2 = make_mock_provider("NoKey", is_configured=False)

        manager = FallbackManager()
        with patch.object(manager, "_build_provider_list", return_value=[p1, p2]):
            statuses = manager.provider_statuses()

        assert len(statuses) == 2
        status_map = {s["name"]: s["status"] for s in statuses}
        assert status_map["Ready"] == "ready"
        assert status_map["NoKey"] == "no_key"

    def test_rate_limited_status_shown(self):
        """A provider on cooldown must show status 'rate_limited'."""
        p1 = make_mock_provider("P1", is_configured=True)

        manager = FallbackManager()
        manager._set_cooldown("P1", 60)

        with patch.object(manager, "_build_provider_list", return_value=[p1]):
            statuses = manager.provider_statuses()

        assert statuses[0]["status"] == "rate_limited"
        assert statuses[0]["cooldown_remaining"] > 0
