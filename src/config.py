"""
config.py – Central configuration for the AI Study Assistant.

All constants and environment variables are read here and imported
by every other module.  Nothing else should ever call os.getenv() directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env (safe to call even if the file does not exist) ──────────────────
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

# Root of the project (the folder that contains src/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Where FAISS index and chunk metadata are stored (gitignored)
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH: Path = DATA_DIR / "index.faiss"
CHUNKS_METADATA_PATH: Path = DATA_DIR / "chunks.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# Embedding model (always local)
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

TOP_K: int = int(os.getenv("TOP_K", "4"))

# ─────────────────────────────────────────────────────────────────────────────
# LLM Provider Models (read from .env, never hardcoded)
# ─────────────────────────────────────────────────────────────────────────────

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

CEREBRAS_API_KEY: str = os.getenv("CEREBRAS_API_KEY", "")
CEREBRAS_MODEL: str = os.getenv("CEREBRAS_MODEL", "llama3.1-8b")

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.2-3b-instruct:free")

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "ollama")  # Ollama ignores this value
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# ─────────────────────────────────────────────────────────────────────────────
# Provider order (comma-separated string → list)
# ─────────────────────────────────────────────────────────────────────────────

_raw_order: str = os.getenv(
    "LLM_PROVIDER_ORDER", "gemini,groq,cerebras,openrouter,ollama"
)
LLM_PROVIDER_ORDER: list[str] = [p.strip() for p in _raw_order.split(",") if p.strip()]

# ─────────────────────────────────────────────────────────────────────────────
# Fallback manager settings
# ─────────────────────────────────────────────────────────────────────────────

# Seconds a rate-limited provider is put on cooldown before being tried again
RATE_LIMIT_COOLDOWN_SECONDS: int = 60
