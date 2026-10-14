"""
app.py – Streamlit UI for the AI Study Assistant.

This file is ONLY responsible for the user interface.  All business logic
lives in src/ and is imported here.

Layout:
  Sidebar  → Upload PDFs, settings (top-k, offline mode, BYOK, provider status)
  Main     → Chat history, question input, answer + sources expander
"""

import logging
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import (
    TOP_K,
    RATE_LIMIT_COOLDOWN_SECONDS,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)
from src.loader import load_pdfs, has_extractable_text
from src.chunker import chunk_pages
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.rag_pipeline import answer_question
from src.llm.fallback_manager import FallbackManager

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ─── Page config (must be the very first Streamlit call) ──────────────────────
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_session_state() -> None:
    """Initialise all session-state keys exactly once per browser session."""
    defaults: dict[str, Any] = {
        "chat_history": [],          # list of {role, content, sources, provider}
        "embedder": None,            # Embedder singleton
        "vector_store": None,        # VectorStore singleton
        "indexed_files": [],         # file names already in the index
        "runtime_keys": {},          # BYOK keys {provider_name: key}
        "offline_mode": False,
        "ollama_model": OLLAMA_MODEL,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ─────────────────────────────────────────────────────────────────────────────
# Cached singletons
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_embedder() -> Embedder:
    """Load the embedding model once per server process (cached across reruns)."""
    return Embedder()


@st.cache_resource
def get_vector_store() -> VectorStore:
    """
    Get (or create) the VectorStore.  On first call, tries to load a saved
    index from disk so the user doesn't have to re-upload after a restart.
    """
    store = VectorStore()
    store.load()   # no-op and returns False if no saved index exists
    return store


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _get_installed_ollama_models() -> list[str]:
    """Query the local Ollama daemon for currently installed models."""
    try:
        import json
        import urllib.request
        base = OLLAMA_BASE_URL.removesuffix("/v1").rstrip("/")
        req = urllib.request.Request(f"{base}/api/tags", headers={"User-Agent": "AIStudyAssistant"})
        with urllib.request.urlopen(req, timeout=1.2) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", []) if "name" in m]
    except Exception:
        return []


def render_sidebar() -> dict[str, Any]:
    """
    Render the full sidebar and return the current settings as a dict.

    Returns:
        dict with keys: top_k (int), offline_mode (bool).
    """
    st.sidebar.title("📚 AI Study Assistant")
    st.sidebar.caption("RAG-powered Q&A grounded in your documents.")

    # ── PDF Upload ─────────────────────────────────────────────────────────────
    st.sidebar.header("1. Upload PDFs")
    uploaded_files = st.sidebar.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        label_visibility="collapsed",
    )
    if uploaded_files:
        _process_uploads(uploaded_files)

    # Show indexed files
    store = get_vector_store()
    if store.is_ready:
        st.sidebar.success(
            f"✅ **{store.total_chunks}** chunks indexed from "
            f"**{len(store.get_indexed_files())}** file(s)."
        )
        with st.sidebar.expander("Indexed files"):
            for f in store.get_indexed_files():
                st.write(f"• {f}")
    else:
        st.sidebar.info("No documents indexed yet. Upload a PDF to get started.")

    st.sidebar.divider()

    # ── Retrieval settings ─────────────────────────────────────────────────────
    st.sidebar.header("2. Retrieval Settings")
    top_k = st.sidebar.slider(
        "Top-k chunks to retrieve",
        min_value=1,
        max_value=8,
        value=TOP_K,
        help="How many document passages are sent to the LLM as context.",
    )

    st.sidebar.divider()

    # ── Offline mode ───────────────────────────────────────────────────────────
    st.sidebar.header("3. LLM Settings")
    offline_mode = st.sidebar.toggle(
        "🔌 Offline mode (Ollama only)",
        value=st.session_state.offline_mode,
        help=(
            "When ON, only your local Ollama model is used. "
            "Nothing is sent over the internet. "
            "Requires Ollama to be running: https://ollama.com/download"
        ),
    )
    st.session_state.offline_mode = offline_mode

    # Ollama model selection (auto-detects local models)
    installed = _get_installed_ollama_models()
    curr_model = st.session_state.ollama_model
    if installed:
        default_idx = installed.index(curr_model) if curr_model in installed else 0
        chosen_model = st.sidebar.selectbox(
            "🦙 Ollama Model",
            options=installed,
            index=default_idx,
            help="Local models detected from your running Ollama instance.",
        )
        st.session_state.ollama_model = chosen_model
    else:
        chosen_input = st.sidebar.text_input(
            "🦙 Ollama Model",
            value=curr_model,
            help="Local model name (e.g. qwen2.5-coder:7b)",
        )
        if chosen_input.strip():
            st.session_state.ollama_model = chosen_input.strip()

    if offline_mode:
        st.sidebar.info(
            f"🔒 **Offline mode**: using local model `{st.session_state.ollama_model}`."
        )

    # ── Bring-your-own-key ─────────────────────────────────────────────────────
    if not offline_mode:
        with st.sidebar.expander("🔑 Bring your own API keys (optional)"):
            st.caption(
                "Keys entered here are **only kept in memory for this session** — "
                "never written to disk or logged."
            )
            for provider_id, label in [
                ("gemini", "Gemini API key"),
                ("groq", "Groq API key"),
                ("cerebras", "Cerebras API key"),
                ("openrouter", "OpenRouter API key"),
            ]:
                entered = st.text_input(
                    label,
                    type="password",
                    key=f"byok_{provider_id}",
                    placeholder="Leave blank to use .env default",
                )
                if entered:
                    st.session_state.runtime_keys[provider_id] = entered

    # ── Provider status ────────────────────────────────────────────────────────
    with st.sidebar.expander("📡 Provider status"):
        manager = _build_manager()
        statuses = manager.provider_statuses()
        for s in statuses:
            icon, badge = _status_badge(s)
            st.markdown(f"{icon} **{s['name']}** — {badge}")
            if s["status"] == "rate_limited":
                st.caption(f"  ↳ Cooldown: {s['cooldown_remaining']:.0f}s remaining")

    st.sidebar.divider()

    # ── Danger zone ────────────────────────────────────────────────────────────
    st.sidebar.header("4. Reset")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset index", use_container_width=True, type="secondary"):
            store = get_vector_store()
            store.reset()
            st.session_state.indexed_files = []
            # Clear the cached resource so get_vector_store() re-creates it
            get_vector_store.clear()
            st.success("Index reset.")
            st.rerun()

    return {"top_k": top_k, "offline_mode": offline_mode}


# ─────────────────────────────────────────────────────────────────────────────
# PDF processing
# ─────────────────────────────────────────────────────────────────────────────

def _process_uploads(uploaded_files) -> None:
    """
    For each uploaded file, check for duplicates, extract text, chunk,
    embed and add to the FAISS index.  Shows a spinner for large files.
    """
    store = get_vector_store()
    embedder = get_embedder()
    already_indexed = store.get_indexed_files()

    new_files = [f for f in uploaded_files if f.name not in already_indexed]

    if not new_files:
        return   # all files already in the index — nothing to do

    with st.spinner(f"Processing {len(new_files)} new file(s)…"):
        for uploaded in new_files:
            st.sidebar.info(f"Processing: **{uploaded.name}**…")

            # Save to a temp file because pypdf needs a file path
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)

            try:
                # Load & extract text
                pages = load_pdfs([tmp_path])

                # Rename pages to use the original file name (not the tmp path)
                for p in pages:
                    p["file_name"] = uploaded.name

                # Warn if no text was extracted (scanned PDF)
                if not has_extractable_text(pages):
                    st.sidebar.warning(
                        f"⚠️ **{uploaded.name}** has no extractable text. "
                        "It may be a scanned (image-only) PDF. "
                        "OCR is not supported — try a PDF with a text layer."
                    )
                    continue

                # Chunk → embed → add to index
                chunks = chunk_pages(pages)
                texts = [c["chunk_text"] for c in chunks]
                vectors = embedder.embed(texts)
                store.add(vectors, chunks)
                store.save()

                st.sidebar.success(f"✅ Indexed **{uploaded.name}** ({len(chunks)} chunks).")

            except Exception as exc:
                st.sidebar.error(f"❌ Error processing **{uploaded.name}**: {exc}")
            finally:
                tmp_path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chat UI
# ─────────────────────────────────────────────────────────────────────────────

def render_chat(settings: dict[str, Any]) -> None:
    """Render the main chat interface."""
    st.title("💬 AI Study Assistant")
    st.caption("Ask questions about your uploaded documents — answers are grounded in your PDFs.")

    # ── Chat history ───────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show sources expander for assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"], msg.get("provider", ""))

    # ── Question input ─────────────────────────────────────────────────────────
    question = st.chat_input(
        "Ask a question about your documents…",
        disabled=not get_vector_store().is_ready,
    )

    if not get_vector_store().is_ready:
        st.info("👈 Upload one or more PDFs from the sidebar to get started.")
        return

    if not question:
        return

    if not question.strip():
        st.warning("Please enter a question.")
        return

    # ── Show the user's message immediately ───────────────────────────────────
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    # ── Generate answer ────────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        with st.spinner("Searching your documents…"):
            try:
                manager = _build_manager()
                answer, sources, provider = answer_question(
                    question=question,
                    embedder=get_embedder(),
                    store=get_vector_store(),
                    manager=manager,
                    top_k=settings["top_k"],
                )
            except RuntimeError as exc:
                st.error(str(exc))
                return

        st.markdown(answer)
        st.caption(f"_Answered by: **{provider}**_")
        _render_sources(sources, provider)

    # Save to history
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "provider": provider,
        }
    )


def _render_sources(sources: list[dict[str, Any]], provider: str) -> None:
    """Render an expandable Sources section below an answer."""
    if not sources:
        return

    with st.expander(f"📎 Sources ({len(sources)} passage{'s' if len(sources) > 1 else ''} used)"):
        for i, src in enumerate(sources, start=1):
            score_pct = f"{src['score'] * 100:.1f}%"
            st.markdown(
                f"**[{i}] {src['file_name']}** — Page {src['page_num']} "
                f"| Similarity: `{score_pct}`"
            )
            snippet = src["chunk_text"][:400].replace("\n", " ")
            st.markdown(f"> {snippet}…")
            if i < len(sources):
                st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_manager() -> FallbackManager:
    """Build a FallbackManager with the current session settings."""
    return FallbackManager(
        runtime_keys=st.session_state.runtime_keys,
        offline_mode=st.session_state.offline_mode,
        ollama_model=st.session_state.ollama_model,
    )


def _status_badge(status_dict: dict[str, Any]) -> tuple[str, str]:
    """Return (icon, text) for a provider status dict."""
    status = status_dict["status"]
    match status:
        case "ready":
            return "🟢", "ready"
        case "no_key":
            return "⚪", "no key configured"
        case "rate_limited":
            return "🟡", "rate-limited (cooldown)"
        case "unreachable":
            return "🔴", "unreachable (not running)"
        case _:
            return "⚫", status


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    init_session_state()
    settings = render_sidebar()
    render_chat(settings)


if __name__ == "__main__":
    main()
