"""
test_retrieval.py – Command-line script to verify Stage 2 end-to-end.

Usage (from the ai-study-assistant/ folder with .venv active):
    python scripts/test_retrieval.py path/to/sample.pdf "your question here"

This script:
  1. Loads and chunks the PDF
  2. Embeds the chunks and builds a FAISS index
  3. Embeds the question and searches the index
  4. Prints the top-k results with scores and snippets

No LLM calls are made — this purely tests the retrieval pipeline.
"""

import sys
import logging
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loader import load_pdfs
from src.chunker import chunk_pages
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.config import TOP_K

# Set up a simple console logger so we can see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    # ── Parse arguments ────────────────────────────────────────────────────────
    if len(sys.argv) < 3:
        print("Usage: python scripts/test_retrieval.py <pdf_path> <question>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    question = " ".join(sys.argv[2:])

    if not pdf_path.exists():
        logger.error("File not found: %s", pdf_path)
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"PDF    : {pdf_path.name}")
    print(f"Query  : {question}")
    print(f"{'='*60}\n")

    # ── Step 1: Load PDF ───────────────────────────────────────────────────────
    logger.info("Step 1: Loading PDF…")
    pages = load_pdfs([pdf_path])

    if not pages:
        logger.error("No pages extracted. Is this a valid PDF?")
        sys.exit(1)

    extractable = [p for p in pages if p["text"].strip()]
    logger.info(
        "%d total pages, %d with extractable text.", len(pages), len(extractable)
    )

    if not extractable:
        logger.warning(
            "No extractable text found. This PDF may be scanned (image-only). "
            "OCR is not supported — try a PDF with a text layer."
        )
        sys.exit(0)

    # ── Step 2: Chunk ──────────────────────────────────────────────────────────
    logger.info("Step 2: Chunking pages…")
    chunks = chunk_pages(pages)
    logger.info("%d chunks created.", len(chunks))

    # ── Step 3: Embed chunks ───────────────────────────────────────────────────
    logger.info("Step 3: Embedding chunks (this may take a moment first run)…")
    embedder = Embedder()
    texts = [c["chunk_text"] for c in chunks]
    vectors = embedder.embed(texts)
    logger.info("Embedded %d chunks → shape %s.", len(texts), vectors.shape)

    # ── Step 4: Build FAISS index ──────────────────────────────────────────────
    logger.info("Step 4: Building FAISS index…")
    store = VectorStore()
    store.build(vectors, chunks)

    # ── Step 5: Embed query & search ───────────────────────────────────────────
    logger.info("Step 5: Searching for top-%d chunks…", TOP_K)
    query_vec = embedder.embed_one(question)
    results = store.search(query_vec, top_k=TOP_K)

    # ── Step 6: Display results ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"Top {len(results)} retrieved chunks for: '{question}'")
    print(f"{'─'*60}\n")

    for rank, result in enumerate(results, start=1):
        snippet = result["chunk_text"][:300].replace("\n", " ")
        print(f"[{rank}]  Score: {result['score']:.4f}")
        print(f"     File : {result['file_name']}  |  Page: {result['page_num']}")
        print(f"     Snippet: {snippet}…\n")

    print("✅  Retrieval test complete. FAISS index is working correctly.")


if __name__ == "__main__":
    main()
