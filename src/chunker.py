"""
chunker.py – Split page text into overlapping chunks with source metadata.

Strategy: simple character-level sliding window.
  - CHUNK_SIZE   (~800 chars) keeps chunks small enough to embed well.
  - CHUNK_OVERLAP (~150 chars) ensures a sentence split across a boundary
    still appears in full in at least one chunk.

Each chunk carries the same metadata as its source page so we can
always trace an answer back to the exact file and page number.
"""

import logging
from typing import Any

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

# Type alias: chunk dict produced by this module
ChunkData = dict[str, Any]


def chunk_page(
    page_text: str,
    file_name: str,
    page_num: int,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[ChunkData]:
    """
    Split a single page's text into overlapping character-level chunks.

    Args:
        page_text:    The raw text of one PDF page.
        file_name:    Original PDF filename (carried into every chunk).
        page_num:     1-based page number (carried into every chunk).
        chunk_size:   Target size of each chunk in characters.
        chunk_overlap: Number of characters to re-include from the end of
                       the previous chunk at the start of the next chunk.

    Returns:
        List of ChunkData dicts, each with keys:
          ``chunk_text``, ``file_name``, ``page_num``, ``chunk_index``.
        Returns an empty list if ``page_text`` is blank.
    """
    text = page_text.strip()
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})."
        )

    chunks: list[ChunkData] = []
    start = 0
    chunk_index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append(
            {
                "chunk_text": chunk_text,
                "file_name": file_name,
                "page_num": page_num,
                "chunk_index": chunk_index,  # position within this page
            }
        )

        chunk_index += 1
        # Advance start by (chunk_size - overlap) so the next chunk
        # shares `chunk_overlap` characters with this one.
        start += chunk_size - chunk_overlap

    return chunks


def chunk_pages(
    pages: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[ChunkData]:
    """
    Chunk all pages from one or more PDFs into a flat list of ChunkData dicts.

    Pages with empty text are silently skipped (they were already warned about
    in loader.py).

    Args:
        pages:        List of PageData dicts produced by loader.load_pdfs().
        chunk_size:   Forwarded to chunk_page().
        chunk_overlap: Forwarded to chunk_page().

    Returns:
        Flat list of ChunkData dicts ready for embedding.
    """
    all_chunks: list[ChunkData] = []

    for page in pages:
        page_chunks = chunk_page(
            page_text=page["text"],
            file_name=page["file_name"],
            page_num=page["page_num"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(page_chunks)

    logger.info(
        "Chunking complete: %d pages → %d chunks "
        "(size=%d, overlap=%d).",
        len(pages),
        len(all_chunks),
        chunk_size,
        chunk_overlap,
    )
    return all_chunks
