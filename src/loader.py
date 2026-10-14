"""
loader.py – Extract text from PDF files, one page at a time.

Each page becomes a dict with three keys:
  - text      : the extracted string (may be empty for scanned pages)
  - file_name : the original filename (not the full path, for privacy)
  - page_num  : 1-based page number (matches how humans count PDF pages)
"""

import logging
from pathlib import Path
from typing import Generator

from pypdf import PdfReader

logger = logging.getLogger(__name__)


# Type alias for a single page's data
PageData = dict[str, str | int]


def load_pdf(file_path: str | Path) -> Generator[PageData, None, None]:
    """
    Yield one PageData dict per page in the PDF.

    Args:
        file_path: Absolute or relative path to the PDF file.

    Yields:
        dict with keys: ``text`` (str), ``file_name`` (str), ``page_num`` (int).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PDF.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

    file_name = path.name
    logger.info("Loading PDF: %s", file_name)

    reader = PdfReader(str(path))
    total_pages = len(reader.pages)
    logger.info("  → %d pages found", total_pages)

    for page_index, page in enumerate(reader.pages):
        page_num = page_index + 1  # 1-based for human readability
        text = page.extract_text() or ""  # pypdf returns None for scanned pages

        if not text.strip():
            logger.warning(
                "  → Page %d of '%s' has no extractable text "
                "(possibly a scanned image). Skipping.",
                page_num,
                file_name,
            )

        yield {
            "text": text,
            "file_name": file_name,
            "page_num": page_num,
        }


def load_pdfs(file_paths: list[str | Path]) -> list[PageData]:
    """
    Load multiple PDFs and return all pages as a flat list.

    Scanned pages (empty text) are included in the list so the caller
    can decide what to do with them (e.g. show a warning in the UI).

    Args:
        file_paths: List of paths to PDF files.

    Returns:
        Flat list of PageData dicts for every page across all PDFs.
    """
    all_pages: list[PageData] = []
    for path in file_paths:
        try:
            pages = list(load_pdf(path))
            all_pages.extend(pages)
            logger.info("Loaded %d pages from '%s'.", len(pages), Path(path).name)
        except (FileNotFoundError, ValueError) as exc:
            logger.error("Could not load '%s': %s", path, exc)
    return all_pages


def has_extractable_text(pages: list[PageData]) -> bool:
    """Return True if at least one page has non-empty text."""
    return any(p["text"].strip() for p in pages)
