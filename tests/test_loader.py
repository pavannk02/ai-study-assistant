"""
test_loader.py – Tests for src/loader.py.

Uses tmp_path (pytest's built-in temp directory fixture) to create real
temporary PDF files on disk.  No mocking needed because pypdf is stable.

Note on make_simple_pdf: pypdf 6.x requires PdfObject instances for page
dictionary keys, so we cannot use the old string-key content-stream trick.
Blank pages are used instead — the tests here only check page counts and
metadata, not text content, so blank pages are sufficient.
"""

import pytest
from pypdf import PdfWriter

from src.loader import load_pdf, load_pdfs, has_extractable_text


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_simple_pdf(path, num_pages: int = 2) -> None:
    """
    Create a PDF with ``num_pages`` blank pages.

    Blank pages are enough for tests that only check page counts and
    metadata (file_name, page_num).  pypdf returns an empty string for
    blank pages, which is fine — we test the text value separately.
    """
    writer = PdfWriter()
    for _ in range(num_pages):
        writer.add_blank_page(width=612, height=792)
    with open(path, "wb") as f:
        writer.write(f)


# ─── load_pdf ─────────────────────────────────────────────────────────────────

class TestLoadPdf:
    def test_yields_correct_number_of_pages(self, tmp_path):
        """load_pdf should yield one dict per page."""
        pdf = tmp_path / "sample.pdf"
        make_simple_pdf(pdf, num_pages=3)

        pages = list(load_pdf(pdf))
        assert len(pages) == 3

    def test_page_metadata_is_correct(self, tmp_path):
        """Each page dict must have the right file_name and 1-based page_num."""
        pdf = tmp_path / "notes.pdf"
        make_simple_pdf(pdf, num_pages=2)

        pages = list(load_pdf(pdf))

        assert pages[0]["file_name"] == "notes.pdf"
        assert pages[0]["page_num"] == 1
        assert pages[1]["page_num"] == 2

    def test_text_key_is_always_a_string(self, tmp_path):
        """Even for blank pages the text key must be a str (never None)."""
        pdf = tmp_path / "blank.pdf"
        make_simple_pdf(pdf, num_pages=1)  # blank page — pypdf returns "" not None

        pages = list(load_pdf(pdf))
        assert isinstance(pages[0]["text"], str)

    def test_raises_file_not_found(self, tmp_path):
        """load_pdf must raise FileNotFoundError for a missing file."""
        with pytest.raises(FileNotFoundError):
            list(load_pdf(tmp_path / "nonexistent.pdf"))

    def test_raises_value_error_for_non_pdf(self, tmp_path):
        """load_pdf must raise ValueError for non-PDF extensions."""
        txt_file = tmp_path / "document.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError):
            list(load_pdf(txt_file))


# ─── load_pdfs ────────────────────────────────────────────────────────────────

class TestLoadPdfs:
    def test_loads_multiple_pdfs(self, tmp_path):
        """load_pdfs should concatenate pages from all given files."""
        pdf1 = tmp_path / "a.pdf"
        pdf2 = tmp_path / "b.pdf"
        make_simple_pdf(pdf1, num_pages=2)
        make_simple_pdf(pdf2, num_pages=3)

        pages = load_pdfs([pdf1, pdf2])
        assert len(pages) == 5

    def test_bad_path_is_skipped_not_raised(self, tmp_path):
        """load_pdfs should skip invalid files and return the rest."""
        good = tmp_path / "good.pdf"
        make_simple_pdf(good, num_pages=1)
        bad = tmp_path / "missing.pdf"  # does not exist

        pages = load_pdfs([good, bad])
        # Only the 1 good page should be returned
        assert len(pages) == 1

    def test_empty_list_returns_empty(self):
        """load_pdfs([]) must return an empty list, not raise."""
        assert load_pdfs([]) == []


# ─── has_extractable_text ─────────────────────────────────────────────────────

class TestHasExtractableText:
    def test_returns_true_when_text_present(self):
        pages = [{"text": "Hello world", "file_name": "f.pdf", "page_num": 1}]
        assert has_extractable_text(pages) is True

    def test_returns_false_when_all_blank(self):
        pages = [
            {"text": "", "file_name": "f.pdf", "page_num": 1},
            {"text": "   ", "file_name": "f.pdf", "page_num": 2},
        ]
        assert has_extractable_text(pages) is False

    def test_returns_false_for_empty_list(self):
        assert has_extractable_text([]) is False
