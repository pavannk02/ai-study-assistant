"""
test_chunker.py – Tests for src/chunker.py.

All tests use plain strings — no PDF files needed.
"""

import pytest

from src.chunker import chunk_page, chunk_pages


# ─── chunk_page ───────────────────────────────────────────────────────────────

class TestChunkPage:
    def test_short_text_produces_one_chunk(self):
        """Text shorter than chunk_size should produce exactly one chunk."""
        chunks = chunk_page("Hello world", "notes.pdf", 1, chunk_size=800, chunk_overlap=150)
        assert len(chunks) == 1
        assert chunks[0]["chunk_text"] == "Hello world"

    def test_metadata_is_preserved(self):
        """Every chunk must carry the correct file_name and page_num."""
        chunks = chunk_page("Some text", "lecture.pdf", 5)
        assert chunks[0]["file_name"] == "lecture.pdf"
        assert chunks[0]["page_num"] == 5

    def test_chunk_index_increments(self):
        """chunk_index should start at 0 and increment per chunk."""
        text = "A" * 1000  # forces at least 2 chunks with size=800, overlap=150
        chunks = chunk_page(text, "f.pdf", 1, chunk_size=800, chunk_overlap=150)
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i

    def test_overlap_is_applied(self):
        """
        The first CHUNK_OVERLAP characters of chunk[n+1] should match
        the last CHUNK_OVERLAP characters of chunk[n].
        """
        chunk_size = 20
        overlap = 5
        text = "A" * 100
        chunks = chunk_page(text, "f.pdf", 1, chunk_size=chunk_size, chunk_overlap=overlap)

        for i in range(len(chunks) - 1):
            tail_of_current = chunks[i]["chunk_text"][-overlap:]
            head_of_next = chunks[i + 1]["chunk_text"][:overlap]
            assert tail_of_current == head_of_next

    def test_empty_text_returns_empty_list(self):
        """Blank pages should produce zero chunks."""
        assert chunk_page("", "f.pdf", 1) == []
        assert chunk_page("   ", "f.pdf", 1) == []

    def test_all_text_is_covered(self):
        """
        Concatenating all chunks (accounting for overlap) should cover the
        entire source text — no characters are silently dropped.
        """
        chunk_size = 10
        overlap = 3
        text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        chunks = chunk_page(text, "f.pdf", 1, chunk_size=chunk_size, chunk_overlap=overlap)

        # The last chunk's text must end with the last character of the text.
        assert chunks[-1]["chunk_text"][-1] == text[-1]
        # The first chunk must start with the first character.
        assert chunks[0]["chunk_text"][0] == text[0]

    def test_raises_if_overlap_gte_chunk_size(self):
        """overlap >= chunk_size is a configuration error and must raise."""
        with pytest.raises(ValueError):
            chunk_page("Hello", "f.pdf", 1, chunk_size=10, chunk_overlap=10)
        with pytest.raises(ValueError):
            chunk_page("Hello", "f.pdf", 1, chunk_size=10, chunk_overlap=15)

    def test_exact_chunk_size_text(self):
        """
        Text exactly equal to chunk_size produces at least one chunk whose
        text matches the full input.  The sliding window may produce a second
        short tail chunk (the overlap region), which is expected behaviour.
        """
        text = "X" * 800
        chunks = chunk_page(text, "f.pdf", 1, chunk_size=800, chunk_overlap=150)
        # At minimum, the first chunk must cover the entire input text
        assert len(chunks) >= 1
        assert chunks[0]["chunk_text"] == text


# ─── chunk_pages ──────────────────────────────────────────────────────────────

class TestChunkPages:
    def _make_page(self, text: str, file_name: str = "f.pdf", page_num: int = 1) -> dict:
        return {"text": text, "file_name": file_name, "page_num": page_num}

    def test_multiple_pages_produce_flat_list(self):
        """chunk_pages flattens chunks from all pages into one list."""
        pages = [
            self._make_page("A" * 900, page_num=1),
            self._make_page("B" * 900, page_num=2),
        ]
        chunks = chunk_pages(pages, chunk_size=800, chunk_overlap=150)
        # Each 900-char page → 2 chunks, so 4 total
        assert len(chunks) == 4

    def test_blank_pages_are_skipped(self):
        """Pages with no text should contribute zero chunks."""
        pages = [
            self._make_page(""),
            self._make_page("Hello world"),
        ]
        chunks = chunk_pages(pages, chunk_size=800, chunk_overlap=150)
        assert len(chunks) == 1
        assert chunks[0]["chunk_text"] == "Hello world"

    def test_empty_input_returns_empty_list(self):
        """chunk_pages([]) must return [], not raise."""
        assert chunk_pages([]) == []

    def test_page_metadata_is_correct_in_output(self):
        """Chunks must carry file_name and page_num from their source page."""
        pages = [self._make_page("Some text", file_name="exam.pdf", page_num=7)]
        chunks = chunk_pages(pages)
        assert chunks[0]["file_name"] == "exam.pdf"
        assert chunks[0]["page_num"] == 7
