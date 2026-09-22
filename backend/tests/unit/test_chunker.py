"""
test_chunker.py
================
Unit tests for chunker.py — formalizing the manual verification we did
back in Module 3 into real, repeatable pytest tests.

Why UNIT tests here (as opposed to going through the full API): these
tests construct ParsedDocument objects DIRECTLY in Python rather than
going through real PDF parsing — this makes them fast (no file I/O,
no PyMuPDF/Tesseract calls) and lets us precisely control the exact
input shape needed to exercise a specific code path (e.g. "a single
paragraph that alone exceeds the token budget"), which would be
awkward and fragile to engineer via an actual PDF file.
"""

from app.ingestion.chunker import MAX_CHUNK_TOKENS, chunk_document
from app.ingestion.schemas import ExtractionMethod, ParsedDocument, ParsedPage, ParsedTable


def _make_document(pages: list[ParsedPage], filename: str = "test.pdf") -> ParsedDocument:
    return ParsedDocument(source_filename=filename, total_pages=len(pages), pages=pages)


def test_short_document_produces_one_chunk():
    """A document well under the token budget should not be split at all."""
    page = ParsedPage(
        page_number=1, text="Short paragraph of text.", extraction_method=ExtractionMethod.NATIVE
    )
    chunks = chunk_document(_make_document([page]))

    assert len(chunks) == 1
    assert chunks[0].token_count <= MAX_CHUNK_TOKENS


def test_no_chunk_exceeds_token_budget():
    """
    Regression test for the real bug found in Module 3: a single
    oversized paragraph used to produce a chunk exceeding
    MAX_CHUNK_TOKENS. This test exists specifically so that bug can
    never silently come back.
    """
    long_paragraph = "This sentence repeats many times to force overflow. " * 60
    page = ParsedPage(
        page_number=1, text=long_paragraph, extraction_method=ExtractionMethod.NATIVE
    )
    chunks = chunk_document(_make_document([page]))

    # Small tolerance for the approximate fallback tokenizer's rounding
    # (documented in Module 3) — the fix in Module 12 made this nearly
    # exact (observed: 506/512), but a small margin remains appropriate
    # since count_tokens is fundamentally an approximation in this
    # environment, not a real tokenizer.
    tolerance = 8
    for chunk in chunks:
        assert chunk.token_count <= MAX_CHUNK_TOKENS + tolerance


def test_tables_are_never_merged_with_surrounding_text():
    """A table must always be its own standalone chunk."""
    page = ParsedPage(
        page_number=1,
        text="Some prose before the table.",
        extraction_method=ExtractionMethod.NATIVE,
        tables=[ParsedTable(page_number=1, rows=[["A", "B"], ["1", "2"]])],
    )
    chunks = chunk_document(_make_document([page]))

    table_chunks = [c for c in chunks if c.chunk_type.value == "table"]
    assert len(table_chunks) == 1
    assert "A" in table_chunks[0].text and "1" in table_chunks[0].text
    # The table's content should NOT appear inside any TEXT chunk.
    text_chunks = [c for c in chunks if c.chunk_type.value == "text"]
    for c in text_chunks:
        assert "| A | B |" not in c.text


def test_consecutive_chunks_share_overlap():
    """
    When a document is large enough to require multiple chunks, the
    end of one chunk should reappear at the start of the next —
    verifying the sliding-window overlap mechanism from Module 3.
    """
    pages = [
        ParsedPage(
            page_number=i,
            text=(f"Paragraph on page {i} discussing quarterly results in detail. " * 16),
            extraction_method=ExtractionMethod.NATIVE,
        )
        for i in range(1, 4)
    ]
    chunks = chunk_document(_make_document(pages))
    assert len(chunks) >= 2, "test setup should produce multiple chunks to actually test overlap"

    for i in range(len(chunks) - 1):
        tail_of_current = chunks[i].text[-30:]
        assert tail_of_current in chunks[i + 1].text, (
            f"Expected overlap between chunk {i} and chunk {i + 1}"
        )


def test_empty_page_produces_no_chunks():
    """A page with no text and no tables should contribute nothing."""
    page = ParsedPage(page_number=1, text="   ", extraction_method=ExtractionMethod.NATIVE)
    chunks = chunk_document(_make_document([page]))
    assert chunks == []
