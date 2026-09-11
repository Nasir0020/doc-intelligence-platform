"""
document_processor.py
======================
The orchestrator for the ingestion/parsing stage. This is the ONE
function the rest of the application calls to turn a raw PDF file into
a structured ParsedDocument — callers don't need to know anything about
native extraction, OCR, or table detection individually.

Why this file is needed:
    Without an orchestrator, every caller (the upload API endpoint,
    background workers, tests) would need to independently know the
    correct order of operations: try native extraction, check which
    pages need OCR, run OCR only on those, separately extract tables,
    then merge everything into one coherent result. Centralizing that
    logic here means it's implemented (and tested) exactly once.

Which other files use this:
    api/routes/documents.py (built in Module 8) will call process_pdf()
    when a user uploads a file.
"""

import logging

from app.ingestion.parsers.ocr_parser import extract_ocr_pages
from app.ingestion.parsers.pdf_parser import extract_native_pages, needs_ocr
from app.ingestion.parsers.table_parser import extract_tables
from app.ingestion.schemas import ParsedDocument, ParsedPage

# Standard library logging, not print(). Why this matters:
#   print() output is unstructured and impossible to filter by severity
#   or route to a monitoring system (e.g. CloudWatch, Datadog) in
#   production. logging lets us later configure WHERE these messages go
#   and at WHAT severity threshold, without touching this code at all.
logger = logging.getLogger(__name__)


def process_pdf(pdf_path: str, source_filename: str) -> ParsedDocument:
    """
    Full ingestion pipeline for a single PDF file:

    1. Extract native text from every page (fast path).
    2. Identify which pages had insufficient native text.
    3. Run OCR only on those flagged pages (slow path, used sparingly).
    4. Extract tables from all pages.
    5. Merge everything into one ParsedDocument.
    """
    native_pages = extract_native_pages(pdf_path)

    pages_needing_ocr = [p.page_number for p in native_pages if needs_ocr(p)]

    if pages_needing_ocr:
        logger.info(
            "OCR required for %d/%d pages of %s",
            len(pages_needing_ocr),
            len(native_pages),
            source_filename,
        )
        ocr_pages = extract_ocr_pages(pdf_path, pages_needing_ocr)
    else:
        ocr_pages = []

    # Build a lookup so we can replace low-text native pages with their
    # OCR'd counterparts by page number, keeping every other native page
    # untouched.
    ocr_by_page_number = {p.page_number: p for p in ocr_pages}

    final_pages: list[ParsedPage] = [
        ocr_by_page_number.get(p.page_number, p) for p in native_pages
    ]

    tables = extract_tables(pdf_path)

    # Attach each extracted table to the ParsedPage it belongs to, so
    # downstream chunking (Module 3) can keep tables together with their
    # surrounding page context instead of treating them as a separate,
    # disconnected list.
    tables_by_page: dict[int, list] = {}
    for table in tables:
        tables_by_page.setdefault(table.page_number, []).append(table)

    for page in final_pages:
        page.tables = tables_by_page.get(page.page_number, [])

    return ParsedDocument(
        source_filename=source_filename,
        total_pages=len(final_pages),
        pages=final_pages,
    )
