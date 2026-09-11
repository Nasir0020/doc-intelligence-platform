"""
table_parser.py
================
Detects and extracts tables from native (non-scanned) PDF pages using
pdfplumber.

Why this file is needed:
    Tables carry structured, often numerical information (financial
    figures, comparison data) that gets badly mangled if we treat it as
    plain flowing text — column alignment is meaningful and plain text
    extraction destroys it. This is a genuinely different extraction
    problem from prose text, hence its own dedicated module.

Which other files use this:
    document_processor.py calls this alongside pdf_parser.py for native
    pages (scanned pages don't get table extraction in this version —
    that would require a layout-detection model, which we note as a
    known limitation rather than silently ignoring it).

Known limitation (stated explicitly, not hidden):
    pdfplumber relies on detecting visual structure (lines, whitespace
    alignment) in the PDF. Tables with no visible gridlines and
    inconsistent spacing can be missed or misaligned. For production use
    at scale, a layout-detection model (e.g. Microsoft's Table
    Transformer) would improve recall on messy real-world tables — noted
    here as a documented tradeoff, not a silent gap.
"""

import pdfplumber

from app.ingestion.schemas import ParsedTable


def extract_tables(pdf_path: str) -> list[ParsedTable]:
    """
    Extracts every detected table across all pages of a PDF.

    pdfplumber re-opens and re-parses the PDF independently of PyMuPDF
    (used in pdf_parser.py). Running two different parsing libraries
    over the same file is intentional, not wasteful: PyMuPDF is
    optimized for fast raw text extraction, while pdfplumber's strength
    is specifically geometric/structural table detection. No single
    library we evaluated does both extremely well, so we use each for
    what it's best at — a common pattern in production data pipelines
    ("use the right tool per sub-task" rather than forcing one library
    to do everything adequately).
    """
    tables: list[ParsedTable] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1

            # find_tables() returns Table objects with both the detected
            # geometry AND an .extract() method that pulls the actual
            # cell text out as a list of rows.
            for raw_table in page.find_tables():
                extracted_rows = raw_table.extract()

                # Skip tables pdfplumber "detected" but which have no
                # real content (can happen on pages with table-like
                # visual formatting that isn't actually a data table,
                # e.g. a bordered box around a pull-quote).
                if not extracted_rows or not any(
                    any(cell for cell in row) for row in extracted_rows
                ):
                    continue

                tables.append(
                    ParsedTable(page_number=page_number, rows=extracted_rows)
                )

    return tables
