"""
schemas.py
==========
Pydantic models describing the OUTPUT of the ingestion/parsing stage.

Why this file is needed:
    pdf_parser.py, ocr_parser.py, and table_parser.py each extract
    different kinds of content (native text, OCR'd text, tables) from a
    document. Without a shared, well-defined output shape, every
    downstream consumer (chunker, indexer) would need to know the
    quirks of each parser individually. These models are the CONTRACT
    between ingestion and everything that comes after it.

Which other files use this:
    pdf_parser.py, ocr_parser.py, and table_parser.py all construct and
    return these models. chunker.py (Module 3) will consume them.
"""

from enum import Enum

from pydantic import BaseModel, Field


class ExtractionMethod(str, Enum):
    """
    An Enum (short for "enumeration") restricts a value to one of a
    fixed, named set of options — here, exactly one of three strings.

    Why an Enum instead of just using plain strings like "native" or
    "ocr" everywhere?
        Plain strings are prone to typos ("natvie" would silently create
        a new, wrong category with no error). An Enum makes invalid
        values IMPOSSIBLE to construct — Pydantic will reject anything
        that isn't one of these three members with a clear validation
        error. It also gives you autocomplete in your editor.

    Why inherit from `str` as well as `Enum`?
        This makes each member behave as an actual string at runtime
        (e.g. `ExtractionMethod.NATIVE == "native"` is True), which
        means it serializes cleanly to JSON without extra configuration.
        Without the `str` base, Pydantic would need extra config to
        serialize Enum members to plain JSON strings.
    """

    NATIVE = "native"      # extracted directly from the PDF's text layer
    OCR = "ocr"             # extracted via Tesseract OCR on a rendered image
    TABLE = "table"         # extracted via pdfplumber's table detection


class ParsedTable(BaseModel):
    """Represents a single table extracted from a page."""

    page_number: int
    rows: list[list[str | None]] = Field(
        description=(
            "Table content as a list of rows, each row a list of cell "
            "strings. Cells can be None when pdfplumber can't confidently "
            "extract a value (e.g. a merged/empty cell)."
        )
    )

    def to_markdown(self) -> str:
        """
        Renders the table as a Markdown table string.

        Why Markdown, specifically?
            When we hand table content to an LLM later (Module 7), LLMs
            are demonstrably better at understanding tabular structure
            when it's formatted as Markdown tables rather than raw
            comma-separated or space-separated text — this is a widely
            observed practical finding, not just a stylistic choice.
        """
        if not self.rows:
            return ""
        lines = []
        header, *body = self.rows
        lines.append("| " + " | ".join(c or "" for c in header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for row in body:
            lines.append("| " + " | ".join(c or "" for c in row) + " |")
        return "\n".join(lines)


class ParsedPage(BaseModel):
    """Represents everything extracted from a single page of a document."""

    page_number: int = Field(ge=1, description="1-indexed page number")
    text: str = Field(default="", description="Extracted plain text content")
    extraction_method: ExtractionMethod
    tables: list[ParsedTable] = Field(default_factory=list)
    ocr_confidence: float | None = Field(
        default=None,
        description=(
            "Tesseract's mean confidence score (0-100) for this page, "
            "only populated when extraction_method is OCR. Used later to "
            "flag low-confidence pages for human review."
        ),
    )


class ParsedDocument(BaseModel):
    """The full result of parsing one document, made up of parsed pages."""

    source_filename: str
    total_pages: int
    pages: list[ParsedPage]

    @property
    def full_text(self) -> str:
        """Convenience accessor: all page text concatenated in order."""
        return "\n\n".join(p.text for p in self.pages)

    @property
    def ocr_page_count(self) -> int:
        """How many pages required OCR — useful for logging/monitoring."""
        return sum(
            1 for p in self.pages if p.extraction_method == ExtractionMethod.OCR
        )
