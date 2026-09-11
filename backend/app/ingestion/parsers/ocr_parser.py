"""
ocr_parser.py
=============
Handles pages with no usable native text layer (scanned documents) by
rendering the page to an image and running Tesseract OCR on it.

Why this file is needed:
    Some PDFs are literally photographs/scans of paper documents — there
    is no text layer at all, just pixel data. This is the ONLY path that
    can extract content from those pages.

Which other files use this:
    document_processor.py calls this only for pages that
    pdf_parser.needs_ocr() flagged as requiring it — we deliberately
    avoid running OCR on every page, since it is much slower and less
    accurate than native extraction where native extraction works.
"""

import io

import fitz
import pytesseract
from PIL import Image

from app.ingestion.schemas import ExtractionMethod, ParsedPage

# Render resolution for OCR. PDFs are normally rendered at 72 DPI
# (dots per inch) by default, which is fine for on-screen display but
# too low-resolution for accurate character recognition. Tesseract's
# own documentation recommends ~300 DPI as the sweet spot: high enough
# for character shapes to be well-defined, without ballooning render
# time and memory for very large documents.
OCR_RENDER_DPI = 300


def _render_page_to_image(page: "fitz.Page") -> Image.Image:
    """
    Converts a single PDF page into a PIL Image at OCR_RENDER_DPI.

    The leading underscore in the function name is a Python convention
    (not enforced by the language) signaling "this is an internal helper,
    not part of this module's public interface" — other files should
    call extract_ocr_pages() below, not this directly.
    """
    # PyMuPDF's coordinate system is based on 72 DPI by default. To
    # render at a higher DPI, we scale the rendering matrix by the
    # ratio of our target DPI to that baseline.
    zoom = OCR_RENDER_DPI / 72
    matrix = fitz.Matrix(zoom, zoom)

    pixmap = page.get_pixmap(matrix=matrix)  # type: ignore[attr-defined]

    # Pixmap stores raw pixel bytes in PNG-compatible form; we go
    # through an in-memory bytes buffer (io.BytesIO) rather than writing
    # a temp file to disk, since we don't need this image to persist —
    # it's a throwaway intermediate step for OCR only, and avoiding disk
    # I/O here is both faster and avoids needing to clean up temp files.
    png_bytes = pixmap.tobytes("png")
    return Image.open(io.BytesIO(png_bytes))


def extract_ocr_pages(pdf_path: str, page_numbers: list[int]) -> list[ParsedPage]:
    """
    Runs OCR on the specified 1-indexed page numbers of a PDF.

    We accept an explicit list of page numbers (rather than re-processing
    the whole document) because OCR is the slow path — this function
    should only ever be called for the specific pages that actually need
    it, decided by pdf_parser.needs_ocr() on the fast native-extraction
    pass.
    """
    results: list[ParsedPage] = []

    with fitz.open(pdf_path) as doc:
        for page_number in page_numbers:
            page = doc[page_number - 1]  # convert back to 0-indexed
            image = _render_page_to_image(page)

            # pytesseract.image_to_data returns per-word bounding boxes
            # AND confidence scores, structured as a dict of parallel
            # lists (one entry per detected word). We use this richer
            # call instead of the simpler image_to_string specifically
            # because we want the confidence scores for quality
            # monitoring, not just the text.
            ocr_result = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT
            )

            words = [w for w in ocr_result["text"] if w.strip()]
            text = " ".join(words)

            # Tesseract reports -1 confidence for non-text regions
            # (e.g. whitespace it detected but didn't attempt to read).
            # We only average confidence over entries that correspond to
            # actual recognized words, matching how we built `words`
            # above, so the reported confidence reflects real detections.
            confidences = [
                int(c)
                for c, w in zip(ocr_result["conf"], ocr_result["text"])
                if w.strip() and int(c) >= 0
            ]
            mean_confidence = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )

            results.append(
                ParsedPage(
                    page_number=page_number,
                    text=text,
                    extraction_method=ExtractionMethod.OCR,
                    ocr_confidence=round(mean_confidence, 2),
                )
            )

    return results
