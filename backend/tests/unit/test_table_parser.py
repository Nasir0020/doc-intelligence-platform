"""
test_table_parser.py
=====================
Unit tests for table_parser.py, using real PDFs built on the fly (a
plain-text "fake table" with no gridlines, and a genuinely ruled table)
to verify pdfplumber's documented limitation (Module 2) is exactly what
we expect, not just asserted in a comment.
"""

import fitz

from app.ingestion.parsers.table_parser import extract_tables


def test_ruled_table_is_extracted(tmp_path):
    doc = fitz.open()
    page = doc.new_page()
    table_data = [["Region", "Growth"], ["APAC", "18%"]]
    col_x = [50, 200, 320]
    row_h = 25
    top = 70
    for r, row in enumerate(table_data):
        y0 = top + r * row_h
        for c, cell in enumerate(row):
            page.insert_text((col_x[c] + 5, y0 + 17), cell, fontsize=10)
        page.draw_line((col_x[0], y0), (col_x[-1], y0), width=0.75)
    page.draw_line(
        (col_x[0], top + len(table_data) * row_h),
        (col_x[-1], top + len(table_data) * row_h),
        width=0.75,
    )
    for x in col_x:
        page.draw_line((x, top), (x, top + len(table_data) * row_h), width=0.75)

    path = tmp_path / "ruled_table.pdf"
    doc.save(str(path))
    doc.close()

    tables = extract_tables(str(path))
    assert len(tables) == 1
    assert any("APAC" in cell for row in tables[0].rows for cell in row if cell)


def test_unruled_text_is_not_detected_as_table(tmp_path):
    """
    Documents pdfplumber's real, known limitation (Module 2): plain
    space-aligned text with no visual gridlines is NOT detected as a
    table. This test exists so that limitation stays a known, verified
    fact rather than only a claim in a comment.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Region      Growth\nAPAC        18%\nEurope      9%", fontsize=10)
    path = tmp_path / "unruled_text.pdf"
    doc.save(str(path))
    doc.close()

    tables = extract_tables(str(path))
    assert tables == []


def test_empty_page_yields_no_tables(tmp_path):
    doc = fitz.open()
    doc.new_page()
    path = tmp_path / "blank.pdf"
    doc.save(str(path))
    doc.close()

    assert extract_tables(str(path)) == []
