"""
schemas.py (retrieval)
=======================
Data models for the generation stage's output — the final answer plus
structured citation metadata the frontend can use to link back to exact
source pages.

Which other files use this:
    generator.py constructs these. api/routes/query.py (Module 8) will
    return a GeneratedAnswer directly as the API response body.
"""

from pydantic import BaseModel


class Citation(BaseModel):
    """One specific source reference used in the generated answer."""

    source_number: int  # matches the [Source N] marker shown to the model
    chunk_id: str
    source_filename: str
    page_numbers: list[int]


class GeneratedAnswer(BaseModel):
    """The final output of the whole RAG pipeline for one user question."""

    answer_text: str
    citations: list[Citation]
    grounded: bool = True  # False if the model indicated it couldn't answer from context
