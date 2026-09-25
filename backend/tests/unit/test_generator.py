"""
test_generator.py
==================
Unit tests for generator.py's prompt construction and citation
extraction — formalizing Module 7's manual mock-based verification.

The Ollama/OpenAI-compatible LLM call is MOCKED here (no real network
request or API key needed) — we're testing OUR OWN logic (prompt
building, citation parsing, and response handling), not the third-party
LLM service itself.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.db.vector_store import SearchResult
from app.retrieval.generator import (
    _build_user_prompt,
    _extract_citations,
    generate_answer,
)

SOURCES = [
    SearchResult(
        chunk_id="doc.pdf_chunk_0000",
        score=0.9,
        text="Regional breakdown.",
        page_numbers=[1],
    ),
    SearchResult(
        chunk_id="doc.pdf_chunk_0001",
        score=0.85,
        text="APAC grew 18%.",
        page_numbers=[1],
    ),
]


def test_prompt_includes_numbered_sources_and_question():
    prompt = _build_user_prompt("What was the APAC growth?", SOURCES)
    assert "[Source 1]" in prompt
    assert "[Source 2]" in prompt
    assert "What was the APAC growth?" in prompt


def test_extract_citations_maps_source_numbers_to_metadata():
    response_text = (
        "APAC grew 18% [Source 2], based on the regional breakdown [Source 1]."
    )
    citations = _extract_citations(response_text, SOURCES)

    assert len(citations) == 2
    assert citations[0].source_number == 1
    assert citations[0].chunk_id == "doc.pdf_chunk_0000"
    assert citations[0].source_filename == "doc.pdf"


def test_extract_citations_drops_out_of_range_source_numbers():
    """
    Regression test for the real defensive-coding case built in Module
    7: a model hallucinating a citation number outside the range of
    sources it was actually given must be dropped, not included.
    """
    response_text = "Revenue grew significantly [Source 99]."
    citations = _extract_citations(response_text, SOURCES)
    assert citations == []


def test_extract_citations_deduplicates_repeated_markers():
    response_text = "APAC grew [Source 1]. This is confirmed again [Source 1]."
    citations = _extract_citations(response_text, SOURCES)
    assert len(citations) == 1


def test_generate_answer_with_no_sources_short_circuits():
    """No sources should mean no LLM call is made at all."""
    with patch("app.retrieval.generator.OpenAI") as MockOpenAI:
        result = generate_answer("Anything?", [])
        MockOpenAI.assert_not_called()

    assert result.grounded is False
    assert result.citations == []


def test_generate_answer_insufficient_context_path():
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="INSUFFICIENT_CONTEXT: not enough info."
                )
            )
        ]
    )

    with patch("app.retrieval.generator.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        result = generate_answer("What was headcount?", SOURCES)

    assert result.grounded is False
    assert result.citations == []


def test_generate_answer_grounded_path_with_valid_citations():
    mock_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="APAC grew 18% [Source 2]."
                )
            )
        ]
    )

    with patch("app.retrieval.generator.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        result = generate_answer("What was the APAC growth?", SOURCES)

    assert result.grounded is True
    assert len(result.citations) == 1
    assert result.citations[0].chunk_id == "doc.pdf_chunk_0001"