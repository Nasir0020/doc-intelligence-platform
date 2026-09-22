"""
test_query_api.py
==================
Integration tests for the full retrieve -> rerank -> generate pipeline,
exercised through the real HTTP layer, with the Anthropic API mocked
(same justification as test_generator.py — we don't spend real API
credits or require a real key just to verify OUR wiring is correct).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_query_before_any_upload_returns_404(client):
    response = client.post("/query", json={"question": "What is the APAC growth?"})
    assert response.status_code == 404


def test_query_rejects_empty_question(client):
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422


def test_query_rejects_missing_question_field(client):
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_full_query_flow_with_mocked_llm(client, native_pdf_path):
    """
    The end-to-end happy path: upload a real document, ask a question
    about it, and verify a grounded, cited answer comes back through
    the real API layer.
    """
    with open(native_pdf_path, "rb") as f:
        upload_response = client.post(
            "/documents/upload", files={"file": ("native_report.pdf", f, "application/pdf")}
        )
    assert upload_response.status_code == 201

    fake_answer = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="APAC revenue grew 18% [Source 1].")]
    )
    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = fake_answer
        MockAnthropic.return_value = mock_client

        response = client.post(
            "/query", json={"question": "What was the APAC growth?", "top_k": 3}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert len(body["citations"]) == 1
    assert body["citations"][0]["source_filename"] == "native_report.pdf"


def test_query_returns_502_when_llm_call_fails(client, native_pdf_path):
    """
    Verifies our 502 (upstream failure) semantics from Module 8: when
    the underlying Anthropic call raises, the API should surface a
    clear 502, not an opaque 500.
    """
    with open(native_pdf_path, "rb") as f:
        client.post(
            "/documents/upload", files={"file": ("native_report.pdf", f, "application/pdf")}
        )

    with patch("anthropic.Anthropic") as MockAnthropic:
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("simulated API outage")
        MockAnthropic.return_value = mock_client

        response = client.post("/query", json={"question": "What was the APAC growth?"})

    assert response.status_code == 502
