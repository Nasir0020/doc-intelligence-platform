"""
test_documents_api.py
======================
Integration tests: exercising the REAL HTTP layer (FastAPI's
TestClient), the REAL parsing/chunking/embedding pipeline, and a REAL
(SQLite-substitute) database — everything except the Anthropic API,
which never gets called by these endpoints anyway.

This is the difference between "unit" and "integration" tests in this
suite: unit tests isolate one function/module with everything else
faked out; these tests run the actual wiring between multiple real
modules, catching the kind of bug that only shows up at integration
boundaries (exactly like the empty-corpus bug found in Module 8).
"""


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/documents/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 400


def test_upload_and_list_native_pdf(client, native_pdf_path):
    with open(native_pdf_path, "rb") as f:
        response = client.post(
            "/documents/upload", files={"file": ("native_report.pdf", f, "application/pdf")}
        )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "native_report.pdf"
    assert body["total_chunks"] > 0
    assert body["ocr_pages_used"] == 0

    list_response = client.get("/documents")
    assert list_response.status_code == 200
    assert len(list_response.json()) == 1


def test_upload_scanned_pdf_uses_ocr(client, scanned_pdf_path):
    """
    Verifies the OCR path is actually exercised through the real HTTP
    endpoint, not just at the unit level (Module 2) — confirming the
    full parse -> chunk -> embed -> index pipeline correctly handles a
    scanned document end to end.
    """
    with open(scanned_pdf_path, "rb") as f:
        response = client.post(
            "/documents/upload", files={"file": ("scanned_contract.pdf", f, "application/pdf")}
        )

    assert response.status_code == 201
    body = response.json()
    assert body["ocr_pages_used"] == 1


def test_duplicate_filename_upload_is_rejected(client, native_pdf_path):
    """
    Regression test for the real database-level uniqueness constraint
    built in Module 11.
    """
    with open(native_pdf_path, "rb") as f:
        first = client.post(
            "/documents/upload", files={"file": ("native_report.pdf", f, "application/pdf")}
        )
    assert first.status_code == 201

    with open(native_pdf_path, "rb") as f:
        second = client.post(
            "/documents/upload", files={"file": ("native_report.pdf", f, "application/pdf")}
        )
    assert second.status_code == 409


def test_list_documents_empty_by_default(client):
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == []
