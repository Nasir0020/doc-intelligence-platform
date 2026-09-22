"""
documents.py
============
Endpoints for uploading documents into the system and listing what's
been indexed so far.

Which other files use this:
    main.py mounts this router. The frontend (Module 10) will call
    POST /documents/upload when a user drags in a file.
"""

import logging
import os
import tempfile

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.dependencies import get_retriever
from app.db.postgres import get_db
from app.ingestion.chunker import chunk_document
from app.ingestion.document_processor import process_pdf
from app.ingestion.embedder import get_embedder
from app.models.document import DocumentRecord
from app.retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


class DocumentSummary(BaseModel):
    """API response shape for a successfully indexed document."""

    filename: str
    total_pages: int
    total_chunks: int
    ocr_pages_used: int


@router.post("/upload", response_model=DocumentSummary, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    retriever: HybridRetriever = Depends(get_retriever),
    db: Session = Depends(get_db),
) -> DocumentSummary:
    """
    Accepts a PDF upload, runs it through the full ingestion pipeline
    (parse -> chunk -> embed -> index), persists its metadata to
    Postgres, and returns a summary.

    Why `async def` here specifically (recall Module 1's discussion of
    async vs sync routes): `await file.read()` is a genuinely async
    I/O operation (reading the uploaded file's bytes off the network
    connection). The CPU-heavy parts that follow (PDF parsing, OCR,
    embedding) are all blocking/synchronous — FastAPI automatically
    runs `async def` route bodies on the event loop, but since we don't
    have other concurrent requests contending for that specific loop
    tick in THIS simple version, this is acceptable here. A
    higher-throughput production version would push the blocking
    parsing/embedding work into a background task queue (Celery, or
    FastAPI's own BackgroundTasks) instead of blocking the request
    entirely — noted as a real scaling improvement for Module 13, not
    implemented here to keep this module focused on wiring.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    existing = db.scalar(
        select(DocumentRecord).where(DocumentRecord.filename == file.filename)
    )
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail=f"A document named '{file.filename}' has already been indexed.",
        )

    # Our parsing pipeline (PyMuPDF, pdfplumber) operates on file PATHS,
    # not in-memory bytes, so we write the upload to a temp file first.
    # delete=False is deliberate: on some platforms, a NamedTemporaryFile
    # opened for writing can't be simultaneously reopened for reading by
    # another library (like PyMuPDF) while still held open by Python —
    # so we close our own handle (the `with` block exits) before handing
    # the path to process_pdf, and clean the file up explicitly in the
    # `finally` block below instead of relying on automatic deletion.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        parsed = process_pdf(tmp_path, file.filename)
        chunks = chunk_document(parsed)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No extractable content found in this document.",
            )

        embedder = get_embedder()
        vectors = embedder.embed_documents([c.text for c in chunks])

        retriever.add_chunks(
            chunk_ids=[c.chunk_id for c in chunks],
            vectors=vectors,
            texts=[c.text for c in chunks],
            page_numbers=[c.page_numbers for c in chunks],
        )

        record = DocumentRecord(
            filename=file.filename,
            total_pages=parsed.total_pages,
            total_chunks=len(chunks),
            ocr_pages_used=parsed.ocr_page_count,
        )
        # add() stages the new row in this session; commit() actually
        # sends the INSERT and makes it durable; refresh() re-reads the
        # row back from the database, populating any DB-generated
        # values (like the auto-incremented `id`) onto our Python
        # object — without refresh(), `record.id` would still be None
        # after commit(), since we never set it ourselves.
        db.add(record)
        db.commit()
        db.refresh(record)

        logger.info(
            "Indexed '%s': %d pages, %d chunks (%d via OCR).",
            file.filename,
            parsed.total_pages,
            len(chunks),
            parsed.ocr_page_count,
        )
        return DocumentSummary(
            filename=record.filename,
            total_pages=record.total_pages,
            total_chunks=record.total_chunks,
            ocr_pages_used=record.ocr_pages_used,
        )

    finally:
        # Always clean up the temp file, whether ingestion succeeded or
        # raised — a `finally` block runs in BOTH cases, unlike code
        # placed after the try block (which is skipped if an exception
        # propagates out).
        os.unlink(tmp_path)


@router.get("", response_model=list[DocumentSummary])
def list_documents(db: Session = Depends(get_db)) -> list[DocumentSummary]:
    """Returns every document indexed so far, read from Postgres."""
    records = db.scalars(select(DocumentRecord)).all()
    return [
        DocumentSummary(
            filename=r.filename,
            total_pages=r.total_pages,
            total_chunks=r.total_chunks,
            ocr_pages_used=r.ocr_pages_used,
        )
        for r in records
    ]
