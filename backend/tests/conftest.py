"""
conftest.py
===========
Shared pytest fixtures, automatically discovered by pytest for every
test file in this directory tree (no explicit import needed — this is
a pytest convention: any file literally named conftest.py is loaded
automatically).

Why fixtures instead of just writing setup code in each test?
    A fixture is REUSABLE, composable setup/teardown logic. Multiple
    test files need "a sample PDF" or "a test database session" —
    without fixtures, that setup code would be copy-pasted across every
    test file (and every fix to it would need repeating everywhere).
    pytest also handles fixture SCOPE (function/module/session) and
    automatic cleanup (via `yield`) for us.
"""

import tempfile

import fitz
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.postgres import Base, get_db
from app.main import app
from app.models.document import DocumentRecord  # noqa: F401 -- registers the table with Base


@pytest.fixture()
def native_pdf_path(tmp_path):
    """
    Creates a small NATIVE-text PDF (a real text layer, no OCR needed)
    and returns its path.

    Why `tmp_path` (a built-in pytest fixture) instead of a fixed path
    like /tmp/test_pdfs: `tmp_path` gives every individual test its own
    fresh, automatically-cleaned-up temporary directory — tests never
    interfere with each other by accidentally sharing files, and
    nothing is left behind after the test run. This also makes the
    suite fully HERMETIC: it doesn't depend on any file existing
    outside of what the test itself creates, so it runs identically on
    any machine or CI runner, not just this sandbox.
    """
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), "Regional Breakdown\nAPAC revenue grew 18% this quarter.", fontsize=11)
    path = tmp_path / "native_report.pdf"
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture()
def scanned_pdf_path(tmp_path):
    """Creates a genuinely image-only PDF (no text layer) to exercise the OCR path."""
    src = fitz.open()
    p = src.new_page()
    p.insert_text((50, 100), "Employment Agreement", fontsize=18)
    p.insert_text((50, 150), "This contract is effective January 1, 2026.", fontsize=12)
    pix = p.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_path = tmp_path / "_scan_source.png"
    pix.save(str(img_path))
    src.close()

    scanned = fitz.open()
    rect = fitz.Rect(0, 0, 595, 842)
    page = scanned.new_page(width=rect.width, height=rect.height)
    page.insert_image(rect, filename=str(img_path))
    out_path = tmp_path / "scanned_contract.pdf"
    scanned.save(str(out_path))
    scanned.close()
    return str(out_path)


@pytest.fixture()
def test_db_engine():
    """
    An isolated in-memory SQLite engine, one per test, with StaticPool
    (see the Module 11 discussion of why StaticPool is required for
    SQLite :memory: mode to work correctly with multiple connections).

    Honest, documented limitation: this validates our SQLAlchemy MODEL
    and QUERY logic, not Postgres-specific behavior — a real CI
    pipeline would ALSO run a smaller set of integration tests against
    a real Postgres instance (e.g. via a Postgres service container in
    GitHub Actions, wired up in Module 11's CI workflow) for full
    confidence, in addition to these fast, dependency-free SQLite tests.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture(autouse=True)
def _reset_singletons():
    """
    get_retriever() and get_reranker() (dependencies.py) are @lru_cache'd
    — deliberately, so the real app only loads models ONCE per server
    process (see Module 8's rationale). But that exact same caching
    becomes a TEST ISOLATION BUG here: without resetting it, a document
    uploaded in one test would silently still be present in the
    retriever's index when a LATER, unrelated test runs — tests would
    pass or fail depending on execution ORDER, which is one of the most
    insidious classes of flaky test.

    `autouse=True` means this fixture runs automatically for EVERY test
    in the suite without needing to be explicitly requested — exactly
    what we want here, since this isn't optional setup, it's a
    correctness requirement for the whole suite.
    """
    from app.api.dependencies import get_reranker, get_retriever

    get_retriever.cache_clear()
    get_reranker.cache_clear()
    yield
    get_retriever.cache_clear()
    get_reranker.cache_clear()


@pytest.fixture()
def client(test_db_engine):
    """
    A FastAPI TestClient with the real `get_db` dependency swapped for
    one bound to the isolated test database — every test gets this
    automatically via pytest's fixture injection (just name `client` as
    a test function argument) rather than needing to repeat this setup
    in every test file.
    """
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()
    # Clearing dependency_overrides after each test is important: without
    # it, overrides from one test could silently leak into a LATER test
    # that never asked for them, causing confusing cross-test contamination.
