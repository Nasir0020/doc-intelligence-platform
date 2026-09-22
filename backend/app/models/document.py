"""
document.py (models)
=====================
The SQLAlchemy ORM model for a persisted document record — the durable
counterpart to the DocumentSummary Pydantic model used in the API layer.

Why TWO separate classes (this ORM model AND the Pydantic
DocumentSummary in api/routes/documents.py) representing similar data?
    This is a deliberate separation of concerns: the ORM model
    describes how data is SHAPED IN THE DATABASE (with DB-specific
    concerns like primary keys and indexes); the Pydantic model
    describes how data is SHAPED IN THE API RESPONSE (what the
    frontend actually needs to see). They usually look similar, but
    conflating them means every future database-only concern (like
    adding an internal `updated_at` audit column) would leak directly
    into the public API contract unless we remember to explicitly
    exclude it — keeping them separate makes that exclusion the
    default, not something we have to remember.

Which other files use this:
    api/routes/documents.py queries and creates instances of this model.
"""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.postgres import Base


class DocumentRecord(Base):
    __tablename__ = "documents"

    # Mapped[int] + mapped_column(...) is SQLAlchemy 2.0's typed ORM
    # syntax: the Python type hint (Mapped[int]) and the actual SQL
    # column definition (mapped_column(...)) work together so both
    # SQLAlchemy AND static type checkers (mypy) know this attribute is
    # genuinely an int, not just "some database value."
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    filename: Mapped[str] = mapped_column(String, unique=True, index=True)
    # unique=True: the database itself enforces no two rows can share a
    # filename (a stronger guarantee than checking in application code,
    # which has a race-condition window between "check" and "insert"
    # under concurrent requests). index=True: filename lookups (which
    # we do on every upload, to detect re-uploads) are the most common
    # query against this table, so we tell the database to build a
    # index structure for fast lookups rather than scanning every row.

    total_pages: Mapped[int] = mapped_column(Integer)
    total_chunks: Mapped[int] = mapped_column(Integer)
    ocr_pages_used: Mapped[int] = mapped_column(Integer)

    # `default=lambda: ...` (a CALLABLE), not `default=datetime.now(...)`
    # (a fixed VALUE): the lambda is evaluated fresh at INSERT time for
    # EVERY row. Using a plain value instead would compute
    # datetime.now() exactly once, at class-definition time (when
    # Python first imports this module) — every single row would then
    # get that same, frozen timestamp forever, which is almost never
    # what you want for an "uploaded_at" column.
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
