"""
postgres.py
===========
SQLAlchemy engine, session factory, and the FastAPI dependency that
provides a database session per request.

Why this replaces the in-memory dict from Module 8:
    That dict lived only in process RAM — restarting the server lost
    every uploaded document's metadata (though the actual vector/BM25
    index was ALSO in-memory, so this alone wasn't the full fix; full
    persistence needs Weaviate too, covered later in this module).
    Postgres gives us the metadata layer (filenames, page/chunk counts,
    upload timestamps) real durability, proper querying, and the
    ability to run multiple backend server processes sharing the same
    data — none of which an in-process dict can do.

Which other files use this:
    models/document.py's ORM model inherits from Base defined here.
    api/routes/documents.py depends on get_db() for every request that
    touches document metadata.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

# pool_pre_ping=True: before handing out a connection from the pool,
# SQLAlchemy sends a cheap "is this connection still alive?" check
# first. Without this, a connection that's gone stale (e.g. the DB
# restarted, or a firewall/load balancer silently dropped an idle TCP
# connection — both genuinely common in real deployments) would fail
# with a confusing error on the FIRST real query that tries to use it.
# This trades a small amount of per-checkout latency for avoiding that
# entire class of intermittent, hard-to-reproduce connection errors.
engine = create_engine(settings.postgres_url, pool_pre_ping=True)

# sessionmaker returns a FACTORY, not a session itself — calling
# SessionLocal() creates a new session bound to `engine`. We create one
# NEW session per request (see get_db below) rather than sharing one
# session across the whole app, because SQLAlchemy sessions are NOT
# thread-safe / concurrent-request-safe — each request needs its own.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """
    The base class every ORM model inherits from. SQLAlchemy uses this
    to collect metadata about every table across the whole app (via
    Base.metadata), which is what lets us call
    Base.metadata.create_all(engine) once to create every table at
    once, rather than creating each table manually.
    """

    pass


def get_db():
    """
    A FastAPI dependency using the GENERATOR pattern: code before
    `yield` runs before the request handler executes; code after
    `yield` (in this case, none extra — just the implicit cleanup via
    `finally`) runs after the handler finishes, WHETHER IT SUCCEEDED OR
    RAISED AN EXCEPTION. This is FastAPI's standard idiom for "give
    every request its own resource, and guarantee cleanup no matter
    what" — directly analogous to a `with` context manager, but in a
    form FastAPI's dependency injection system knows how to use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
