"""
main.py
=======
The application entrypoint. This is the file uvicorn points at to start
the server (e.g. `uvicorn app.main:app`).

Why this file is deliberately thin:
    main.py should only be responsible for CREATING and WIRING TOGETHER
    the app — mounting routers, registering middleware/exception handlers.
    It should contain almost no actual business logic. That logic lives
    in api/routes/*.py and the subsystem modules (ingestion/, retrieval/).
    This separation means main.py rarely needs to change as the app
    grows, which keeps merge conflicts and regressions rare in this file.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import documents, health, query
from app.config import settings
from app.db.postgres import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI's LIFESPAN pattern: an async context manager that runs
    startup code before `yield` and shutdown code after it, wrapping
    the entire time the server is running. This replaces the older
    `@app.on_event("startup")` decorator style (deprecated as of recent
    FastAPI versions) with a single, more explicit block.

    Base.metadata.create_all(engine) here creates every table (just
    `documents` so far) if it doesn't already exist — convenient for
    local development, but NOT how schema changes should be managed in
    a real production system: create_all() has no concept of
    MIGRATIONS. If we later add a column to DocumentRecord, create_all()
    won't alter the existing table to add it — it only creates tables
    that don't exist yet. A real project uses a migration tool (Alembic
    is the standard choice for SQLAlchemy) to version and apply schema
    changes safely. We're using create_all() here to keep this module
    focused on the wiring itself; adding Alembic is a documented next
    step, not implemented in this pass.
    """
    Base.metadata.create_all(bind=engine)
    yield


# Creating the FastAPI application instance.
# `title` and `debug` feed directly into the auto-generated OpenAPI docs
# available at /docs when the server is running.
app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# `include_router` mounts all endpoints defined in each router file
# under this app. Routers are how FastAPI apps stay modular as they
# grow — each subsystem (documents, query, health) gets its own router
# file instead of every endpoint being defined directly in main.py.
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
