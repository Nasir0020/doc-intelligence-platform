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

from fastapi import FastAPI

from app.api.routes import health
from app.config import settings

# Creating the FastAPI application instance.
# `title` and `debug` feed directly into the auto-generated OpenAPI docs
# available at /docs when the server is running.
app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG,
)

# `include_router` mounts all endpoints defined in health.py under this
# app. Routers are how FastAPI apps stay modular as they grow — each
# subsystem (documents, query, health) gets its own router file instead
# of every endpoint being defined directly in main.py.
app.include_router(health.router)
