"""
health.py
=========
A health check endpoint. This is the FIRST endpoint any production
service should have — it's what load balancers, Kubernetes, and Docker
Compose use to determine "is this service alive and ready to receive
traffic?" before routing real requests to it.

Why this file is needed:
    Without a health endpoint, a container orchestrator has no reliable
    way to know if your app crashed on startup vs. is just slow to boot.

Which other files use this:
    main.py imports and mounts this router. Later, docker-compose.yml's
    `healthcheck` directive will call this endpoint to decide whether
    the backend container is ready.
"""

from fastapi import APIRouter

from app.config import settings

# APIRouter groups related endpoints together. Later, documents.py and
# query.py will each define their own router, and main.py mounts all of
# them onto the main `app`. This is how large FastAPI apps stay organized
# instead of becoming one giant file with hundreds of endpoints.
router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
def health_check() -> dict:
    """
    Returns basic liveness information.

    Why return a plain dict instead of a Pydantic model here?
    For a genuinely production-grade endpoint, we WOULD define an
    explicit Pydantic response model (we'll do exactly that for every
    other endpoint from Module 8 onward). For a trivial health check with
    no meaningful risk of malformed data, FastAPI can serialize a plain
    dict to JSON directly, and this is a common, accepted exception to
    the "always use a response model" rule.
    """
    return {
        "status": "ok",
        "app_name": settings.APP_NAME,
        "environment": settings.APP_ENV,
    }
