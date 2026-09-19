"""
dependencies.py
================
Shared FastAPI dependencies — objects that need to exist ONCE per
running server process and be reused across every request, rather than
being reconstructed on every single API call.

Why @lru_cache here, same rationale as config.get_settings() and the
embedder/reranker singletons already built in earlier modules:
    Constructing a HybridRetriever or Reranker involves loading models
    (or attempting to and falling back) — expensive, and in the neural
    case, something we want to happen exactly ONCE at server startup,
    not per-request.

KNOWN LIMITATION, stated plainly:
    HybridRetriever wraps an InMemoryVectorStore here — meaning every
    uploaded document lives only in this process's RAM. Restarting the
    server loses everything. This is the correct interim state for
    THIS module (no Docker/Weaviate/Postgres available yet) — Module 11
    swaps this for a real persistent stack. We are not pretending this
    is production-ready; documents don't survive a restart on purpose,
    documented rather than hidden.

Which other files use this:
    api/routes/documents.py and api/routes/query.py both depend on
    get_retriever(); query.py also depends on get_reranker().
"""

from functools import lru_cache

from app.db.vector_store import InMemoryVectorStore
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import HybridRetriever


@lru_cache
def get_retriever() -> HybridRetriever:
    return HybridRetriever(InMemoryVectorStore())


@lru_cache
def get_reranker() -> Reranker:
    return Reranker()
