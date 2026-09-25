"""
query.py
========
The question-answering endpoint: takes a user's question, runs it
through retrieval -> reranking -> generation, and returns a grounded,
cited answer.

Which other files use this:
    main.py mounts this router. The frontend (Module 10) will call
    POST /query whenever the user submits a question in the chat UI.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.dependencies import get_reranker, get_retriever
from app.ingestion.embedder import get_embedder
from app.retrieval.generator import generate_answer
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import HybridRetriever
from app.retrieval.schemas import GeneratedAnswer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

# How many candidates the FIRST-stage hybrid search pulls before
# reranking narrows them down. This is deliberately larger than the
# final top_k a user requests — recall Module 6's rationale: reranking
# needs a wide-enough pool to have a chance of finding the true best
# answer, not just re-sorting whatever the fast stage already narrowed
# down too aggressively.
RETRIEVAL_CANDIDATE_POOL_SIZE = 20


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many source chunks to use for the final generated answer.",
    )


@router.post("", response_model=GeneratedAnswer)
def ask_question(
    request: QueryRequest,
    retriever: HybridRetriever = Depends(get_retriever),
    reranker: Reranker = Depends(get_reranker),
) -> GeneratedAnswer:
    # BUG FOUND VIA TESTING (Module 8): calling embed_query() before
    # ANY document has ever been indexed crashes the TF-IDF/SVD fallback
    # embedder (it has nothing to have been fit on yet) with an
    # unhandled RuntimeError, which surfaces as an opaque 500 instead of
    # a clean, expected 404. Checking is_empty FIRST avoids ever calling
    # embed_query in that state at all — this is also simply more
    # correct regardless of embedder backend: there's no reason to
    # embed a query against an empty corpus.
    if retriever.is_empty:
        raise HTTPException(
            status_code=404,
            detail="No documents have been indexed yet. Upload a document first.",
        )

    embedder = get_embedder()
    query_vector = embedder.embed_query(request.question)

    candidates = retriever.search(
        request.question,
        query_vector,
        top_k=RETRIEVAL_CANDIDATE_POOL_SIZE,
        candidate_pool_size=RETRIEVAL_CANDIDATE_POOL_SIZE,
    )

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="No relevant results found for this question.",
        )

    reranked = reranker.rerank(request.question, candidates, top_k=request.top_k)

    try:
        return generate_answer(request.question, reranked)
    except Exception as exc:
        # 502 Bad Gateway specifically (not 500 Internal Server Error):
        # the failure originates from an UPSTREAM dependency (the
        # # the upstream LLM call inside generate_answer), not from a bug in
        # our own server logic. Distinguishing "our code broke" (500)
        # from "a service we depend on broke" (502) is genuinely useful
        # signal for whoever is debugging or monitoring this in
        # production — it tells them where to look first.
        logger.error("LLM generation failed: %s", exc)
        raise HTTPException(
            status_code=502, detail=f"Answer generation failed: {exc}"
        ) from exc
