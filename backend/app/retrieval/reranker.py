"""
reranker.py
===========
Re-scores a small pool of retrieval candidates using a cross-encoder
for much higher precision than the bi-encoder/BM25 hybrid search alone
can achieve — see the module-level discussion in Module 6 notes for why
cross-encoders can't be used for the FIRST-stage search over an entire
corpus, only this second, narrow-pool re-scoring stage.

Production backend:
    cross-encoder/ms-marco-MiniLM-L-6-v2 via sentence-transformers'
    CrossEncoder class — a well-established reranking model trained
    specifically on query-passage relevance (MS MARCO dataset).

Fallback backend (same network constraint as Modules 3-4 — no route to
huggingface.co in this sandbox):
    A lexical overlap score (weighted Jaccard-style overlap between
    query tokens and candidate tokens). This has NONE of a real
    cross-encoder's contextual understanding — it cannot tell that
    "the company grew" and "the company did not grow" are opposite in
    meaning despite heavy word overlap. This is a materially weaker
    approximation than the fallbacks in earlier modules, and we say so
    plainly rather than dressing it up.

Which other files use this:
    api/routes/query.py (Module 8) calls rerank() on the HybridRetriever's
    output before passing the top few results to the LLM (Module 7).
"""

import logging
import math

from app.db.vector_store import SearchResult
from app.retrieval.retriever import _tokenize

logger = logging.getLogger(__name__)

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self):
        self._model = self._try_load_cross_encoder()
        self.backend_name = "cross_encoder" if self._model else "lexical_overlap_fallback"

    @staticmethod
    def _try_load_cross_encoder():
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
            logger.info("Loaded cross-encoder '%s'.", CROSS_ENCODER_MODEL_NAME)
            return model
        except Exception as exc:
            logger.warning(
                "Cross-encoder unavailable (%s: %s) — falling back to a lexical "
                "overlap heuristic. This fallback has NO contextual/semantic "
                "understanding (e.g. cannot distinguish negation) and is "
                "materially weaker than a real cross-encoder — use only for "
                "keeping the pipeline runnable, not as a quality benchmark.",
                type(exc).__name__,
                str(exc)[:150],
            )
            return None

    def _lexical_overlap_score(self, query: str, text: str) -> float:
        """
        Jaccard-style overlap: |shared tokens| / |query tokens|.

        We deliberately divide by the QUERY's token count (not the
        union, as classic Jaccard would) rather than a symmetric
        Jaccard index, because a long document sharing all of a short
        query's tokens should score highly regardless of how much OTHER
        unrelated content that document also contains — a symmetric
        Jaccard index would unfairly punish long documents just for
        being long, which isn't the property we actually want from a
        relevance-style fallback score.
        """
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return 0.0
        text_tokens = set(_tokenize(text))
        overlap = query_tokens & text_tokens  # set intersection
        return len(overlap) / len(query_tokens)

    def rerank(
        self, query: str, candidates: list[SearchResult], top_k: int = 5
    ) -> list[SearchResult]:
        """
        Re-scores every candidate against the query and returns the
        top_k, re-sorted by the (new) rerank score. The ORIGINAL
        retrieval score (from RRF fusion) is discarded here — the
        reranker's score is a strictly more precise judgment intended
        to replace it, not blend with it, since the reranker sees far
        more of the query/document interaction than the fusion score did.
        """
        if not candidates:
            return []

        if self._model is not None:
            pairs = [(query, c.text) for c in candidates]
            raw_scores = self._model.predict(pairs)
            # Cross-encoders trained on MS MARCO-style data typically
            # output raw, UNBOUNDED logits, not probabilities. Applying
            # a sigmoid squashes them into a (0, 1) range that's more
            # interpretable when we surface scores in the API later —
            # this doesn't change the RANKING (sigmoid is monotonic:
            # preserves relative order) but makes absolute values
            # meaningful as an approximate "relevance confidence."
            scores = [1 / (1 + math.exp(-s)) for s in raw_scores]
        else:
            scores = [self._lexical_overlap_score(query, c.text) for c in candidates]

        rescored = [
            SearchResult(
                chunk_id=c.chunk_id, score=score, text=c.text, page_numbers=c.page_numbers
            )
            for c, score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]
