"""
retriever.py
============
Combines keyword search (BM25) and vector similarity search into a
single ranked result list using Reciprocal Rank Fusion (RRF).

Why we need to FUSE two ranked lists rather than average raw scores:
    BM25 scores and cosine similarity scores live on completely
    different, incomparable numeric scales. BM25 scores are unbounded
    positive numbers that depend on corpus statistics (term rarity,
    document length) — a score of 8.3 might be excellent or mediocre
    depending on the corpus. Cosine similarity is always bounded
    between -1 and 1. Naively averaging "0.87" (cosine) with "8.3"
    (BM25) would let BM25 dominate purely because its numbers are
    numerically larger — not because it's more reliable. RRF sidesteps
    this entirely by working with RANKS (1st place, 2nd place, ...)
    instead of raw scores, since ranks are always directly comparable
    regardless of the underlying scoring scale.

Which other files use this:
    api/routes/query.py (Module 8) will call HybridRetriever.search()
    for every user question.
"""

import logging

import numpy as np
from rank_bm25 import BM25Okapi

from app.db.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

# The RRF constant. 60 is the value used in the original Cormack et al.
# (2009) paper that introduced RRF, and has since become a de facto
# standard default across the industry — we're not required to use 60,
# but deviating without a specific measured reason would just be
# unjustified fiddling.
RRF_K = 60


import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """
    Lowercases, extracts word tokens (stripping punctuation), and
    removes English stopwords.

    Why strip stopwords specifically:
        Found via real testing on a small corpus — BM25's IDF term
        naturally down-weights words that appear in EVERY document, but
        with very few documents, that statistic is noisy: a stopword
        like "what" or "the" can still accumulate enough raw term-
        frequency score across a tiny corpus to meaningfully distort
        rankings toward irrelevant matches. Explicitly removing
        stopwords (using scikit-learn's built-in list — no network
        download required, unlike NLTK's stopword corpus) is the
        standard, direct fix, rather than only relying on IDF alone to
        handle it statistically.

    We still don't stem/lemmatize (documented limitation, same as
    before) — this keeps the tokenizer dependency-free and simple while
    still meaningfully improving on stopword noise, which was the
    concrete, observed problem.
    """
    words = _WORD_PATTERN.findall(text.lower())
    return [w for w in words if w not in ENGLISH_STOP_WORDS]


class HybridRetriever:
    """
    Owns both a vector store and a BM25 keyword index over the SAME
    corpus of chunks, and fuses results from both when searching.

    Why does this class rebuild the entire BM25 index on every add_chunks
    call, rather than updating incrementally?
        rank_bm25's BM25Okapi is built from a static corpus at
        construction time — it has no incremental-update API. Rebuilding
        on every batch add is O(n) in corpus size, which is fine for our
        batch-oriented ingestion pattern (documents are ingested in
        discrete batches, not one chunk trickling in per second), but
        would NOT scale to high-frequency incremental updates. A
        production system with very frequent updates would instead use
        a dedicated search engine with real incremental indexing
        (Elasticsearch, or Weaviate's own built-in BM25/hybrid support —
        recall from Module 1 that Weaviate supports hybrid search
        natively, which is a real alternative to this hand-built fusion
        approach). We build our own fusion here specifically so the
        mechanics are visible and testable, and so it works identically
        against EITHER vector store backend (InMemory or Weaviate).
    """

    def __init__(self, vector_store: VectorStore):
        self._vector_store = vector_store
        self._bm25: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._texts: list[str] = []
        self._page_numbers: list[list[int]] = []

    @property
    def is_empty(self) -> bool:
        return len(self._chunk_ids) == 0

    def add_chunks(
        self,
        chunk_ids: list[str],
        vectors: np.ndarray,
        texts: list[str],
        page_numbers: list[list[int]],
    ) -> None:
        self._vector_store.add(chunk_ids, vectors, texts, page_numbers)

        self._chunk_ids.extend(chunk_ids)
        self._texts.extend(texts)
        self._page_numbers.extend(page_numbers)

        tokenized_corpus = [_tokenize(t) for t in self._texts]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index rebuilt over %d total chunks.", len(self._texts))

    def _bm25_search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Returns [(chunk_id, bm25_score), ...] for the top_k BM25 matches."""
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._chunk_ids[i], float(scores[i])) for i in top_indices]

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int = 5,
        candidate_pool_size: int = 20,
    ) -> list[SearchResult]:
        """
        Runs BOTH search methods, fuses their rankings via RRF, and
        returns the final top_k results.

        candidate_pool_size controls how many results EACH method
        contributes to the fusion pool before final re-ranking — this
        is deliberately larger than top_k, so a chunk that (say) BM25
        ranks 15th but vector search ranks 1st still has a chance to
        surface near the top after fusion, rather than being excluded
        just because it wasn't in BM25's very top few results.
        """
        vector_results = self._vector_store.search(query_vector, top_k=candidate_pool_size)
        bm25_results = self._bm25_search(query, top_k=candidate_pool_size)

        # Reciprocal Rank Fusion:
        #   rrf_score(chunk) = sum over each method, of 1 / (RRF_K + rank)
        #   where `rank` is that chunk's 1-indexed position in that
        #   method's ranked list (a chunk absent from a given method's
        #   list simply contributes 0 from that method — it isn't
        #   penalized further, just not boosted by that method).
        rrf_scores: dict[str, float] = {}

        for rank, result in enumerate(vector_results, start=1):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0.0) + 1.0 / (
                RRF_K + rank
            )

        for rank, (chunk_id, _score) in enumerate(bm25_results, start=1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)

        # We need each chunk's text/page_numbers to build the final
        # SearchResult objects. vector_results already carries that
        # metadata for chunks it found; for chunks BM25 found but
        # vector search didn't include in its pool, we fall back to
        # our own stored corpus lookup by index.
        metadata_by_id = {r.chunk_id: r for r in vector_results}
        chunk_id_to_index = {cid: i for i, cid in enumerate(self._chunk_ids)}

        fused_results: list[SearchResult] = []
        for chunk_id, rrf_score in sorted(
            rrf_scores.items(), key=lambda kv: kv[1], reverse=True
        )[:top_k]:
            if chunk_id in metadata_by_id:
                base = metadata_by_id[chunk_id]
                text, pages = base.text, base.page_numbers
            else:
                idx = chunk_id_to_index[chunk_id]
                text, pages = self._texts[idx], self._page_numbers[idx]

            fused_results.append(
                SearchResult(chunk_id=chunk_id, score=rrf_score, text=text, page_numbers=pages)
            )

        return fused_results
