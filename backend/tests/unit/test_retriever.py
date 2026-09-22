"""
test_retriever.py
==================
Unit tests for the RRF fusion logic in retriever.py, using a real
InMemoryVectorStore (fast, no external dependencies) but controlled,
hand-crafted embeddings so the exact expected ranking is known in
advance — rather than relying on a real embedding model's fuzzy
semantic judgment, which would make the expected result harder to
state precisely in a test.
"""

import numpy as np

from app.db.vector_store import InMemoryVectorStore
from app.retrieval.retriever import HybridRetriever, _tokenize


def test_tokenizer_strips_stopwords_and_punctuation():
    """
    Regression test for the real bug found in Module 5: stopwords were
    distorting BM25 rankings on small corpora.
    """
    tokens = _tokenize("What was the APAC growth percentage?")
    assert "the" not in tokens
    assert "was" not in tokens
    assert "apac" in tokens
    assert "growth" in tokens
    assert "percentage" in tokens


def test_hybrid_search_returns_top_k_results():
    retriever = HybridRetriever(InMemoryVectorStore())
    # Hand-crafted 2D vectors: chunk A points the same direction as the
    # query vector we'll search with; chunk B points a very different
    # direction. This makes the expected ranking a known fact, not a
    # guess about what a real embedding model would produce.
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]])
    retriever.add_chunks(
        chunk_ids=["chunk_a", "chunk_b", "chunk_c"],
        vectors=vectors,
        texts=["apac revenue growth", "unrelated engineering ticket volume", "apac growth figures"],
        page_numbers=[[1], [2], [3]],
    )

    query_vector = np.array([1.0, 0.0])
    results = retriever.search("apac growth", query_vector, top_k=2)

    assert len(results) == 2
    result_ids = {r.chunk_id for r in results}
    # chunk_b is both vector-dissimilar (orthogonal) AND lexically
    # unrelated -- it should not make the top 2.
    assert "chunk_b" not in result_ids


def test_empty_retriever_returns_no_results():
    retriever = HybridRetriever(InMemoryVectorStore())
    assert retriever.is_empty is True

    results = retriever.search("anything", np.array([1.0, 0.0]), top_k=5)
    assert results == []


def test_is_empty_becomes_false_after_adding_chunks():
    retriever = HybridRetriever(InMemoryVectorStore())
    assert retriever.is_empty is True

    retriever.add_chunks(
        chunk_ids=["chunk_a"],
        vectors=np.array([[1.0, 0.0]]),
        texts=["some content"],
        page_numbers=[[1]],
    )
    assert retriever.is_empty is False
