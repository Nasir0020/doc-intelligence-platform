"""
test_eval_harness.py
=====================
Unit tests for eval_harness.py's IR metrics (Recall@K, Precision@K,
MRR), using a controlled HybridRetriever with known, hand-crafted
relevance so each metric's expected value can be computed by hand and
checked exactly — the same principle as test_retriever.py.
"""

import numpy as np

from app.db.vector_store import InMemoryVectorStore
from app.evaluation.eval_harness import evaluate_retrieval
from app.evaluation.schemas import GoldenExample
from app.ingestion.embedder import Embedder
from app.retrieval.retriever import HybridRetriever


class _FakeEmbedder:
    """
    A minimal stand-in for Embedder that returns fixed, known vectors
    per query string rather than loading any real model. This keeps
    the test fast, deterministic, and independent of which embedding
    backend (neural or fallback) happens to be available in whatever
    environment the suite runs in.
    """

    def __init__(self, query_vectors: dict[str, np.ndarray]):
        self._query_vectors = query_vectors

    def embed_query(self, text: str) -> np.ndarray:
        return self._query_vectors[text]


def test_perfect_retrieval_yields_perfect_scores():
    retriever = HybridRetriever(InMemoryVectorStore())
    retriever.add_chunks(
        chunk_ids=["relevant_chunk", "irrelevant_chunk"],
        vectors=np.array([[1.0, 0.0], [0.0, 1.0]]),
        texts=["apac growth figures", "unrelated support tickets"],
        page_numbers=[[1], [2]],
    )

    embedder = _FakeEmbedder({"apac growth?": np.array([1.0, 0.0])})
    golden_set = [GoldenExample(question="apac growth?", relevant_chunk_ids=["relevant_chunk"])]

    report = evaluate_retrieval(retriever, embedder, golden_set, k=1)  # type: ignore[arg-type]

    assert report.recall_at_k == 1.0
    assert report.precision_at_k == 1.0
    assert report.mrr == 1.0


def test_complete_miss_yields_zero_scores():
    retriever = HybridRetriever(InMemoryVectorStore())
    retriever.add_chunks(
        chunk_ids=["some_chunk"],
        vectors=np.array([[1.0, 0.0]]),
        texts=["unrelated content"],
        page_numbers=[[1]],
    )

    embedder = _FakeEmbedder({"question?": np.array([1.0, 0.0])})
    # relevant_chunk_ids references a chunk that was NEVER indexed —
    # simulating a golden example whose true answer retrieval can never
    # find, which should score zero on every metric, not error out.
    golden_set = [
        GoldenExample(question="question?", relevant_chunk_ids=["nonexistent_chunk"])
    ]

    report = evaluate_retrieval(retriever, embedder, golden_set, k=1)  # type: ignore[arg-type]

    assert report.recall_at_k == 0.0
    assert report.precision_at_k == 0.0
    assert report.mrr == 0.0


def test_mrr_rewards_earlier_rank():
    """
    Two golden examples: one where the relevant chunk is ranked first,
    one where it's ranked second. MRR should average 1/1 and 1/2.
    """
    retriever = HybridRetriever(InMemoryVectorStore())
    retriever.add_chunks(
        chunk_ids=["a", "b"],
        vectors=np.array([[1.0, 0.0], [0.9, 0.1]]),
        texts=["exact match content", "close but different content"],
        page_numbers=[[1], [2]],
    )

    embedder = _FakeEmbedder({"q1": np.array([1.0, 0.0]), "q2": np.array([0.9, 0.1])})
    golden_set = [
        GoldenExample(question="q1", relevant_chunk_ids=["a"]),
        GoldenExample(question="q2", relevant_chunk_ids=["a"]),  # "a" ranks 2nd for this query
    ]

    report = evaluate_retrieval(retriever, embedder, golden_set, k=2)  # type: ignore[arg-type]

    # (1/1 + 1/2) / 2 = 0.75
    assert abs(report.mrr - 0.75) < 0.01
