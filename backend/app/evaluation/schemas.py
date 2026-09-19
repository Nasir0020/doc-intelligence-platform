"""
schemas.py (evaluation)
========================
Data models for the evaluation harness: the golden dataset format
(known-correct question/relevant-chunk pairs) and the report produced
by running the harness against it.

Which other files use this:
    eval_harness.py constructs EvaluationReport. test_queries.json is
    hand-written in this schema's shape (a list of GoldenExample).
"""

from pydantic import BaseModel


class GoldenExample(BaseModel):
    """
    One hand-labeled test case: a question, and the chunk_id(s) we
    (as humans, reading the source documents ourselves) have confirmed
    actually contain the answer.

    Why do we need to hand-label this at all, rather than just eyeball
    whether answers "look right"?
        This IS what makes evaluation objective and repeatable instead
        of subjective. "relevant_chunk_ids" is ground truth we define
        ONCE, by reading the source document ourselves — every metric
        below is then a mechanical comparison against that fixed
        reference, not a judgment call made fresh each time.
    """

    question: str
    relevant_chunk_ids: list[str]


class EvaluationReport(BaseModel):
    """Aggregate metrics across an entire golden dataset run."""

    num_examples: int
    k: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
