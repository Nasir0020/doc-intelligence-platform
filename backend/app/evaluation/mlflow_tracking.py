"""
mlflow_tracking.py
===================
Logs evaluation harness runs (Module 9) to MLflow, so retrieval-quality
metrics across different configurations (chunk size, embedding backend,
reranking on/off, etc.) are tracked over time and comparable — instead
of scattered print() output from ad-hoc test runs that vanish the
moment the terminal closes.

Why this matters concretely: without this, "I improved chunking" is an
unverifiable claim. WITH this, you can pull up an MLflow run comparison
showing Recall@5 going from 0.71 to 0.89 across two named, dated,
parameter-tagged experiment runs — a real, defensible artifact for an
interview or a PR description.

Which other files use this:
    A future CI step, or a manual experimentation script, calls
    log_evaluation_run() after each evaluate_retrieval() call from
    eval_harness.py.
"""

import mlflow

from app.evaluation.schemas import EvaluationReport

EXPERIMENT_NAME = "retrieval-quality"


def log_evaluation_run(
    run_name: str,
    params: dict[str, str | int | float],
    report: EvaluationReport,
    tracking_uri: str = "file:./mlruns",
) -> str:
    """
    Logs one evaluation run's configuration (params) and results
    (metrics) to MLflow. Returns the MLflow run ID.

    Why separate PARAMS from METRICS (MLflow's own core distinction):
        Params are INPUTS you chose (chunk size, embedding backend
        name, k) — they don't change during the run. Metrics are
        OUTPUTS you measured (recall, precision, mrr) — the results of
        running with those params. Keeping this distinction lets MLflow's
        UI let you group/filter runs by param values and plot metrics
        across them, which is the entire point of tracking experiments
        instead of just printing numbers once and discarding them.
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(
            {
                "recall_at_k": report.recall_at_k,
                "precision_at_k": report.precision_at_k,
                "mrr": report.mrr,
                "num_examples": report.num_examples,
            }
        )
        return run.info.run_id
