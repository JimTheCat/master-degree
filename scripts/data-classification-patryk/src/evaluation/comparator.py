"""Compare experiment runs from MLflow."""

from dataclasses import dataclass

import pandas as pd

from src.schema.labels import ALL_LABELS


@dataclass
class ExperimentComparison:
    """Side-by-side comparison of multiple experiment runs."""
    summary_df: pd.DataFrame  # Rows = metrics, columns = experiment names
    per_label_df: pd.DataFrame  # Rows = labels, columns = experiment names (F1 values)
    cost_df: pd.DataFrame | None = None  # Cost comparison if available


def compare_mlflow_runs(
    tracking_uri: str,
    experiment_names: list[str] | None = None,
    run_ids: list[str] | None = None,
) -> ExperimentComparison:
    """Load runs from MLflow and build a comparison table.

    Either experiment_names or run_ids must be provided.
    """
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)

    runs_data = []

    if run_ids:
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            runs_data.append({
                "run_name": run.info.run_name or run_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
            })
    elif experiment_names:
        for exp_name in experiment_names:
            experiment = mlflow.get_experiment_by_name(exp_name)
            if experiment is None:
                continue
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            for _, row in runs.iterrows():
                metrics = {k: v for k, v in row.items() if k.startswith("metrics.")}
                metrics = {k.replace("metrics.", ""): v for k, v in metrics.items()}
                runs_data.append({
                    "run_name": row.get("tags.mlflow.runName", row["run_id"]),
                    "metrics": metrics,
                    "params": {
                        k.replace("params.", ""): v
                        for k, v in row.items()
                        if k.startswith("params.")
                    },
                })

    if not runs_data:
        return ExperimentComparison(
            summary_df=pd.DataFrame(),
            per_label_df=pd.DataFrame(),
        )

    # Build summary comparison
    summary_metrics = ["macro_f1", "micro_f1", "weighted_f1", "hamming_loss", "subset_accuracy"]
    summary_rows = {}
    for metric in summary_metrics:
        summary_rows[metric] = {
            rd["run_name"]: rd["metrics"].get(metric, None) for rd in runs_data
        }
    summary_df = pd.DataFrame(summary_rows).T

    # Build per-label F1 comparison
    per_label_rows = {}
    for label in ALL_LABELS:
        per_label_rows[label] = {
            rd["run_name"]: rd["metrics"].get(f"f1_{label}", None) for rd in runs_data
        }
    per_label_df = pd.DataFrame(per_label_rows).T

    # Cost comparison
    cost_metrics = ["total_cost_usd", "total_tokens", "avg_latency_ms"]
    cost_rows = {}
    has_cost = False
    for metric in cost_metrics:
        row = {rd["run_name"]: rd["metrics"].get(metric, None) for rd in runs_data}
        if any(v is not None for v in row.values()):
            has_cost = True
            cost_rows[metric] = row
    cost_df = pd.DataFrame(cost_rows).T if has_cost else None

    return ExperimentComparison(
        summary_df=summary_df,
        per_label_df=per_label_df,
        cost_df=cost_df,
    )
