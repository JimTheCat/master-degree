from pathlib import Path
from typing import Any

import mlflow

from src.evaluation.metrics import EvaluationReport


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, str]:
    """Flatten a nested dict for MLflow param logging."""
    items = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, prefix=f"{key}."))
        else:
            items[key] = str(v)
    return items


class MLflowTracker:
    """Wrapper around MLflow for experiment tracking."""

    def __init__(self, tracking_uri: str = "file:./mlruns"):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self._run = None

    def set_experiment(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str, config: dict | None = None) -> str:
        """Start an MLflow run and log config as params."""
        self._run = mlflow.start_run(run_name=run_name)
        if config:
            # MLflow params are limited to 500 chars, so flatten and truncate
            flat = _flatten_dict(config)
            for k, v in flat.items():
                mlflow.log_param(k, str(v)[:500])
        return self._run.info.run_id

    def log_metric(self, key: str, value: float, step: int | None = None):
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float]):
        mlflow.log_metrics(metrics)

    def log_evaluation(self, report: EvaluationReport):
        """Log all metrics from an evaluation report."""
        metrics_dict = report.to_dict()
        # MLflow metrics must be numeric
        numeric = {k: v for k, v in metrics_dict.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(numeric)

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def log_param(self, key: str, value: Any):
        mlflow.log_param(key, str(value)[:500])

    def end_run(self):
        if self._run:
            mlflow.end_run()
            self._run = None

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run else None
