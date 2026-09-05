"""Unified tracking facade coordinating MLflow and LangSmith."""

from src.evaluation.metrics import EvaluationReport
from src.tracking.langsmith_tracker import LangSmithTracker
from src.tracking.mlflow_tracker import MLflowTracker


class UnifiedTracker:
    """Facade that coordinates MLflow and LangSmith tracking."""

    def __init__(self, config: dict):
        tracking_config = config.get("tracking", {})

        # MLflow setup
        mlflow_config = tracking_config.get("mlflow", {})
        self.mlflow = MLflowTracker(
            tracking_uri=mlflow_config.get("tracking_uri", "file:./mlruns"),
        )

        # LangSmith setup
        ls_config = tracking_config.get("langsmith", {})
        self.langsmith = LangSmithTracker(
            project_name=ls_config.get("project", "parliament-speech-analysis"),
            trace_sample_rate=ls_config.get("trace_sample_rate", 1.0),
        )

        self._experiment_name = None

    def start_experiment(self, experiment_name: str, config: dict) -> str:
        """Start tracking an experiment in both systems."""
        self._experiment_name = experiment_name

        # MLflow
        mlflow_experiment = config.get("experiment", {}).get(
            "mlflow_experiment", experiment_name
        )
        self.mlflow.set_experiment(mlflow_experiment)
        run_id = self.mlflow.start_run(
            run_name=experiment_name,
            config=config,
        )

        return run_id

    def log_evaluation(self, report: EvaluationReport):
        """Log evaluation metrics to MLflow."""
        self.mlflow.log_evaluation(report)

    def log_metric(self, key: str, value: float, step: int | None = None):
        self.mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: dict[str, float]):
        self.mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: str):
        self.mlflow.log_artifact(local_path)

    def should_trace_llm_call(self) -> bool:
        """Check if this LLM call should be traced in LangSmith."""
        return self.langsmith.should_trace()

    def end_experiment(self):
        self.mlflow.end_run()
