"""Train and evaluate the TF-IDF + LogReg classical baseline.

Usage:
    python scripts/run_classical_baseline.py --config configs/baseline_classical.yaml
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.data.loader import load_gold_standard
from src.evaluation.metrics import evaluate, optimize_thresholds
from src.models.classical_baseline import TfidfLogRegBaseline
from src.pipeline.runner import load_config
from src.tracking.mlflow_tracker import MLflowTracker

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Train classical TF-IDF+LogReg baseline")
    parser.add_argument("--config", required=True, help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    experiment_config = config.get("experiment", {})

    console.print("[bold]Loading data splits...[/bold]")
    train_data = load_gold_standard(Path(data_config["splits_dir"]) / "train.jsonl")
    val_data = load_gold_standard(Path(data_config["splits_dir"]) / "val.jsonl")
    test_data = load_gold_standard(Path(data_config["splits_dir"]) / "test.jsonl")
    console.print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    tracker = MLflowTracker(
        tracking_uri=config.get("tracking", {}).get("mlflow", {}).get("tracking_uri", "file:./mlruns"),
    )
    tracker.set_experiment(experiment_config.get("mlflow_experiment", "parliament-baseline"))
    tracker.start_run(experiment_config.get("name", "baseline-tfidf-logreg"), config)

    console.print(f"[bold]Initializing model: {model_config['name']}[/bold]")
    ngram = model_config.get("ngram_range", [1, 2])
    model = TfidfLogRegBaseline(
        max_features=model_config.get("max_features", 20000),
        ngram_range=tuple(ngram),
        min_df=model_config.get("min_df", 2),
        C=model_config.get("C", 1.0),
        threshold=config.get("evaluation", {}).get("threshold", 0.5),
    )

    console.print("[bold]Training...[/bold]")
    fit_info = model.fit(train_data)
    console.print(f"  Features: {fit_info['n_features']}, Labels: {fit_info['n_labels']}")

    console.print("[bold]Optimizing thresholds on validation set...[/bold]")
    val_speeches = [s.speech for s in val_data]
    val_predictions = asyncio.run(model.classify_batch(val_speeches))
    val_gold = [s.gold for s in val_data if s.gold]

    optimal_thresholds = optimize_thresholds(val_predictions, val_gold)
    model.set_thresholds(optimal_thresholds)

    console.print("[bold]Evaluating on test set...[/bold]")
    test_speeches = [s.speech for s in test_data]
    test_predictions = asyncio.run(model.classify_batch(test_speeches))
    test_gold = [s.gold for s in test_data if s.gold]

    report = evaluate(
        test_predictions,
        test_gold,
        model_name=model.model_name,
    )

    tracker.log_evaluation(report)
    tracker.log_param("optimal_thresholds", json.dumps(optimal_thresholds))
    tracker.log_param("n_features", fit_info["n_features"])

    predictions_path = Path(data_config["predictions_dir"]) / f"{experiment_config.get('name', 'baseline-tfidf-logreg')}.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w", encoding="utf-8") as f:
        for pred in test_predictions:
            f.write(pred.model_dump_json() + "\n")
    tracker.log_artifact(str(predictions_path))

    tracker.end_run()

    table = Table(title=f"Results: {model.model_name}")
    table.add_column("Label", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Kappa", justify="right")
    table.add_column("Support", justify="right")

    for row in report.to_table_rows():
        table.add_row(
            row["label"], row["precision"], row["recall"],
            row["f1"], row["kappa"], row["support"],
        )
    console.print(table)
    console.print(f"\n[bold green]Macro-F1: {report.macro_f1:.4f}[/bold green]")
    console.print(f"Micro-F1: {report.micro_f1:.4f}")
    console.print(f"Hamming Loss: {report.hamming_loss:.4f}")


if __name__ == "__main__":
    main()
