"""Run an LLM classification experiment.

Usage:
    python scripts/run_llm_experiment.py --config configs/prompt_zeroshot.yaml
    python scripts/run_llm_experiment.py --config configs/model_claude.yaml

    # Full gold standard (500 speeches) instead of the test split:
    python scripts/run_llm_experiment.py --config configs/model_bielik.yaml --split all

    # Quick sanity run: 5 speeches, no MLflow logging:
    python scripts/run_llm_experiment.py --config configs/model_bielik.yaml --limit 5 --no-track

    # Continue an interrupted run (skips speeches already in the predictions file):
    python scripts/run_llm_experiment.py --config configs/model_bielik.yaml --split all --resume
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.data.loader import load_gold_standard
from src.evaluation.metrics import evaluate
from src.pipeline.runner import create_classifier, load_config
from src.schema.models import Prediction
from src.tracking.unified_tracker import UnifiedTracker

# record=True keeps everything printed so it can be dumped to results/logs/.
console = Console(record=True)

LOG_PATH = Path("results/logs/experiments.log")


def save_console_log(header: str) -> Path:
    """Append everything printed so far to the shared experiment log and clear
    the record buffer (so back-to-back sweep cells don't duplicate output)."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = console.export_text(clear=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 72}\n[{stamp}] {header}\n{'=' * 72}\n{text}\n")
    return LOG_PATH


def _load_eval_data(data_config: dict, split: str):
    if split == "all":
        return load_gold_standard(Path(data_config["gold_standard_path"]))
    return load_gold_standard(Path(data_config["splits_dir"]) / f"{split}.jsonl")


def _load_existing_predictions(path: Path) -> dict[str, Prediction]:
    """Load predictions from a previous (possibly interrupted) run, keyed by speech_id.

    Failed items (ERROR / PARSE_FAILED) are dropped so --resume retries them.
    """
    if not path.exists():
        return {}
    existing: dict[str, Prediction] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pred = Prediction.model_validate(json.loads(line))
            raw = pred.raw_response or ""
            if raw.startswith(("ERROR:", "PARSE_FAILED")):
                continue
            existing[pred.speech_id] = pred
    return existing


async def run_experiment(
    config: dict,
    split: str = "test",
    limit: int | None = None,
    resume: bool = False,
    track: bool = True,
) -> dict:
    """Run one experiment; returns a summary dict (for sweep aggregation).

    Everything printed during the run is appended to results/logs/experiments.log,
    also when the run fails partway.
    """
    # Experiment name: keep historical names for the default test split.
    exp_name = config.get("experiment", {}).get("name", "experiment")
    if split != "test":
        exp_name = f"{exp_name}-{split}"

    try:
        return await _run_experiment(config, exp_name, split, limit, resume, track)
    finally:
        log_path = save_console_log(exp_name)
        print(f"Console log appended to: {log_path}")


async def _run_experiment(
    config: dict,
    exp_name: str,
    split: str,
    limit: int | None,
    resume: bool,
    track: bool,
) -> dict:
    data_config = config["data"]

    # Load data
    console.print("[bold]Loading data...[/bold]")
    train_data = load_gold_standard(Path(data_config["splits_dir"]) / "train.jsonl")
    eval_data = _load_eval_data(data_config, split)
    if limit:
        eval_data = eval_data[:limit]
    console.print(f"  Train (for few-shot): {len(train_data)}, Eval ({split}): {len(eval_data)}")

    # --limit runs are sanity checks: write them to a separate file so they
    # never overwrite the canonical predictions of a full run.
    file_name = f"{exp_name}-limit{limit}.jsonl" if limit else f"{exp_name}.jsonl"
    predictions_path = Path(data_config["predictions_dir"]) / file_name

    existing: dict[str, Prediction] = {}
    if resume:
        existing = _load_existing_predictions(predictions_path)
        console.print(f"  Resume: {len(existing)} predictions already done")

    # Initialize tracker
    tracker = None
    if track:
        tracker = UnifiedTracker(config)
        run_id = tracker.start_experiment(exp_name, config)
        console.print(f"  MLflow run ID: {run_id}")

    # Create classifier
    classifier = create_classifier(config, train_data=train_data)
    console.print(f"  Model: {classifier.model_name}")
    console.print(f"  Prompt variant: {config.get('prompt', {}).get('variant', 'N/A')}")

    # Run classification (skipping already-done speeches on --resume)
    todo = [s.speech for s in eval_data if s.speech.speech_id not in existing]
    console.print(f"\n[bold]Classifying {len(todo)} speeches...[/bold]")
    new_predictions = await classifier.classify_batch(todo) if todo else []

    by_id = dict(existing)
    for pred in new_predictions:
        by_id[pred.speech_id] = pred
    # Keep eval_data order.
    predictions = [by_id[s.speech.speech_id] for s in eval_data if s.speech.speech_id in by_id]

    # Report failures so a partial run is visible instead of silently scoring zeros.
    failed = [
        p for p in predictions
        if (p.raw_response or "").startswith(("ERROR:", "PARSE_FAILED"))
    ]
    if failed:
        console.print(
            f"[bold yellow]WARN {len(failed)}/{len(predictions)} speeches failed "
            f"(ERROR/PARSE_FAILED) — rerun with --resume to retry them[/bold yellow]"
        )

    # Evaluate (aligned by index: only entries that have both gold and a prediction)
    aligned = [
        (by_id[s.speech.speech_id], s.gold)
        for s in eval_data
        if s.gold and s.speech.speech_id in by_id
    ]
    report = evaluate(
        [p for p, _ in aligned],
        [g for _, g in aligned],
        model_name=classifier.model_name,
        prompt_variant=config.get("prompt", {}).get("variant", ""),
    )

    # Calculate cost
    total_input_tokens = sum(p.token_usage.get("prompt_tokens", 0) for p in predictions)
    total_output_tokens = sum(p.token_usage.get("completion_tokens", 0) for p in predictions)
    avg_latency = sum(p.latency_ms or 0 for p in predictions) / max(len(predictions), 1)

    if tracker:
        tracker.log_evaluation(report)
        tracker.log_metrics({
            "total_input_tokens": float(total_input_tokens),
            "total_output_tokens": float(total_output_tokens),
            "avg_latency_ms": avg_latency,
            "failed_predictions": float(len(failed)),
        })

    # Save predictions
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred.model_dump_json() + "\n")
    if tracker:
        tracker.log_artifact(str(predictions_path))
        tracker.end_experiment()

    # Print results
    variant = config.get("prompt", {}).get("variant", "N/A")
    table = Table(title=f"Results: {classifier.model_name} / {variant} / split={split}")
    table.add_column("Label", style="cyan")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("Kappa", justify="right")

    for row in report.to_table_rows():
        table.add_row(row["label"], row["precision"], row["recall"], row["f1"], row["kappa"])

    console.print(table)
    console.print(f"\n[bold green]Macro-F1: {report.macro_f1:.4f}[/bold green]")
    console.print(f"Micro-F1: {report.micro_f1:.4f}")
    console.print(f"Total tokens: {total_input_tokens + total_output_tokens:,}")
    console.print(f"Avg latency: {avg_latency:.0f}ms")

    return {
        "experiment": exp_name,
        "model": classifier.model_name,
        "variant": variant,
        "split": split,
        "n_speeches": len(predictions),
        "n_failed": len(failed),
        "macro_f1": report.macro_f1,
        "micro_f1": report.micro_f1,
        "emotions_macro_f1": report.emotions_macro_f1,
        "rhetorical_macro_f1": report.rhetorical_macro_f1,
        "total_tokens": total_input_tokens + total_output_tokens,
        "avg_latency_ms": round(avg_latency, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Run LLM experiment")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val", "train", "all"],
        help="Which dataset to classify: a split or 'all' = full 500-speech gold standard",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Classify only the first N speeches"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip speeches already present in the predictions file; retry failed ones",
    )
    parser.add_argument("--no-track", action="store_true", help="Skip MLflow/LangSmith logging")
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    asyncio.run(run_experiment(
        config,
        split=args.split,
        limit=args.limit,
        resume=args.resume,
        track=not args.no_track,
    ))


if __name__ == "__main__":
    main()
