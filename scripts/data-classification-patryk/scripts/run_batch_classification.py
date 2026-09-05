"""Run classification on the full corpus with checkpointing and cost control.

Usage:
    python scripts/run_batch_classification.py --config configs/batch_processing.yaml --model-config configs/model_gpt.yaml
    python scripts/run_batch_classification.py --config configs/batch_processing.yaml --model-config configs/model_gpt.yaml --dry-run
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich.console import Console

from src.data.loader import load_corpus, load_gold_standard
from src.pipeline.batch_processor import BatchProcessor
from src.pipeline.cost_manager import estimate_cost
from src.pipeline.runner import create_classifier, load_config
from src.tracking.unified_tracker import UnifiedTracker

console = Console()


async def run_batch(batch_config: dict, model_config: dict, dry_run: bool = False):
    data_config = model_config["data"]
    experiment_config = batch_config.get("experiment", {})
    batch_settings = batch_config.get("batch_processing", {})

    # Load corpus
    console.print("[bold]Loading corpus...[/bold]")
    corpus_path = batch_settings.get("input_path") or data_config["corpus_path"]
    speeches = list(load_corpus(corpus_path))
    console.print(f"  Source: {corpus_path}")
    console.print(f"  Total speeches: {len(speeches):,}")

    # Load train data for few-shot examples
    train_data = None
    if model_config.get("prompt", {}).get("variant", "").startswith("fewshot"):
        train_data = load_gold_standard(Path(data_config["splits_dir"]) / "train.jsonl")

    # Create classifier
    classifier = create_classifier(model_config, train_data=train_data)
    console.print(f"  Model: {classifier.model_name}")

    # Cost estimation on sample
    console.print("\n[bold]Estimating cost on sample...[/bold]")
    sample = speeches[:50]
    sample_predictions = await classifier.classify_batch(sample)
    avg_input = sum(p.token_usage.get("prompt_tokens", 0) for p in sample_predictions) / len(sample)
    avg_output = sum(p.token_usage.get("completion_tokens", 0) for p in sample_predictions) / len(sample)

    cost_est = estimate_cost(
        classifier.model_name, len(speeches), avg_input, avg_output,
    )
    console.print(f"  Avg input tokens: {avg_input:.0f}")
    console.print(f"  Avg output tokens: {avg_output:.0f}")
    console.print(f"  [bold]Estimated cost: ${cost_est.estimated_cost_usd:.2f}[/bold]")
    console.print(f"  Batch API cost: ${cost_est.batch_api_cost_usd:.2f}")

    if dry_run:
        console.print("\n[yellow]Dry run — stopping before full classification.[/yellow]")
        return

    budget = batch_settings.get("cost_limits", {}).get("max_budget_usd", 100.0)
    if cost_est.estimated_cost_usd > budget:
        console.print(f"\n[red]Estimated cost ${cost_est.estimated_cost_usd:.2f} exceeds budget ${budget:.2f}[/red]")
        console.print("Increase budget in configs/batch_processing.yaml or use a cheaper model.")
        return

    # Initialize tracker
    tracker = UnifiedTracker(batch_config)
    tracker.start_experiment(experiment_config.get("name", "batch"), batch_config)

    # Determine rate limits
    model_type = model_config.get("model", {}).get("type", "openai")
    rate_config = batch_settings.get("rate_limits", {}).get(model_type, {})
    max_concurrent = rate_config.get("max_concurrent", 20)

    # Run batch processing
    output_path = Path(
        batch_settings.get("output_path")
        or (Path(data_config["predictions_dir"]) / "full_corpus_predictions.jsonl")
    )
    processor = BatchProcessor(
        classifier=classifier,
        tracker=tracker,
        max_concurrent=max_concurrent,
        cost_budget_usd=budget,
        cost_warn_usd=batch_settings.get("cost_limits", {}).get("warn_at_usd", 50.0),
    )

    console.print(f"\n[bold]Starting batch classification ({max_concurrent} concurrent)...[/bold]")
    result = await processor.process(speeches, output_path)

    # Log summary
    tracker.log_metrics({
        "total_processed": float(result["processed"]),
        "total_errors": float(result["errors"]),
        "total_cost_usd": result["total_cost_usd"],
        "elapsed_seconds": result["elapsed_seconds"],
    })
    tracker.log_artifact(str(output_path))
    tracker.end_experiment()

    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Processed: {result['processed']:,}/{result['total']:,}")
    console.print(f"  Errors: {result['errors']}")
    console.print(f"  Cost: ${result['total_cost_usd']:.2f}")
    console.print(f"  Time: {result['elapsed_seconds']:.0f}s")


def main():
    parser = argparse.ArgumentParser(description="Batch classify full corpus")
    parser.add_argument("--config", required=True, help="Batch processing config")
    parser.add_argument("--model-config", required=True, help="Model config")
    parser.add_argument("--dry-run", action="store_true", help="Only estimate cost, don't classify")
    args = parser.parse_args()

    load_dotenv()
    batch_config = load_config(args.config)
    model_config = load_config(args.model_config)

    asyncio.run(run_batch(batch_config, model_config, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
