"""Sweep LLM experiments across a matrix of models x prompt variants.

Each combination is dispatched through run_llm_experiment.run_experiment, so
MLflow and LangSmith tracking match a single-run invocation exactly.

Per-run console output is appended to results/logs/experiments.log; the sweep
additionally writes a summary table to results/sweep_summary_<timestamp>.csv.

Usage:
    python scripts/run_llm_sweep.py
    python scripts/run_llm_sweep.py --models configs/model_gpt.yaml configs/model_claude.yaml
    python scripts/run_llm_sweep.py --prompts zeroshot fewshot
    python scripts/run_llm_sweep.py --split all --resume
"""

import argparse
import asyncio
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich.table import Table

from scripts.run_llm_experiment import console, run_experiment, save_console_log
from src.pipeline.runner import load_config

RESULTS_DIR = Path("results")

DEFAULT_MODEL_CONFIGS = [
    "configs/model_gpt.yaml",
    "configs/model_claude.yaml",
    "configs/model_gemini.yaml",
    "configs/model_gemini_flash.yaml",
    "configs/model_bielik.yaml",
    "configs/model_gemma.yaml",
]

DEFAULT_PROMPT_VARIANTS = ["zeroshot", "fewshot", "detailed", "fewshot_detailed"]

SUMMARY_FIELDS = [
    "experiment", "model", "variant", "split", "n_speeches", "n_failed",
    "macro_f1", "micro_f1", "emotions_macro_f1", "rhetorical_macro_f1",
    "total_tokens", "avg_latency_ms", "error",
]


def _apply_prompt_override(config: dict, prompt_variant: str) -> dict:
    prompt = config.setdefault("prompt", {})
    prompt["variant"] = prompt_variant
    if prompt_variant in ("fewshot", "fewshot_detailed"):
        prompt.setdefault("num_examples", 3)
        prompt.setdefault("example_selection", "stratified")

    experiment = config.setdefault("experiment", {})
    base_name = experiment.get("name", "experiment")
    experiment["name"] = f"{base_name}-{prompt_variant}"
    experiment["mlflow_experiment"] = "parliament-model-comparison"
    return config


def _write_summary(rows: list[dict], split: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"sweep_summary_{stamp}.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)

    table = Table(title=f"Sweep summary (split={split})")
    table.add_column("Model", style="cyan")
    table.add_column("Variant")
    table.add_column("Macro-F1", justify="right")
    table.add_column("Micro-F1", justify="right")
    table.add_column("Failed", justify="right")
    for row in sorted(rows, key=lambda r: r.get("macro_f1") or 0.0, reverse=True):
        if row.get("error"):
            table.add_row(row["model"], row["variant"], "[red]FAILED[/red]", "-", "-")
        else:
            table.add_row(
                row["model"],
                row["variant"],
                f"{row['macro_f1']:.4f}",
                f"{row['micro_f1']:.4f}",
                str(row["n_failed"]),
            )
    console.print(table)
    console.print(f"Summary CSV: {out_path}")
    save_console_log("sweep-summary")
    return out_path


async def run_sweep(model_paths: list[str], prompt_variants: list[str], split: str,
                    resume: bool, limit: int | None, track: bool):
    rows: list[dict] = []
    for model_path in model_paths:
        for variant in prompt_variants:
            console.rule(f"[bold cyan]{Path(model_path).stem} x {variant}")
            config = load_config(model_path)
            config = _apply_prompt_override(config, variant)
            try:
                summary = await run_experiment(
                    config, split=split, resume=resume, limit=limit, track=track
                )
                summary["error"] = ""
                rows.append(summary)
            except Exception as exc:
                console.print(f"[red]FAILED {model_path} / {variant}: {exc}[/red]")
                rows.append({
                    "experiment": config.get("experiment", {}).get("name", "?"),
                    "model": Path(model_path).stem,
                    "variant": variant,
                    "split": split,
                    "error": f"{type(exc).__name__}: {exc}"[:200],
                })
    _write_summary(rows, split)


def main():
    parser = argparse.ArgumentParser(description="Sweep LLM x prompt matrix")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODEL_CONFIGS)
    parser.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPT_VARIANTS)
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val", "train", "all"],
        help="Dataset per cell: a split or 'all' = full 500-speech gold standard",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip speeches already present in each cell's predictions file",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Classify only the first N speeches per cell (sanity runs)",
    )
    parser.add_argument("--no-track", action="store_true", help="Skip MLflow/LangSmith logging")
    args = parser.parse_args()

    load_dotenv()
    asyncio.run(run_sweep(
        args.models, args.prompts, split=args.split, resume=args.resume,
        limit=args.limit, track=not args.no_track,
    ))


if __name__ == "__main__":
    main()
