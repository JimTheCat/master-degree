"""Query MLflow for all model x prompt runs and select the best configuration.

Combines macro-F1, cost, and latency to recommend a single config for full-corpus
inference. Outputs:
  - results/comparison_table.md
  - results/best_config.yaml
  - results/cost_effectiveness.png

Usage:
    python scripts/select_best_config.py
    python scripts/select_best_config.py --min-macro-f1 0.55 --metric macro_f1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table

from src.analysis.visualizations import plot_cost_effectiveness

console = Console()

# USD per 1M tokens (input, output) — rough public pricing as of 2025-2026.
# Local models = 0. Approximate; update when pricing shifts.
MODEL_PRICING = {
    "claude-opus-4-7": (15.00, 75.00),
    "claude-opus": (15.00, 75.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-1.5-pro": (1.25, 5.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "speakleash/bielik-11b-v2.3-instruct": (0.0, 0.0),
    "google/gemma-4-26b-a4b": (0.0, 0.0),
    "tfidf-logreg": (0.0, 0.0),
    "allegro/herbert-base-cased": (0.0, 0.0),
}


def estimate_cost(model_name: str, input_tokens: float, output_tokens: float) -> float:
    """Best-effort cost estimate. Returns USD."""
    in_price, out_price = (0.0, 0.0)
    for key, prices in MODEL_PRICING.items():
        if key.lower() in model_name.lower():
            in_price, out_price = prices
            break
    return (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price


def fetch_runs(experiment_names: list[str], tracking_uri: str) -> pd.DataFrame:
    mlflow.set_tracking_uri(tracking_uri)
    rows = []
    for exp_name in experiment_names:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            console.print(f"[yellow]Experiment {exp_name!r} not found, skipping[/yellow]")
            continue
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id], output_format="pandas")
        if runs.empty:
            continue
        runs["experiment_name"] = exp_name
        rows.append(runs)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def normalize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    """Pull out the columns we care about and add cost estimate."""
    if runs.empty:
        return runs

    out = pd.DataFrame()
    out["experiment"] = runs["experiment_name"]
    out["run_name"] = runs.get("tags.mlflow.runName", runs.get("run_id"))
    out["model"] = runs.get("params.model.name", "")
    out["model_type"] = runs.get("params.model.type", "")
    out["prompt"] = runs.get("params.prompt.variant", "")
    out["macro_f1"] = pd.to_numeric(runs.get("metrics.macro_f1"), errors="coerce")
    out["micro_f1"] = pd.to_numeric(runs.get("metrics.micro_f1"), errors="coerce")
    out["weighted_f1"] = pd.to_numeric(runs.get("metrics.weighted_f1"), errors="coerce")
    out["hamming_loss"] = pd.to_numeric(runs.get("metrics.hamming_loss"), errors="coerce")
    out["jaccard"] = pd.to_numeric(runs.get("metrics.jaccard"), errors="coerce")
    out["emotions_macro_f1"] = pd.to_numeric(runs.get("metrics.emotions_macro_f1"), errors="coerce")
    out["rhetorical_macro_f1"] = pd.to_numeric(runs.get("metrics.rhetorical_macro_f1"), errors="coerce")
    out["input_tokens"] = pd.to_numeric(runs.get("metrics.total_input_tokens", 0), errors="coerce").fillna(0)
    out["output_tokens"] = pd.to_numeric(runs.get("metrics.total_output_tokens", 0), errors="coerce").fillna(0)
    out["avg_latency_ms"] = pd.to_numeric(runs.get("metrics.avg_latency_ms", 0), errors="coerce").fillna(0)

    out["cost_usd"] = out.apply(
        lambda r: estimate_cost(str(r["model"]), r["input_tokens"], r["output_tokens"]),
        axis=1,
    )

    # Drop rows without F1 (failed runs).
    out = out.dropna(subset=["macro_f1"]).reset_index(drop=True)
    return out


def write_comparison_table(df: pd.DataFrame, path: Path):
    cols = [
        "experiment", "model", "prompt", "macro_f1", "micro_f1",
        "emotions_macro_f1", "rhetorical_macro_f1", "hamming_loss",
        "jaccard", "cost_usd", "avg_latency_ms",
    ]
    table = df[cols].sort_values("macro_f1", ascending=False).copy()
    for col in ("macro_f1", "micro_f1", "emotions_macro_f1", "rhetorical_macro_f1", "hamming_loss", "jaccard"):
        table[col] = table[col].round(4)
    table["cost_usd"] = table["cost_usd"].round(4)
    table["avg_latency_ms"] = table["avg_latency_ms"].round(0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Model x Prompt comparison (sorted by macro-F1)\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n")
    console.print(f"[green]Wrote {path}[/green]")


def select_best(df: pd.DataFrame, min_macro_f1: float) -> dict:
    """Pick best config: highest macro-F1, tiebreak by lowest cost."""
    candidates = df[df["macro_f1"] >= min_macro_f1]
    pool = candidates if not candidates.empty else df
    pool = pool.sort_values(by=["macro_f1", "cost_usd"], ascending=[False, True])
    if pool.empty:
        raise SystemExit("No runs available — run sweep first.")
    best = pool.iloc[0].to_dict()
    return best


def write_best_config(best: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "selected_at_macro_f1": float(best["macro_f1"]),
        "model": {
            "name": best["model"],
            "type": best["model_type"],
        },
        "prompt": {
            "variant": best["prompt"],
        },
        "estimated_cost_usd_test": float(best["cost_usd"]),
        "avg_latency_ms": float(best["avg_latency_ms"]),
        "source_run": {
            "experiment": best["experiment"],
            "run_name": best.get("run_name", ""),
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)
    console.print(f"[green]Wrote {path}[/green]")


def print_summary(df: pd.DataFrame, best: dict):
    table = Table(title="Top configurations by Macro-F1")
    table.add_column("Model", style="cyan")
    table.add_column("Prompt", style="magenta")
    table.add_column("Macro-F1", justify="right")
    table.add_column("Micro-F1", justify="right")
    table.add_column("Cost USD", justify="right")
    table.add_column("Latency ms", justify="right")
    for _, row in df.sort_values("macro_f1", ascending=False).head(10).iterrows():
        table.add_row(
            str(row["model"]), str(row["prompt"]),
            f"{row['macro_f1']:.4f}", f"{row['micro_f1']:.4f}",
            f"{row['cost_usd']:.4f}", f"{row['avg_latency_ms']:.0f}",
        )
    console.print(table)
    console.print(f"\n[bold green]Recommended: {best['model']} / {best['prompt']} (macro-F1={best['macro_f1']:.4f}, ~${best['cost_usd']:.4f})[/bold green]")


def main():
    parser = argparse.ArgumentParser(description="Select best model x prompt from MLflow runs")
    parser.add_argument("--tracking-uri", default="file:./mlruns")
    parser.add_argument("--experiments", nargs="+", default=[
        "parliament-model-comparison",
        "parliament-baseline",
    ])
    parser.add_argument("--min-macro-f1", type=float, default=0.0,
                        help="Minimum macro-F1 to be eligible. If none meet it, falls back to global best.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    runs = fetch_runs(args.experiments, args.tracking_uri)
    if runs.empty:
        raise SystemExit("No runs found in any of the given experiments.")
    df = normalize_runs(runs)
    if df.empty:
        raise SystemExit("All runs lack macro_f1 metric.")

    results_dir = Path(args.results_dir)
    write_comparison_table(df, results_dir / "comparison_table.md")
    df.to_csv(results_dir / "comparison_table.csv", index=False)

    plot_cost_effectiveness(
        [{"name": f"{r['model']}/{r['prompt']}", "macro_f1": r["macro_f1"], "cost_usd": r["cost_usd"]}
         for _, r in df.iterrows()],
        results_dir / "cost_effectiveness.png",
    )

    best = select_best(df, args.min_macro_f1)
    write_best_config(best, results_dir / "best_config.yaml")
    print_summary(df, best)


if __name__ == "__main__":
    main()
