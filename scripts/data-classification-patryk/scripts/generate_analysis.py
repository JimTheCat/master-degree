"""Generate final statistical analysis and visualizations from full corpus predictions.

Usage:
    python scripts/generate_analysis.py
    python scripts/generate_analysis.py --predictions data/predictions/full_corpus_predictions.jsonl --corpus data/corpus/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.analysis.political_groups import aggregate_by_group, aggregate_by_party
from src.analysis.statistical_tests import run_all_chi_squared
from src.analysis.visualizations import plot_party_rhetoric_profile
from src.data.loader import load_corpus
from src.schema.models import Prediction

console = Console()


def load_predictions(path: str | Path) -> list[Prediction]:
    predictions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(Prediction.model_validate_json(line))
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Generate political analysis")
    parser.add_argument(
        "--predictions",
        default="data/predictions/full_corpus_predictions.jsonl",
        help="Path to corpus predictions",
    )
    parser.add_argument("--corpus", default="data/corpus/", help="Path to corpus")
    parser.add_argument("--output-dir", default="outputs/analysis", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    console.print("[bold]Loading corpus and predictions...[/bold]")
    speeches = list(load_corpus(args.corpus))
    predictions = load_predictions(args.predictions)

    # Align by speech_id
    pred_by_id = {p.speech_id: p for p in predictions}
    aligned_speeches = []
    aligned_predictions = []
    for speech in speeches:
        if speech.speech_id in pred_by_id:
            aligned_speeches.append(speech)
            aligned_predictions.append(pred_by_id[speech.speech_id])

    console.print(f"  Matched: {len(aligned_speeches):,} speeches")

    # Aggregate by party
    console.print("\n[bold]Aggregating by party...[/bold]")
    party_df = aggregate_by_party(aligned_speeches, aligned_predictions)
    party_df.to_csv(output_dir / "party_aggregation.csv", index=False)

    table = Table(title="Label frequency by party")
    table.add_column("Party", style="cyan")
    table.add_column("Speeches", justify="right")
    from src.schema.labels import ALL_LABELS

    for label in ALL_LABELS[:5]:  # Show first 5 to keep it readable
        table.add_column(label[:20], justify="right")

    for _, row in party_df.iterrows():
        values = [row["party"], str(row["total_speeches"])]
        for label in ALL_LABELS[:5]:
            values.append(f"{row[label]:.3f}")
        table.add_row(*values)
    console.print(table)

    # Aggregate by political group
    group_df = aggregate_by_group(aligned_speeches, aligned_predictions)
    group_df.to_csv(output_dir / "group_aggregation.csv", index=False)

    # Chi-squared tests
    console.print("\n[bold]Running chi-squared tests...[/bold]")
    chi2_results = run_all_chi_squared(party_df, group_col="party")
    chi2_results.to_csv(output_dir / "chi_squared_results.csv", index=False)

    table2 = Table(title="Chi-squared tests (label ~ party)")
    table2.add_column("Label", style="cyan")
    table2.add_column("Chi2", justify="right")
    table2.add_column("p-value", justify="right")
    table2.add_column("Cramér's V", justify="right")
    table2.add_column("Significant", justify="center")

    for _, row in chi2_results.iterrows():
        if "error" in row and pd.notna(row.get("error")):
            continue
        sig = "[green]Yes[/green]" if row.get("significant") else "[red]No[/red]"
        table2.add_row(
            str(row["label"]),
            f"{row.get('chi2', 0):.2f}",
            f"{row.get('p_value', 1):.4f}",
            f"{row.get('cramers_v', 0):.3f}",
            sig,
        )
    console.print(table2)

    # Visualizations
    console.print("\n[bold]Generating visualizations...[/bold]")
    plot_party_rhetoric_profile(
        party_df,
        output_dir / "party_rhetoric_profile.png",
        group_col="party",
    )
    console.print(f"  Saved: {output_dir / 'party_rhetoric_profile.png'}")

    if not group_df.empty:
        plot_party_rhetoric_profile(
            group_df,
            output_dir / "group_rhetoric_profile.png",
            group_col="group",
            title="Rhetorical profile by parliamentary group",
        )
        console.print(f"  Saved: {output_dir / 'group_rhetoric_profile.png'}")

    console.print(f"\n[bold green]Analysis complete! Results in {output_dir}[/bold green]")


if __name__ == "__main__":
    import pandas as pd
    main()
