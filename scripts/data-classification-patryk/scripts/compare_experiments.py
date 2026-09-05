"""Compare experiment runs and generate comparison tables and charts.

Usage:
    python scripts/compare_experiments.py
    python scripts/compare_experiments.py --experiments parliament-prompt-engineering parliament-model-comparison
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.analysis.visualizations import plot_label_f1_heatmap, plot_radar_chart
from src.evaluation.comparator import compare_mlflow_runs

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Compare experiment runs")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=[
            "parliament-baseline",
            "parliament-prompt-engineering",
            "parliament-model-comparison",
        ],
        help="MLflow experiment names to compare",
    )
    parser.add_argument(
        "--tracking-uri",
        default="file:./mlruns",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/comparison",
        help="Directory for output charts",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Comparing experiments: {', '.join(args.experiments)}[/bold]")

    comparison = compare_mlflow_runs(
        tracking_uri=args.tracking_uri,
        experiment_names=args.experiments,
    )

    if comparison.summary_df.empty:
        console.print("[yellow]No runs found in the specified experiments.[/yellow]")
        return

    # Print summary table
    table = Table(title="Experiment Comparison - Summary Metrics")
    table.add_column("Metric", style="cyan")
    for col in comparison.summary_df.columns:
        table.add_column(str(col), justify="right")

    for metric, row in comparison.summary_df.iterrows():
        values = [f"{v:.4f}" if v is not None else "N/A" for v in row]
        table.add_row(str(metric), *values)
    console.print(table)

    # Print per-label F1 table
    if not comparison.per_label_df.empty:
        table2 = Table(title="Per-label F1 Scores")
        table2.add_column("Label", style="cyan")
        for col in comparison.per_label_df.columns:
            table2.add_column(str(col), justify="right")

        for label, row in comparison.per_label_df.iterrows():
            values = [f"{v:.3f}" if v is not None else "N/A" for v in row]
            table2.add_row(str(label), *values)
        console.print(table2)

        # Generate charts
        console.print("\n[bold]Generating charts...[/bold]")
        plot_label_f1_heatmap(
            comparison.per_label_df,
            output_dir / "label_f1_heatmap.png",
        )
        console.print(f"  Saved: {output_dir / 'label_f1_heatmap.png'}")

        plot_radar_chart(
            comparison.per_label_df,
            output_dir / "radar_chart.png",
        )
        console.print(f"  Saved: {output_dir / 'radar_chart.png'}")

    # Cost comparison
    if comparison.cost_df is not None:
        table3 = Table(title="Cost Comparison")
        table3.add_column("Metric", style="cyan")
        for col in comparison.cost_df.columns:
            table3.add_column(str(col), justify="right")

        for metric, row in comparison.cost_df.iterrows():
            values = [f"{v:.2f}" if v is not None else "N/A" for v in row]
            table3.add_row(str(metric), *values)
        console.print(table3)

    console.print(f"\n[green]Charts saved to {output_dir}[/green]")


if __name__ == "__main__":
    main()
