"""Convert raw annotation files to canonical JSONL format.

Usage:
    python scripts/convert_annotations.py --input data/raw/annotations.csv --output data/gold_standard/speeches.jsonl
    python scripts/convert_annotations.py --input data/raw/annotations.xlsx --output data/gold_standard/speeches.jsonl

Adapt column_mapping and label_columns to match your raw data format.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.data.converter import convert_csv, convert_excel
from src.data.validation import validate_gold_standard

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Convert raw annotations to JSONL")
    parser.add_argument("--input", required=True, help="Path to input file (CSV or Excel)")
    parser.add_argument("--output", required=True, help="Path for output JSONL file")
    parser.add_argument(
        "--format", choices=["csv", "excel"], default=None,
        help="Input format (auto-detected from extension if not specified)",
    )
    parser.add_argument(
        "--label-columns", nargs="*", default=None,
        help="Column names for binary label indicators. If not given, expects 'labels' column.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        console.print(f"[red]Error: Input file not found: {input_path}[/red]")
        sys.exit(1)

    # Auto-detect format
    fmt = args.format
    if fmt is None:
        if input_path.suffix in (".xlsx", ".xls"):
            fmt = "excel"
        else:
            fmt = "csv"

    # ---------------------------------------------------------------
    # ADAPT THIS MAPPING to match your raw data columns.
    # Keys = canonical field names, values = column names in your file.
    # ---------------------------------------------------------------
    column_mapping = {
        "speech_id": "speech_id",
        "text": "text",
        "speaker": "speaker",
        "party": "party",
        "political_group": "political_group",
        "date": "date",
        "session_id": "session_id",
    }

    console.print(f"[bold]Converting {input_path} ({fmt}) → {args.output}[/bold]")

    if fmt == "excel":
        result = convert_excel(
            input_path, args.output,
            column_mapping=column_mapping,
            label_columns=args.label_columns,
        )
    else:
        result = convert_csv(
            input_path, args.output,
            column_mapping=column_mapping,
            label_columns=args.label_columns,
        )

    # Print results
    console.print(f"\n[green]Converted: {result['converted']}/{result['total_rows']} rows[/green]")

    if result["errors"]:
        console.print(f"[yellow]Warnings ({len(result['errors'])}):[/yellow]")
        for err in result["errors"][:10]:
            console.print(f"  - {err}")
        if len(result["errors"]) > 10:
            console.print(f"  ... and {len(result['errors']) - 10} more")

    # Label distribution table
    table = Table(title="Label Distribution")
    table.add_column("Label", style="cyan")
    table.add_column("Count", style="green", justify="right")
    for label, count in sorted(result["label_distribution"].items(), key=lambda x: -x[1]):
        table.add_row(label, str(count))
    console.print(table)

    # Validate output
    console.print("\n[bold]Validating output...[/bold]")
    validation = validate_gold_standard(args.output)
    console.print(f"Valid entries: {validation['valid']}/{validation['total']}")
    console.print(f"Unique speakers: {validation['speakers']}")
    console.print(f"Unique parties: {validation['parties']}")

    if validation["errors"]:
        console.print(f"[yellow]Validation warnings: {len(validation['errors'])}[/yellow]")


if __name__ == "__main__":
    main()
