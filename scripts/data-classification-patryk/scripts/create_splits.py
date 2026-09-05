"""Create stratified train/val/test splits from gold standard data.

Usage:
    python scripts/create_splits.py
    python scripts/create_splits.py --config configs/base.yaml
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from rich.console import Console

from src.data.splitter import create_splits

console = Console()


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits")
    parser.add_argument("--config", default="configs/base.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]

    gold_path = data_config["gold_standard_path"]
    splits_dir = data_config["splits_dir"]
    ratios = data_config["split_ratios"]
    seed = config["project"]["random_seed"]

    if not Path(gold_path).exists():
        console.print(f"[red]Gold standard not found: {gold_path}[/red]")
        console.print("Run scripts/convert_annotations.py first.")
        sys.exit(1)

    console.print(f"[bold]Creating splits from {gold_path}[/bold]")
    console.print(f"  Train: {ratios['train']:.0%}, Val: {ratios['val']:.0%}, Test: {ratios['test']:.0%}")
    console.print(f"  Seed: {seed}")

    counts = create_splits(
        gold_standard_path=gold_path,
        output_dir=splits_dir,
        train_ratio=ratios["train"],
        val_ratio=ratios["val"],
        test_ratio=ratios["test"],
        random_seed=seed,
    )

    console.print(f"\n[green]Splits created in {splits_dir}:[/green]")
    for split_name, count in counts.items():
        console.print(f"  {split_name}: {count} speeches")


if __name__ == "__main__":
    main()
