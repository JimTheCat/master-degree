import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.data.loader import load_speeches_jsonl
from src.schema.labels import ALL_LABELS
from src.schema.models import GoldStandardEntry


def _build_label_matrix(entries: list[GoldStandardEntry]) -> np.ndarray:
    """Build a binary label matrix for stratified splitting."""
    matrix = np.zeros((len(entries), len(ALL_LABELS)), dtype=int)
    label_to_idx = {label: i for i, label in enumerate(ALL_LABELS)}
    for i, entry in enumerate(entries):
        if entry.annotations:
            for label in entry.annotations.labels:
                if label in label_to_idx:
                    matrix[i, label_to_idx[label]] = 1
    return matrix


def _stratify_key(matrix: np.ndarray) -> list[str]:
    """Create a string key per sample for approximate multi-label stratification.

    Converts the binary label vector to a string like '10010...' so that
    sklearn's stratified split groups samples with identical label combinations.
    """
    return ["".join(str(x) for x in row) for row in matrix]


def create_splits(
    gold_standard_path: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> dict[str, int]:
    """Create stratified train/val/test splits from gold standard data.

    Returns a dict with the count of entries in each split.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    entries = load_speeches_jsonl(gold_standard_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_matrix = _build_label_matrix(entries)
    strat_keys = _stratify_key(label_matrix)

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    try:
        train_entries, val_test_entries, _, val_test_keys = train_test_split(
            entries,
            strat_keys,
            test_size=val_test_ratio,
            random_state=random_seed,
            stratify=strat_keys,
        )
    except ValueError:
        # If stratification fails (too few samples per class), fall back to random split
        train_entries, val_test_entries, _, val_test_keys = train_test_split(
            entries,
            strat_keys,
            test_size=val_test_ratio,
            random_state=random_seed,
        )

    # Second split: val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    try:
        val_entries, test_entries = train_test_split(
            val_test_entries,
            test_size=relative_test_ratio,
            random_state=random_seed,
            stratify=val_test_keys,
        )
    except ValueError:
        val_entries, test_entries = train_test_split(
            val_test_entries,
            test_size=relative_test_ratio,
            random_state=random_seed,
        )

    # Write splits
    splits = {"train": train_entries, "val": val_entries, "test": test_entries}
    counts = {}
    for split_name, split_entries in splits.items():
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in split_entries:
                f.write(entry.model_dump_json() + "\n")
        counts[split_name] = len(split_entries)

    return counts
