"""Convert raw annotation files to canonical JSONL format.

This module provides utilities to convert various raw data formats (CSV, Excel, JSON)
into the canonical GoldStandardEntry JSONL format used throughout the project.

The actual conversion logic depends on the raw data format. Adapt the `convert_csv`
or `convert_excel` functions to match your specific column names and structure.
"""

import json
from pathlib import Path

import pandas as pd

from src.schema.labels import ALL_LABELS
from src.schema.models import GoldStandardEntry


def convert_csv(
    input_path: str | Path,
    output_path: str | Path,
    column_mapping: dict[str, str] | None = None,
    label_columns: list[str] | None = None,
) -> dict:
    """Convert a CSV file with annotations to canonical JSONL format.

    Args:
        input_path: Path to the input CSV file.
        output_path: Path for the output JSONL file.
        column_mapping: Mapping from CSV column names to canonical field names.
            Example: {"tekst": "text", "mowca": "speaker", "partia": "party"}
        label_columns: List of column names that represent binary label indicators.
            If None, expects a single 'labels' column with comma-separated label names.

    Returns:
        Stats dict with counts and any conversion errors.
    """
    default_mapping = {
        "speech_id": "speech_id",
        "text": "text",
        "speaker": "speaker",
        "party": "party",
        "political_group": "political_group",
        "date": "date",
        "session_id": "session_id",
    }
    mapping = {**(column_mapping or {})}
    # Fill in defaults for unmapped fields
    for canonical, default_col in default_mapping.items():
        if canonical not in mapping:
            mapping[canonical] = default_col

    df = pd.read_csv(input_path, encoding="utf-8")
    errors = []
    entries = []

    for idx, row in df.iterrows():
        try:
            # Extract labels
            if label_columns:
                # Binary indicator columns (one per label)
                labels = [col for col in label_columns if row.get(col, 0) == 1]
            else:
                # Comma-separated labels in a single column
                labels_raw = str(row.get(mapping.get("labels", "labels"), ""))
                labels = [l.strip() for l in labels_raw.split(",") if l.strip()]

            # Validate labels
            valid_labels = [l for l in labels if l in ALL_LABELS]
            invalid_labels = [l for l in labels if l not in ALL_LABELS]
            if invalid_labels:
                errors.append(f"Row {idx}: invalid labels {invalid_labels}")

            entry = GoldStandardEntry(
                speech_id=str(row.get(mapping["speech_id"], f"speech_{idx:04d}")),
                text=str(row[mapping["text"]]),
                speaker=str(row.get(mapping["speaker"], "unknown")),
                party=str(row.get(mapping["party"], "unknown")),
                political_group=row.get(mapping["political_group"]) or None,
                date=str(row.get(mapping["date"], "2000-01-01")),
                session_id=row.get(mapping["session_id"]) or None,
                annotations={
                    "speech_id": str(row.get(mapping["speech_id"], f"speech_{idx:04d}")),
                    "labels": valid_labels,
                    "annotator": str(row.get("annotator", "manual")),
                },
            )
            entries.append(entry)
        except Exception as e:
            errors.append(f"Row {idx}: {e}")

    # Write output
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.model_dump_json() + "\n")

    return {
        "total_rows": len(df),
        "converted": len(entries),
        "errors": errors,
        "label_distribution": _count_labels(entries),
    }


def convert_excel(
    input_path: str | Path,
    output_path: str | Path,
    sheet_name: str | int = 0,
    column_mapping: dict[str, str] | None = None,
    label_columns: list[str] | None = None,
) -> dict:
    """Convert an Excel file with annotations to canonical JSONL format.

    Same logic as convert_csv but reads from Excel.
    """
    df = pd.read_excel(input_path, sheet_name=sheet_name)
    # Save as temporary CSV and reuse convert_csv logic
    temp_csv = Path(input_path).with_suffix(".tmp.csv")
    try:
        df.to_csv(temp_csv, index=False, encoding="utf-8")
        result = convert_csv(temp_csv, output_path, column_mapping, label_columns)
    finally:
        temp_csv.unlink(missing_ok=True)
    return result


def _count_labels(entries: list[GoldStandardEntry]) -> dict[str, int]:
    """Count label occurrences across entries."""
    counts: dict[str, int] = {}
    for entry in entries:
        if entry.annotations:
            for label in entry.annotations.labels:
                counts[label] = counts.get(label, 0) + 1
    return counts
