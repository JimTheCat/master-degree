"""Merge gold standard from three source files into canonical JSONL.

Source files:
    1. zloty-standard-badanie-patryk_cleaned.txt — 500 speeches (id + text, tab-separated)
    2. anotacje.csv — first annotation pass (id;JSON with emocje + techniki_retoryczne)
    3. rozbieznosci_resolved.csv — resolved disagreements (id;kategoria;resolved_labels;reasoning)

Logic:
    - Base labels come from anotacje.csv
    - Where disagreements were resolved (rozbieznosci_resolved.csv), those override the base
    - Speech text comes from the TXT file
    - Metadata (speaker, party, date) is omitted — will be joined from another file later

Usage:
    python scripts/merge_gold_standard.py
    python scripts/merge_gold_standard.py --texts zloty-standard-badanie-patryk_cleaned.txt --annotations anotacje.csv --resolved rozbieznosci_resolved.csv --output data/gold_standard/speeches.jsonl
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.schema.labels import ALL_LABELS

console = Console()

# Mapping from Polish-character label names (in source data) to ASCII canonical names
LABEL_NORMALIZATION = {
    # These labels have Polish diacritics in source files
    "OBLĘŻONA_TWIERDZA": "OBLEZIONA_TWIERDZA",
    "PRZYPISYWANIE_ZŁYCH_INTENCJI": "PRZYPISYWANIE_ZLYCH_INTENCJI",
    "DOWÓD_ANEGDOTYCZNY": "DOWOD_ANEGDOTYCZNY",
    "APEL_O_JEDNOŚĆ": "APEL_O_JEDNOSC",
    # These are already ASCII — map to themselves
    "AGRESJA_WERBALNA": "AGRESJA_WERBALNA",
    "STRATEGIA_STRACHU": "STRATEGIA_STRACHU",
    "DEHUMANIZACJA_POGARDA": "DEHUMANIZACJA_POGARDA",
    "DUMA_I_SUKCES": "DUMA_I_SUKCES",
    "MESJANIZM_MORALNY": "MESJANIZM_MORALNY",
    "POLARYZACJA_MY_ONI": "POLARYZACJA_MY_ONI",
    "AD_HOMINEM": "AD_HOMINEM",
    "WHATABOUTISM": "WHATABOUTISM",
    "SOFIZMAT_ROZSZERZENIA": "SOFIZMAT_ROZSZERZENIA",
    "OBLEZIONA_TWIERDZA": "OBLEZIONA_TWIERDZA",
    "PRZYPISYWANIE_ZLYCH_INTENCJI": "PRZYPISYWANIE_ZLYCH_INTENCJI",
    "DOWOD_ANEGDOTYCZNY": "DOWOD_ANEGDOTYCZNY",
    "APEL_O_JEDNOSC": "APEL_O_JEDNOSC",
}

# Labels to skip (errors in resolution)
SKIP_LABELS = {"NO_RESPONSE", "PARSE_ERROR", ""}


def normalize_label(label: str) -> str | None:
    """Normalize a label name to canonical ASCII form. Returns None if invalid."""
    label = label.strip()
    if label in SKIP_LABELS:
        return None
    normalized = LABEL_NORMALIZATION.get(label)
    if normalized and normalized in ALL_LABELS:
        return normalized
    if label in ALL_LABELS:
        return label
    return None


def load_texts(path: str) -> dict[str, str]:
    """Load speech texts from tab-separated file (id\\ttext)."""
    texts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) == 2:
                texts[parts[0]] = parts[1]
    return texts


def load_base_annotations(path: str) -> dict[str, dict]:
    """Load base annotations from anotacje.csv (id;JSON)."""
    annotations = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            speech_id = row[0].strip()
            try:
                data = json.loads(row[1])
                emocje = [normalize_label(l) for l in data.get("emocje", [])]
                techniki = [normalize_label(l) for l in data.get("techniki_retoryczne", [])]
                annotations[speech_id] = {
                    "emocje": [l for l in emocje if l is not None],
                    "techniki_retoryczne": [l for l in techniki if l is not None],
                }
            except (json.JSONDecodeError, KeyError) as e:
                console.print(f"[yellow]Warning: could not parse annotations for {speech_id}: {e}[/yellow]")
                annotations[speech_id] = {"emocje": [], "techniki_retoryczne": []}
    return annotations


def load_resolved(path: str) -> dict[str, dict]:
    """Load resolved disagreements from rozbieznosci_resolved.csv.

    Returns: {speech_id: {"emocje": [...], "techniki_retoryczne": [...]}}
    Only includes categories that were actually resolved.
    """
    resolved: dict[str, dict[str, list[str]]] = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            speech_id = row[0].strip()
            kategoria = row[1].strip()
            labels_raw = row[2].strip().strip('"')

            # Parse comma-separated labels
            if labels_raw:
                labels = [normalize_label(l) for l in labels_raw.split(",")]
                labels = [l for l in labels if l is not None]
            else:
                labels = []

            if kategoria == "emocje":
                resolved[speech_id]["emocje"] = labels
            elif kategoria == "techniki_retoryczne":
                resolved[speech_id]["techniki_retoryczne"] = labels

    return dict(resolved)


def merge(
    texts: dict[str, str],
    base_annotations: dict[str, dict],
    resolved: dict[str, dict],
) -> list[dict]:
    """Merge all sources into final gold standard entries."""
    entries = []
    stats = {
        "total": 0,
        "with_labels": 0,
        "resolved_overrides": 0,
        "no_annotations": 0,
    }

    for speech_id, text in texts.items():
        stats["total"] += 1

        # Start with base annotations
        base = base_annotations.get(speech_id, {"emocje": [], "techniki_retoryczne": []})
        emocje = list(base.get("emocje", []))
        techniki = list(base.get("techniki_retoryczne", []))

        # Override with resolved disagreements where available
        res = resolved.get(speech_id, {})
        if "emocje" in res:
            emocje = res["emocje"]
            stats["resolved_overrides"] += 1
        if "techniki_retoryczne" in res:
            techniki = res["techniki_retoryczne"]
            stats["resolved_overrides"] += 1

        all_labels = emocje + techniki

        if all_labels:
            stats["with_labels"] += 1
        elif speech_id not in base_annotations:
            stats["no_annotations"] += 1

        entry = {
            "speech_id": speech_id,
            "text": text,
            "speaker": "unknown",
            "party": "unknown",
            "date": "2000-01-01",
            "metadata": {
                "source": "parlaMint",
                "needs_metadata": True,
            },
            "annotations": {
                "speech_id": speech_id,
                "labels": all_labels,
                "annotator": "gold_standard",
            },
        }
        entries.append(entry)

    return entries, stats


def main():
    parser = argparse.ArgumentParser(description="Merge gold standard from source files")
    parser.add_argument("--texts", default="zloty-standard-badanie-patryk_cleaned.txt")
    parser.add_argument("--annotations", default="anotacje.csv")
    parser.add_argument("--resolved", default="rozbieznosci_resolved.csv")
    parser.add_argument("--output", default="data/gold_standard/speeches.jsonl")
    args = parser.parse_args()

    console.print("[bold]Merging gold standard...[/bold]")

    # Load all sources
    console.print("  Loading texts...")
    texts = load_texts(args.texts)
    console.print(f"    {len(texts)} speeches loaded")

    console.print("  Loading base annotations...")
    base_annotations = load_base_annotations(args.annotations)
    console.print(f"    {len(base_annotations)} annotation entries loaded")

    console.print("  Loading resolved disagreements...")
    resolved = load_resolved(args.resolved)
    console.print(f"    {len(resolved)} speeches with resolved disagreements")

    # Merge
    entries, stats = merge(texts, base_annotations, resolved)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    console.print(f"\n[green]Written {len(entries)} entries to {args.output}[/green]")
    console.print(f"  Total speeches: {stats['total']}")
    console.print(f"  With labels: {stats['with_labels']}")
    console.print(f"  Resolved overrides: {stats['resolved_overrides']}")
    console.print(f"  Without annotations: {stats['no_annotations']}")

    # Label distribution
    label_counts: dict[str, int] = {}
    for entry in entries:
        for label in entry["annotations"]["labels"]:
            label_counts[label] = label_counts.get(label, 0) + 1

    table = Table(title="Label Distribution (Gold Standard)")
    table.add_column("Label", style="cyan")
    table.add_column("Count", style="green", justify="right")
    table.add_column("% of speeches", justify="right")

    for label in ALL_LABELS:
        count = label_counts.get(label, 0)
        pct = count / len(entries) * 100 if entries else 0
        table.add_row(label, str(count), f"{pct:.1f}%")

    console.print(table)

    # Show speeches with no labels at all
    no_labels = [e for e in entries if not e["annotations"]["labels"]]
    console.print(f"\n  Speeches with no labels (neutral): {len(no_labels)}")


if __name__ == "__main__":
    main()
