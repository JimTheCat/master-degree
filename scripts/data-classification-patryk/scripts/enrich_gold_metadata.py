"""Fill in gold-standard speech metadata from the full ParlaMint export.

data/gold_standard/speeches.jsonl was created with placeholder metadata
(speaker/party = "unknown", date = 2000-01-01, needs_metadata flag).
data/raw/merged_all.tsv holds the full per-utterance ParlaMint-PL metadata,
keyed by the same utterance ID. This script joins the two and rewrites
speeches.jsonl and the derived split files in place (originals backed up
as *.bak once).

Usage:
    python scripts/enrich_gold_metadata.py
"""

import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

METADATA_TSV = Path("data/raw/merged_all.tsv")
GOLD_PATH = Path("data/gold_standard/speeches.jsonl")
SPLIT_PATHS = [Path("data/splits") / f"{name}.jsonl" for name in ("train", "val", "test")]

console = Console()


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value if value and value != "-" else None


def load_metadata(needed_ids: set[str]) -> dict[str, dict]:
    """Stream the (large) TSV and keep only rows for the gold-standard ids."""
    found: dict[str, dict] = {}
    with open(METADATA_TSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            utt_id = row.get("ID", "")
            if utt_id in needed_ids:
                found[utt_id] = row
                if len(found) == len(needed_ids):
                    break
    return found


def enrich_record(record: dict, meta: dict) -> dict:
    party = _clean(meta.get("Speaker_party")) or _clean(meta.get("Speaker_party_name"))
    speaker = _clean(meta.get("Speaker_name")) or _clean(meta.get("Speaker_ID"))
    record["speaker"] = speaker or "unknown"
    record["party"] = party or "unknown"
    record["political_group"] = _clean(meta.get("Speaker_party_name"))
    record["date"] = _clean(meta.get("Date")) or record["date"]

    extra = {
        "body": _clean(meta.get("Body")),
        "term": _clean(meta.get("Term")),
        "speaker_role": _clean(meta.get("Speaker_role")),
        "speaker_id": _clean(meta.get("Speaker_ID")),
        "speaker_gender": _clean(meta.get("Speaker_gender")),
        "party_status": _clean(meta.get("Party_status")),
        "party_orientation": _clean(meta.get("Party_orientation")),
    }
    metadata = dict(record.get("metadata") or {})
    metadata.pop("needs_metadata", None)
    metadata.update({k: v for k, v in extra.items() if v is not None})
    record["metadata"] = metadata
    return record


def rewrite_file(path: Path, meta_by_id: dict[str, dict]) -> tuple[int, int]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    enriched = 0
    for record in records:
        meta = meta_by_id.get(record["speech_id"])
        if meta:
            enrich_record(record, meta)
            enriched += 1

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)

    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return enriched, len(records)


def main() -> None:
    if not METADATA_TSV.exists():
        console.print(f"[red]Brak pliku {METADATA_TSV}[/red]")
        sys.exit(1)

    with open(GOLD_PATH, encoding="utf-8") as f:
        needed_ids = {json.loads(line)["speech_id"] for line in f if line.strip()}
    console.print(f"Gold standard: {len(needed_ids)} wypowiedzi")

    console.print(f"Czytanie metadanych z {METADATA_TSV} (strumieniowo)...")
    meta_by_id = load_metadata(needed_ids)
    missing = needed_ids - set(meta_by_id)
    console.print(f"Znaleziono metadane: {len(meta_by_id)}/{len(needed_ids)}")
    if missing:
        console.print(
            f"[yellow]WARN brak metadanych dla {len(missing)} id, "
            f"np. {sorted(missing)[:3]}[/yellow]"
        )

    for path in [GOLD_PATH, *SPLIT_PATHS]:
        enriched, total = rewrite_file(path, meta_by_id)
        console.print(f"  {path}: uzupełniono {enriched}/{total} (backup: {path.name}.bak)")

    console.print("[green]OK[/green]")


if __name__ == "__main__":
    main()
