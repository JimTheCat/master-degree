"""Fill in full-corpus speech metadata from the ParlaMint export.

data/corpus/speeches.jsonl was built with placeholder metadata
(speaker/party = "unknown", needs_metadata flag) because build_corpus.py ran
without --speaker-metadata. data/raw/merged_all.tsv holds the full
per-utterance ParlaMint-PL metadata, keyed by the same utterance ID.

Unlike enrich_gold_metadata.py (500 records, loads everything), this streams
the corpus record-by-record and rewrites it via a temp file; the original is
backed up once as speeches.jsonl.bak.

Usage:
    python scripts/enrich_corpus_metadata.py
    python scripts/enrich_corpus_metadata.py --corpus data/corpus/speeches.jsonl
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console

METADATA_TSV = Path("data/raw/merged_all.tsv")

# Columns kept per utterance; the full DictReader row would hold Title/Agenda
# strings that multiply memory use ~5x across 220k+ rows.
META_COLUMNS = (
    "Date",
    "Body",
    "Term",
    "Speaker_role",
    "Speaker_ID",
    "Speaker_name",
    "Speaker_gender",
    "Speaker_party",
    "Speaker_party_name",
    "Party_status",
    "Party_orientation",
)

console = Console()


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value if value and value != "-" else None


def load_metadata() -> dict[str, dict]:
    found: dict[str, dict] = {}
    with open(METADATA_TSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            utt_id = row.get("ID", "")
            if utt_id:
                found[utt_id] = {col: row.get(col) for col in META_COLUMNS}
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich full corpus with ParlaMint metadata")
    parser.add_argument("--corpus", default="data/corpus/speeches.jsonl")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not METADATA_TSV.exists():
        console.print(f"[red]Brak pliku {METADATA_TSV}[/red]")
        sys.exit(1)
    if not corpus_path.exists():
        console.print(f"[red]Brak pliku {corpus_path}[/red]")
        sys.exit(1)

    console.print(f"Czytanie metadanych z {METADATA_TSV}...")
    meta_by_id = load_metadata()
    console.print(f"  Zaladowano metadane dla {len(meta_by_id):,} wypowiedzi")

    backup = corpus_path.with_suffix(corpus_path.suffix + ".bak")
    if not backup.exists():
        console.print(f"Backup: {backup}")
        shutil.copy2(corpus_path, backup)

    tmp_path = corpus_path.with_suffix(".jsonl.tmp")
    n_total = 0
    n_enriched = 0
    n_with_party = 0
    with open(corpus_path, encoding="utf-8") as in_f, \
            open(tmp_path, "w", encoding="utf-8") as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n_total += 1
            meta = meta_by_id.get(record["speech_id"])
            if meta:
                enrich_record(record, meta)
                n_enriched += 1
                if record["party"] != "unknown":
                    n_with_party += 1
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if n_total % 50000 == 0:
                console.print(f"  ...{n_total:,} rekordow")

    tmp_path.replace(corpus_path)

    console.print(f"[green]OK: {corpus_path}[/green]")
    console.print(f"  Rekordy:            {n_total:,}")
    console.print(f"  Uzupelnione:        {n_enriched:,}")
    console.print(f"  Ze znana partia:    {n_with_party:,}")
    console.print(f"  Bez partii (np. Marszalek): {n_total - n_with_party:,}")


if __name__ == "__main__":
    main()
