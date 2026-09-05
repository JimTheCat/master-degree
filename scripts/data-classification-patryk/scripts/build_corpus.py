"""Build full corpus JSONL from a directory of ParlaMint TSV drop-ins.

Usage:
    python scripts/build_corpus.py
    python scripts/build_corpus.py --input-dir data/raw/corpus_input --output data/corpus/speeches.jsonl
    python scripts/build_corpus.py --speaker-metadata data/raw/speaker_party_map.tsv
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.data.corpus_loader import (
    load_speaker_metadata,
    parse_parlamint_tsv,
    to_corpus_record,
)

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Build corpus JSONL from ParlaMint TSV files")
    parser.add_argument("--input-dir", default="data/raw/corpus_input",
                        help="Directory of TSV files (drop-in)")
    parser.add_argument("--output", default="data/corpus/speeches.jsonl")
    parser.add_argument("--speaker-metadata", default="",
                        help="Optional speaker→party mapping (TSV/CSV)")
    parser.add_argument("--min-text-length", type=int, default=20)
    parser.add_argument("--patterns", nargs="+", default=["*.txt", "*.tsv"])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        console.print(f"[red]Input dir not found: {input_dir}[/red]")
        console.print(f"  Create it and drop ParlaMint TSV files in. See data/raw/zloty-standard-badanie-patryk_cleaned.txt for format.")
        sys.exit(1)

    files: list[Path] = []
    for pattern in args.patterns:
        files.extend(sorted(input_dir.glob(pattern)))
    if not files:
        console.print(f"[red]No files matching {args.patterns} in {input_dir}[/red]")
        sys.exit(1)

    console.print(f"[bold]Building corpus from {len(files)} file(s) in {input_dir}[/bold]")

    speaker_meta = load_speaker_metadata(args.speaker_metadata) if args.speaker_metadata else {}
    if speaker_meta:
        console.print(f"  Loaded speaker metadata for {len(speaker_meta)} speech IDs")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    chamber_counts: Counter[str] = Counter()
    year_counts: Counter[str] = Counter()
    n_in = 0
    n_out = 0
    n_skipped_short = 0
    n_skipped_dup = 0
    n_skipped_invalid = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for src in files:
            console.print(f"  Reading {src.name}")
            for parsed in parse_parlamint_tsv(src, min_text_length=args.min_text_length):
                n_in += 1
                speech_id = parsed["speech_id"]
                if speech_id in seen:
                    n_skipped_dup += 1
                    continue
                seen.add(speech_id)
                record = to_corpus_record(parsed, speaker_meta)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_out += 1
                chamber_counts[parsed["chamber"]] += 1
                year_counts[parsed["date"][:4]] += 1

    # Note: parse_parlamint_tsv silently skips short and invalid lines, so we
    # approximate counts from input file line totals.
    total_lines = sum(1 for f in files for _ in open(f, encoding="utf-8"))
    n_skipped_invalid_or_short = total_lines - n_in - n_skipped_dup
    if n_skipped_invalid_or_short > 0:
        n_skipped_short = n_skipped_invalid_or_short  # Best-effort attribution.

    console.print(f"\n[green]Wrote {n_out} records to {output_path}[/green]")

    summary = Table(title="Corpus build summary")
    summary.add_column("Metric")
    summary.add_column("Count", justify="right")
    summary.add_row("Files processed", str(len(files)))
    summary.add_row("Records emitted", str(n_out))
    summary.add_row("Duplicates skipped", str(n_skipped_dup))
    summary.add_row("Invalid/short skipped", str(n_skipped_short))
    summary.add_row("Unique speech IDs", str(len(seen)))
    console.print(summary)

    if chamber_counts:
        chambers = Table(title="Chamber distribution")
        chambers.add_column("Chamber")
        chambers.add_column("Count", justify="right")
        for chamber, c in chamber_counts.most_common():
            chambers.add_row(chamber, str(c))
        console.print(chambers)

    if year_counts:
        years = Table(title="Year distribution (top 15)")
        years.add_column("Year")
        years.add_column("Count", justify="right")
        for year, c in sorted(year_counts.items()):
            years.add_row(year, str(c))
        console.print(years)


if __name__ == "__main__":
    main()
