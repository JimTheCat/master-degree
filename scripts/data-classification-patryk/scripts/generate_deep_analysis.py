"""Chamber and speaker-level analyses on full-corpus predictions.

Complements generate_corpus_analysis.py, which compares parties, coalition
status, terms and time. Both dimensions here need controls that the aggregate
script does not apply:

  - Sejm vs Senat: restricted to speaking MPs/senators, then repeated on the
    speakers who appear in both chambers, so party and person are held constant.
  - Speakers: label density per 1000 characters next to the per-speech figure,
    with a split-half reliability check and the share of between-speaker
    variance that party explains.

Reads:
  - Corpus speeches: data/corpus/speeches.jsonl
  - Predictions: data/predictions/corpus_full.jsonl

Writes to results/analysis/: chamber_by_label.csv, chamber_chi_squared.csv,
chamber_density.csv, chamber_paired_speakers.csv, speaker_density.csv and four PNGs.

Usage:
    python scripts/generate_deep_analysis.py
    python scripts/generate_deep_analysis.py --min-paired-speeches 30
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.chamber import (
    chamber_density_summary,
    paired_chamber_test,
    speaker_chamber_density,
)
from src.analysis.labels_pl import pl_name
from src.analysis.speakers import (
    speaker_density,
    split_half_reliability,
    variance_share_by_party,
)
from src.analysis.statistical_tests import run_all_chi_squared
from src.analysis.temporal import aggregate_by_metadata_key
from src.analysis.visualizations import (
    plot_grouped_bars,
    plot_paired_chamber,
    plot_ranking_comparison,
    plot_speaker_density_by_party,
)
from src.data.loader import load_corpus
from src.schema.models import Prediction

console = Console()

CHAMBER_PL = {"sejm": "Sejm", "senat": "Senat"}


def load_aligned(corpus_path: Path, pred_path: Path, exclude_parties: set[str]):
    """Load corpus and predictions, keeping only speeches present in both."""
    preds: dict[str, Prediction] = {}
    with open(pred_path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                pred = Prediction.model_validate(json.loads(line))
                preds[pred.speech_id] = pred

    speeches, aligned = [], []
    for speech in load_corpus(corpus_path):
        pred = preds.get(speech.speech_id)
        if pred is None or speech.party in exclude_parties:
            continue
        speeches.append(speech)
        aligned.append(pred)
    return speeches, aligned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus/speeches.jsonl")
    parser.add_argument("--predictions", default="data/predictions/corpus_full.jsonl")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--exclude-parties", nargs="*", default=["unknown"])
    parser.add_argument("--min-speaker-speeches", type=int, default=50)
    parser.add_argument("--min-paired-speeches", type=int, default=20)
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--drop-top-speakers", type=int, default=10,
                        help="Robustness check: busiest speakers per chamber to drop.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[bold]Loading {args.predictions}...[/bold]")
    speeches, preds = load_aligned(
        Path(args.corpus), Path(args.predictions), set(args.exclude_parties)
    )
    console.print(f"  {len(speeches):,} speeches with predictions")

    # --- Chamber, speaking members only -------------------------------------
    kept = [(s, p) for s, p in zip(speeches, preds)
            if (s.metadata or {}).get("speaker_role") == "Parlamentarzysta"]
    mp_speeches = [s for s, _ in kept]
    mp_preds = [p for _, p in kept]
    console.print(f"  role=Parlamentarzysta: {len(mp_speeches):,}")

    chamber_df = aggregate_by_metadata_key(mp_speeches, mp_preds, "chamber")
    chamber_df["chamber"] = chamber_df["chamber"].map(lambda c: CHAMBER_PL.get(c, c))
    chamber_df.to_csv(out_dir / "chamber_by_label.csv", index=False)

    chi_df = run_all_chi_squared(chamber_df, group_col="chamber")
    chi_df.to_csv(out_dir / "chamber_chi_squared.csv", index=False)

    plot_grouped_bars(
        chamber_df,
        out_dir / "chamber_by_label.png",
        group_col="chamber",
        title="Częstość etykiet w Sejmie i w Senacie (wystąpienia parlamentarzystów)",
        legend_title="Izba",
    )

    table = Table(title="Sejm vs Senat (rola Parlamentarzysta)")
    for column in ("Etykieta", "Sejm %", "Senat %", "V Craméra", "p"):
        table.add_column(column, justify="right" if column != "Etykieta" else "left")
    rates = chamber_df.set_index("chamber")
    for _, row in chi_df.sort_values("cramers_v", ascending=False).iterrows():
        table.add_row(
            pl_name(row["label"]),
            f"{rates.loc['Sejm', row['label']] * 100:.2f}",
            f"{rates.loc['Senat', row['label']] * 100:.2f}",
            f"{row['cramers_v']:.3f}",
            f"{row['p_value']:.2g}",
        )
    console.print(table)

    summary = pd.concat([
        chamber_density_summary(mp_speeches, mp_preds, roles=None, drop_top_speakers=drop)
        for drop in (0, args.drop_top_speakers)
    ], ignore_index=True)
    summary.to_csv(out_dir / "chamber_density.csv", index=False)
    console.print("\n[bold]Chamber density[/bold] (second block drops the busiest speakers)")
    for _, row in summary.iterrows():
        console.print(
            f"  {row['chamber']:6s} drop={int(row['dropped_speakers']):2d} "
            f"n={int(row['n_speeches']):6d} {row['labels_per_speech']:.3f}/speech "
            f"{row['labels_per_1000_chars']:.3f}/1000 chars"
        )

    # --- Chamber, same speakers in both -------------------------------------
    paired = speaker_chamber_density(
        mp_speeches, mp_preds, roles=None,
        min_speeches_per_chamber=args.min_paired_speeches,
    )
    paired.to_csv(out_dir / "chamber_paired_speakers.csv", index=False)
    test = paired_chamber_test(paired)
    console.print("\n[bold]Speakers present in both chambers[/bold]")
    console.print(
        f"  n={test.get('n', 0)}  Sejm {test.get('mean_sejm', 0):.3f} -> "
        f"Senat {test.get('mean_senat', 0):.3f} labels/1000 chars; "
        f"lower in Senat for {test.get('n_lower_in_senat', 0)}; "
        f"Wilcoxon p={test.get('p_value', float('nan')):.3g}"
    )
    for direction in ("senat_later", "sejm_later"):
        entry = test.get(direction, {})
        console.print(
            f"  {direction}: n={entry.get('n', 0)}, "
            f"lower in Senat for {entry.get('n_lower_in_senat', 0)}, "
            f"p={entry.get('p_value', float('nan')):.3g}"
        )
    plot_paired_chamber(paired, out_dir / "chamber_paired.png")

    # --- Speakers ------------------------------------------------------------
    density = speaker_density(
        speeches, preds,
        min_speeches=args.min_speaker_speeches,
        min_chars=args.min_chars,
    )
    density.to_csv(out_dir / "speaker_density.csv", index=False)
    reliability = split_half_reliability(density)
    variance = variance_share_by_party(density)
    console.print(
        f"\n[bold]Speakers[/bold]: {len(density)} with >= {args.min_speaker_speeches} speeches; "
        f"split-half rho={reliability.get('spearman', float('nan')):.3f} "
        f"(Spearman-Brown {reliability.get('spearman_brown', float('nan')):.3f}); "
        f"party eta^2={variance.get('eta_squared', float('nan')):.3f}"
    )

    rank_shift = (density["labels_per_speech"].rank(ascending=False)
                  - density["labels_per_1000_chars"].rank(ascending=False))
    console.print("  largest rank shifts between the two indices:")
    for idx in rank_shift.abs().nlargest(5).index:
        row = density.loc[idx]
        console.print(
            f"    {row['speaker']:32s} {row['labels_per_speech']:.2f}/speech "
            f"{row['labels_per_1000_chars']:.2f}/1000 chars "
            f"(median {row['median_chars']:.0f} chars)"
        )

    plot_speaker_density_by_party(density, out_dir / "speaker_density_by_party.png")
    plot_ranking_comparison(density, out_dir / "speaker_ranking_comparison.png")

    console.print(f"\n[green]Written to {out_dir}[/green]")


if __name__ == "__main__":
    main()
