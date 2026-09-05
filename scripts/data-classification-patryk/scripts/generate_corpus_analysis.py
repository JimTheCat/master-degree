"""Run political-group analyses on full-corpus predictions.

Reads:
  - Corpus speeches (with party/date/role metadata): data/corpus/speeches.jsonl
  - Predictions: data/predictions/corpus_full.jsonl

Writes to results/analysis/: per-party frequencies, speaker rankings, temporal
dynamics, coalition-vs-opposition and per-term comparisons, chi-squared tests,
the accompanying PNGs, and a Polish-language REPORT.md tying them together.

Usage:
    python scripts/generate_corpus_analysis.py
    python scripts/generate_corpus_analysis.py --period-freq Y --skip-speakers
    python scripts/generate_corpus_analysis.py --predictions data/predictions/corpus_full.jsonl
"""

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.analysis.labels_pl import pl_name, pl_names
from src.analysis.political_groups import aggregate_by_party, compute_rhetoric_profile
from src.analysis.speakers import aggregate_by_speaker
from src.analysis.statistical_tests import run_all_chi_squared
from src.analysis.temporal import (
    aggregate_by_metadata_key,
    aggregate_by_period,
    aggregate_by_period_and_party,
)
from src.analysis.visualizations import (
    plot_confusion_heatmap,
    plot_grouped_bars,
    plot_party_rhetoric_profile,
    plot_radar_chart,
    plot_speaker_ranking,
    plot_temporal_by_party,
    plot_temporal_heatmap,
    plot_temporal_lines,
)
from src.data.loader import load_corpus
from src.schema.labels import ALL_LABELS
from src.schema.models import Prediction

console = Console()

# Ranking charts: (metric column, output file, chart title, x-axis label).
SPEAKER_RANKINGS = [
    ("emotion_rate", "ranking_emocje.png",
     "Ranking mówców: nasycenie emocjonalne", "Średnia liczba etykiet emocji na wypowiedź"),
    ("rhetoric_rate", "ranking_retoryka.png",
     "Ranking mówców: techniki retoryczne", "Średnia liczba technik retorycznych na wypowiedź"),
    ("AGRESJA_WERBALNA", "ranking_agresja.png",
     "Ranking mówców: agresja werbalna", "Odsetek wypowiedzi z agresją werbalną"),
]

# Temporal per-party chart: label plotted, output file.
PARTY_TREND_LABEL = "POLARYZACJA_MY_ONI"

# ParlaMint ships party_status in English; the report is Polish.
GROUP_VALUE_PL = {"Coalition": "Koalicja", "Opposition": "Opozycja"}

# "8. kadencja Sejmu" -> ("8", "Sejmu"). Term numbering restarts per chamber, so
# "9. kadencja" alone is ambiguous: Senat 9. ran 2015-2019, Sejm 9. ran 2019-2022.
TERM_PATTERN = re.compile(r"^(\d+)\.\s*kadencja\s+(\w+)", re.IGNORECASE)
CHAMBER_PL = {"sejmu": "Sejm", "senatu": "Senat"}


def relabel_term(term: str, first_date: str, last_date: str) -> str:
    """Make a term label self-explanatory: 'Sejm 9. (2019–2022)'."""
    match = TERM_PATTERN.match(term)
    years = f"{first_date[:4]}–{last_date[:4]}"
    if not match:
        return f"{term} ({years})"
    number, chamber = match.groups()
    return f"{CHAMBER_PL.get(chamber.lower(), chamber)} {number}. ({years})"

# Metadata dimensions compared as grouped bars: (metadata key, file stem,
# chart/section title, legend title).
GROUP_COMPARISONS = [
    ("party_status", "coalition_vs_opposition", "Koalicja vs opozycja", "Status"),
    ("term", "by_term", "Kadencje i izby", "Kadencja"),
]


def load_predictions(path: Path) -> dict[str, Prediction]:
    preds: dict[str, Prediction] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            pred = Prediction.model_validate(data)
            preds[pred.speech_id] = pred
    return preds


def align_corpus_and_predictions(corpus_path: Path, predictions: dict[str, Prediction]):
    """Yield (speech, prediction) pairs in matching order, dropping unmatched."""
    speeches = []
    aligned_preds = []
    n_unmatched = 0
    for speech in load_corpus(corpus_path):
        pred = predictions.get(speech.speech_id)
        if pred is None:
            n_unmatched += 1
            continue
        speeches.append(speech)
        aligned_preds.append(pred)
    return speeches, aligned_preds, n_unmatched


def lookup_test_scores(model_name: str, variant: str) -> dict[str, float] | None:
    """Find gold-standard F1 for the model behind the predictions.

    Scans results/sweep_summary_*.csv newest-first so the Limitations section
    can state how reliable the corpus-wide rates actually are.
    """
    for path in sorted(glob.glob("results/sweep_summary_*.csv"), reverse=True):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "macro_f1" not in df.columns:
            continue
        match = df[
            (df.get("model").astype(str) == model_name)
            & (df.get("variant").astype(str) == variant)
            & df["macro_f1"].notna()
            & (df["macro_f1"] > 0)
        ]
        if not match.empty:
            row = match.iloc[0]
            return {
                "source": Path(path).name,
                "macro_f1": float(row["macro_f1"]),
                "emotions_macro_f1": float(row["emotions_macro_f1"]),
                "rhetorical_macro_f1": float(row["rhetorical_macro_f1"]),
            }
    return None


def label_rate_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Long-form table of label rates per group, ready for to_markdown."""
    label_cols = [c for c in ALL_LABELS if c in df.columns]
    out = df.set_index(group_col)[label_cols].T
    out.index = pl_names(out.index)
    out.index.name = "Etykieta"
    return (out * 100).round(2)


def write_report(
    out_dir: Path,
    sections: list[str],
    written: list[str],
):
    lines = ["# Raport z analizy korpusu", ""]
    lines.extend(sections)
    lines.append("## Pliki wyjściowe")
    lines.append("")
    for name in written:
        lines.append(f"- `{name}`")
    lines.append("")
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Generate corpus-wide political analysis")
    parser.add_argument("--corpus", default="data/corpus/speeches.jsonl")
    parser.add_argument("--predictions", default="data/predictions/corpus_full.jsonl")
    parser.add_argument("--output-dir", default="results/analysis")
    parser.add_argument("--min-party-speeches", type=int, default=50,
                        help="Drop parties with fewer speeches before stat tests")
    parser.add_argument("--exclude-parties", nargs="*", default=["unknown"],
                        help="Parties dropped from all analyses (default: unknown, "
                             "i.e. chair/procedural speakers without party metadata)")
    parser.add_argument("--min-speaker-speeches", type=int, default=50,
                        help="Drop speakers with fewer qualifying speeches from rankings")
    parser.add_argument("--min-speech-chars", type=int, default=200,
                        help="Drop speeches shorter than this from speaker rankings")
    parser.add_argument("--top-n", type=int, default=20, help="Speakers shown per ranking chart")
    parser.add_argument("--period-freq", default="Q", choices=["Y", "Q", "M"],
                        help="Time granularity for the temporal section")
    parser.add_argument("--skip-speakers", action="store_true")
    parser.add_argument("--skip-temporal", action="store_true")
    parser.add_argument("--skip-groups", action="store_true")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    pred_path = Path(args.predictions)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not corpus_path.exists():
        console.print(f"[red]Corpus not found: {corpus_path}[/red]")
        sys.exit(1)
    if not pred_path.exists():
        console.print(f"[red]Predictions not found: {pred_path}[/red]")
        sys.exit(1)

    console.print(f"[bold]Loading predictions from {pred_path}...[/bold]")
    predictions = load_predictions(pred_path)
    console.print(f"  {len(predictions):,} predictions")

    console.print(f"[bold]Aligning with corpus {corpus_path}...[/bold]")
    speeches, aligned_preds, n_unmatched = align_corpus_and_predictions(corpus_path, predictions)
    console.print(f"  Aligned: {len(speeches):,}, Unmatched: {n_unmatched:,}")

    n_total = len(speeches)
    if args.exclude_parties:
        excluded = set(args.exclude_parties)
        kept = [(s, p) for s, p in zip(speeches, aligned_preds) if s.party not in excluded]
        n_excluded = len(speeches) - len(kept)
        speeches = [s for s, _ in kept]
        aligned_preds = [p for _, p in kept]
        console.print(f"  Excluded parties {sorted(excluded)}: {n_excluded:,} speeches dropped")

    if not speeches:
        console.print("[red]No aligned speeches — check corpus and predictions IDs[/red]")
        sys.exit(1)

    sample_pred = aligned_preds[0]
    role_counts = Counter((s.metadata or {}).get("speaker_role") for s in speeches)
    written: list[str] = []
    sections: list[str] = []

    # ---------------------------------------------------------------- summary
    sections.extend([
        "## Podsumowanie",
        "",
        f"- Model: `{sample_pred.model_name}`, wariant promptu: `{sample_pred.prompt_variant}`",
        f"- Wypowiedzi w korpusie z predykcją: {n_total:,} (niedopasowanych: {n_unmatched:,})",
        f"- Po odfiltrowaniu partii {sorted(args.exclude_parties)}: {len(speeches):,}",
        "- Role mówców: " + ", ".join(
            f"{role or 'brak'} {count:,}" for role, count in role_counts.most_common()
        ),
        "",
    ])

    # ------------------------------------------------------------ per party
    console.print("[bold]Aggregating by party...[/bold]")
    party_df = aggregate_by_party(speeches, aligned_preds)

    enough = party_df[party_df["total_speeches"] >= args.min_party_speeches].copy()
    console.print(
        f"  Parties: {len(party_df)}, "
        f"with ≥{args.min_party_speeches} speeches: {len(enough)}"
    )

    party_df.to_csv(out_dir / "party_label_frequencies.csv", index=False)
    written.append("party_label_frequencies.csv")

    console.print("[bold]Running chi-squared tests...[/bold]")
    chi_df = run_all_chi_squared(enough, group_col="party")
    chi_df.to_csv(out_dir / "chi_squared_results.csv", index=False)
    written.append("chi_squared_results.csv")

    if "significant" in chi_df.columns:
        n_sig = int(chi_df["significant"].sum())
        console.print(f"  Significant labels (p<0.05): {n_sig}/{len(chi_df)}")

    console.print("[bold]Generating visualizations...[/bold]")
    plot_party_rhetoric_profile(
        enough,
        out_dir / "party_rhetoric_stacked.png",
        group_col="party",
        title="Profil retoryczny wg partii",
    )
    written.append("party_rhetoric_stacked.png")
    console.print("  party_rhetoric_stacked.png")

    radar_df = enough.set_index("party")[ALL_LABELS].T
    radar_df.index = pl_names(radar_df.index)
    plot_radar_chart(
        radar_df,
        out_dir / "rhetoric_profiles.png",
        title="Częstość etykiet wg partii",
    )
    written.append("rhetoric_profiles.png")
    console.print("  rhetoric_profiles.png")

    cooc_df = pd.DataFrame(
        [{label: 1 if label in pred.labels else 0 for label in ALL_LABELS}
         for pred in aligned_preds]
    )
    cooc_df.columns = pl_names(cooc_df.columns)
    plot_confusion_heatmap(
        cooc_df,
        out_dir / "cooccurrence_heatmap.png",
        title="Współwystępowanie etykiet (korelacja Pearsona)",
    )
    written.append("cooccurrence_heatmap.png")
    console.print("  cooccurrence_heatmap.png")

    profile_df = compute_rhetoric_profile(enough)
    profile_df.to_csv(out_dir / "rhetoric_profiles_normalized.csv", index=False)
    written.append("rhetoric_profiles_normalized.csv")

    # fillna: rows where a test errored out carry NaN, not False.
    significant = (
        chi_df[chi_df["significant"].fillna(False).astype(bool)]
        .sort_values("cramers_v", ascending=False)
        if "significant" in chi_df.columns else pd.DataFrame()
    )
    sections.extend([
        "## Zróżnicowanie partyjne",
        "",
        f"Partie uwzględnione w testach (≥{args.min_party_speeches} wypowiedzi): "
        f"{', '.join(enough['party'].tolist())}",
        "",
        "### Etykiety najsilniej różnicujące partie (chi², wg Craméra V)",
        "",
    ])
    if significant.empty:
        sections.append("_Żadna etykieta nie osiągnęła istotności p<0.05._")
    else:
        table = significant[["label", "chi2", "p_value", "cramers_v"]].copy()
        table["label"] = table["label"].map(pl_name)
        table.columns = ["Etykieta", "chi²", "p", "Cramér V"]
        sections.append(table.to_markdown(index=False))
    sections.append("")

    # ---------------------------------------------------------- speakers
    if not args.skip_speakers:
        console.print("[bold]Aggregating by speaker...[/bold]")
        speaker_df = aggregate_by_speaker(
            speeches,
            aligned_preds,
            min_speeches=args.min_speaker_speeches,
            min_chars=args.min_speech_chars,
        )
        if speaker_df.empty:
            console.print("[yellow]  No speakers passed the filters — skipping rankings[/yellow]")
        else:
            console.print(
                f"  Speakers: {len(speaker_df)}, "
                f"qualifying speeches: {int(speaker_df['total_speeches'].sum()):,}"
            )
            speaker_df.to_csv(out_dir / "speaker_rankings.csv", index=False)
            written.append("speaker_rankings.csv")

            for metric, filename, title, xlabel in SPEAKER_RANKINGS:
                plot_speaker_ranking(
                    speaker_df,
                    out_dir / filename,
                    metric=metric,
                    top_n=args.top_n,
                    title=f"{title} (top {args.top_n})",
                    xlabel=xlabel,
                )
                written.append(filename)
                console.print(f"  {filename}")

            sections.extend([
                "## Ranking mówców",
                "",
                f"Filtry: rola `Parlamentarzysta`, ≥{args.min_speaker_speeches} wypowiedzi, "
                f"≥{args.min_speech_chars} znaków na wypowiedź. "
                f"Uwzględniono {len(speaker_df)} mówców "
                f"({int(speaker_df['total_speeches'].sum()):,} wypowiedzi).",
                "",
            ])
            for metric, label in (
                ("emotion_rate", "emocji"),
                ("rhetoric_rate", "technik retorycznych"),
            ):
                top = speaker_df.nlargest(10, metric)[
                    ["speaker", "party", "total_speeches", metric]
                ].copy()
                top[metric] = top[metric].round(3)
                top.columns = ["Mówca", "Partia", "Wypowiedzi", "Wskaźnik"]
                sections.extend([
                    f"### Top 10 — nasycenie {label} na wypowiedź",
                    "",
                    top.to_markdown(index=False),
                    "",
                ])

    # ---------------------------------------------------------- temporal
    if not args.skip_temporal:
        console.print("[bold]Aggregating over time...[/bold]")
        period_df = aggregate_by_period(speeches, aligned_preds, freq=args.period_freq)
        period_df.to_csv(out_dir / "temporal_periods.csv", index=False)
        written.append("temporal_periods.csv")
        console.print(
            f"  Periods: {len(period_df)} "
            f"({period_df['period'].iloc[0]} → {period_df['period'].iloc[-1]})"
        )

        top_labels = (
            period_df[ALL_LABELS].mean().nlargest(5).index.tolist()
        )
        plot_temporal_lines(
            period_df,
            out_dir / "temporal_lines.png",
            labels=top_labels,
            title="Dynamika 5 najczęstszych etykiet w czasie",
        )
        written.append("temporal_lines.png")
        plot_temporal_heatmap(
            period_df,
            out_dir / "temporal_heatmap.png",
            title="Nasilenie etykiet w czasie (z-score w obrębie etykiety)",
        )
        written.append("temporal_heatmap.png")
        console.print("  temporal_lines.png, temporal_heatmap.png")

        main_parties = enough.nlargest(5, "total_speeches")["party"].tolist()
        party_period_df = aggregate_by_period_and_party(
            speeches, aligned_preds, freq=args.period_freq, parties=main_parties
        )
        party_period_df.to_csv(out_dir / "temporal_by_party.csv", index=False)
        written.append("temporal_by_party.csv")
        plot_temporal_by_party(
            party_period_df,
            out_dir / "temporal_party_polaryzacja.png",
            label=PARTY_TREND_LABEL,
        )
        written.append("temporal_party_polaryzacja.png")
        console.print("  temporal_by_party.csv, temporal_party_polaryzacja.png")

        # Year-over-year change uses full calendar years only: the corpus starts
        # 2015-11-12 and ends 2022-06-30, so those two years are not comparable.
        year_df = aggregate_by_period(speeches, aligned_preds, freq="Y")
        full_years = year_df[~year_df["period"].isin({"2015", "2022"})]
        sections.extend([
            "## Dynamika czasowa",
            "",
            f"Granulacja: `{args.period_freq}`, zakres {period_df['period'].iloc[0]} – "
            f"{period_df['period'].iloc[-1]}. Wartości to odsetek wypowiedzi w okresie, "
            "nie liczby bezwzględne — 2015 (od 12 listopada) i 2022 (do 30 czerwca) "
            "są niepełne, więc zmiana rok-do-roku liczona jest na "
            f"{full_years['period'].iloc[0]}–{full_years['period'].iloc[-1]}.",
            "",
            "### Szczyt i zmiana w czasie dla każdej etykiety",
            "",
        ])
        rows = []
        first, last = full_years.iloc[0], full_years.iloc[-1]
        for label in ALL_LABELS:
            peak = full_years.loc[full_years[label].idxmax()]
            rows.append({
                "Etykieta": pl_name(label),
                "Rok szczytowy": peak["period"],
                "Szczyt %": round(peak[label] * 100, 2),
                f"{first['period']} %": round(first[label] * 100, 2),
                f"{last['period']} %": round(last[label] * 100, 2),
                "Zmiana p.p.": round((last[label] - first[label]) * 100, 2),
            })
        sections.append(
            pd.DataFrame(rows).sort_values("Zmiana p.p.", ascending=False).to_markdown(index=False)
        )
        sections.append("")

    # ------------------------------------------------ coalition / terms
    if not args.skip_groups:
        console.print("[bold]Aggregating by party status and term...[/bold]")
        for key, filename_stem, title, legend_title in GROUP_COMPARISONS:
            group_df = aggregate_by_metadata_key(speeches, aligned_preds, key)
            if group_df.empty:
                console.print(f"[yellow]  No '{key}' metadata — skipping[/yellow]")
                continue

            if key == "term":
                group_df[key] = [
                    relabel_term(row[key], row["first_date"], row["last_date"])
                    for _, row in group_df.iterrows()
                ]
            else:
                group_df[key] = group_df[key].replace(GROUP_VALUE_PL)
            # Re-sort on the display labels: the aggregation sorted the raw
            # values, where "10. kadencja Senatu" precedes "9. kadencja Sejmu".
            group_df = group_df.sort_values(["first_date", key]).reset_index(drop=True)
            coverage = int(group_df["total_speeches"].sum())
            group_df.to_csv(out_dir / f"{filename_stem}.csv", index=False)
            written.append(f"{filename_stem}.csv")
            plot_grouped_bars(
                group_df,
                out_dir / f"{filename_stem}.png",
                group_col=key,
                title=title,
                legend_title=legend_title,
            )
            written.append(f"{filename_stem}.png")

            group_chi = run_all_chi_squared(group_df, group_col=key)
            group_chi.to_csv(out_dir / f"chi_squared_by_{key}.csv", index=False)
            written.append(f"chi_squared_by_{key}.csv")
            console.print(f"  {filename_stem}.csv/.png, chi_squared_by_{key}.csv")

            rates = label_rate_table(group_df, key)
            if "cramers_v" in group_chi.columns:
                effect = group_chi.set_index("label")["cramers_v"]
                rates["Cramér V"] = [
                    round(float(effect.get(label, float("nan"))), 3) for label in ALL_LABELS
                ]
            sections.extend([
                f"## {title}",
                "",
                f"Pokrycie metadanych `{key}`: {coverage:,} z {len(speeches):,} wypowiedzi "
                f"({coverage / len(speeches):.1%}). Wartości w % wypowiedzi danej grupy.",
                "",
                rates.to_markdown(),
                "",
            ])

    # -------------------------------------------------------- limitations
    scores = lookup_test_scores(sample_pred.model_name, sample_pred.prompt_variant)
    sections.extend(["## Ograniczenia", ""])
    if scores:
        sections.append(
            f"Wszystkie powyższe wskaźniki pochodzą z predykcji jednego modelu "
            f"(`{sample_pred.model_name}`, `{sample_pred.prompt_variant}`), którego jakość "
            f"na zbiorze złotego standardu wynosi: macro-F1 **{scores['macro_f1']:.3f}** "
            f"(emocje {scores['emotions_macro_f1']:.3f}, retoryka "
            f"{scores['rhetorical_macro_f1']:.3f}; źródło: `{scores['source']}`)."
        )
    else:
        sections.append(
            f"Wskaźniki pochodzą z predykcji jednego modelu (`{sample_pred.model_name}`, "
            f"`{sample_pred.prompt_variant}`); nie znaleziono dla niego wyników na zbiorze "
            "testowym w `results/sweep_summary_*.csv`."
        )
    sections.extend([
        "",
        "Konsekwencje, które trzeba czytać razem z każdym wykresem:",
        "",
        "- Mierzone jest to, co model wykrywa, a nie zjawisko samo w sobie. Rankingi i "
        "różnice między grupami są rzetelne o tyle, o ile błąd modelu rozkłada się "
        "równomiernie między mówcami, partiami i latami — czego nie sprawdzono.",
        "- Etykiety o niższym F1 (zwłaszcza retoryczne) mają rankingi obarczone większym błędem "
        "niż etykiety emocji.",
        "- Liczba etykiet rośnie z długością wypowiedzi (średnio 0.18 dla wypowiedzi poniżej "
        f"500 znaków wobec 2.85 powyżej 3000). Filtr `--min-speech-chars {args.min_speech_chars}` "
        "i przeliczanie na wypowiedź ograniczają to obciążenie, ale go nie usuwają: mówcy "
        "wygłaszający dłuższe wystąpienia są systematycznie wyżej w rankingach.",
        "- Wypowiedzi o roli `Przewodniczący` (prowadzenie obrad) są wyłączone z rankingu mówców, "
        "ale wchodzą do analiz partyjnych i czasowych, gdzie zaniżają wskaźniki.",
        "",
    ])

    write_report(out_dir, sections, written)
    console.print(f"[green]Wrote {out_dir / 'REPORT.md'}[/green]")

    summary = Table(title="Per-party speech counts (top 10)")
    summary.add_column("Party", style="cyan")
    summary.add_column("Speeches", justify="right")
    for _, row in enough.sort_values("total_speeches", ascending=False).head(10).iterrows():
        summary.add_row(str(row["party"]), str(int(row["total_speeches"])))
    console.print(summary)


if __name__ == "__main__":
    main()
