"""Compute inter-annotator agreement (Cohen's kappa) for the emotion/rhetoric study.

The repo does not store both annotators' raw label sets, but they can be
reconstructed exactly:
  - data/gold_standard/speeches.jsonl holds the final (consensus) labels,
  - data/raw/rozbieznosci.csv lists, per (speech, dimension), each annotator's
    labels wherever they disagreed.
For every (speech, dimension) NOT listed in rozbieznosci.csv both annotators
agreed, so their labels equal the final gold labels for that dimension.

Outputs a per-label kappa / percent-agreement table to the console and
results/agreement_emotions_rhetoric.csv.

Usage:
    python scripts/compute_agreement.py
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.metrics import cohen_kappa_score

from src.schema.labels import ALL_LABELS, is_emotion

GOLD_PATH = Path("data/gold_standard/speeches.jsonl")
CONFLICTS_PATH = Path("data/raw/rozbieznosci.csv")
OUT_PATH = Path("results/agreement_emotions_rhetoric.csv")

# The annotation CSVs use Polish diacritics; the canonical taxonomy is ASCII.
_PL_ASCII = str.maketrans("ĄĆĘŁŃÓŚŹŻ", "ACELNOSZZ")

# The canonical id spells it OBLEZIONA (see src/schema/labels.py), while the
# annotation CSVs use the orthographic OBLĘŻONA -> ascii OBLEZONA.
_ALIASES = {"OBLEZONA_TWIERDZA": "OBLEZIONA_TWIERDZA"}

console = Console()


def normalize_label(raw: str) -> str | None:
    label = raw.strip().strip('"').upper().translate(_PL_ASCII)
    label = _ALIASES.get(label, label)
    if not label:
        return None
    if label not in ALL_LABELS:
        raise ValueError(f"Nieznana etykieta w rozbieznosci.csv: {raw!r} -> {label!r}")
    return label


def parse_label_list(cell: str) -> set[str]:
    labels = set()
    for part in cell.split(","):
        label = normalize_label(part)
        if label:
            labels.add(label)
    return labels


def dimension_of(label: str) -> str:
    return "emocje" if is_emotion(label) else "techniki_retoryczne"


def main() -> None:
    # Final gold labels per speech.
    gold: dict[str, set[str]] = {}
    with open(GOLD_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            ann = row.get("annotations") or {}
            gold[row["speech_id"]] = set(ann.get("labels") or [])

    # Conflicted (speech, dimension) cells with each annotator's labels.
    conflicts: dict[tuple[str, str], tuple[set[str], set[str]]] = {}
    with open(CONFLICTS_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            key = (row["id"], row["kategoria"])
            conflicts[key] = (
                parse_label_list(row.get("annotator_1") or ""),
                parse_label_list(row.get("annotator_2") or ""),
            )

    missing = {sid for sid, _ in conflicts} - set(gold)
    if missing:
        console.print(
            f"[yellow]WARN {len(missing)} speech ids z rozbieznosci.csv "
            f"nie ma w gold standard[/yellow]"
        )

    # Reconstruct full binary matrices for both annotators.
    speech_ids = sorted(gold)
    label_idx = {label: i for i, label in enumerate(ALL_LABELS)}
    a = np.zeros((len(speech_ids), len(ALL_LABELS)), dtype=int)
    b = np.zeros((len(speech_ids), len(ALL_LABELS)), dtype=int)

    for i, sid in enumerate(speech_ids):
        for dim in ("emocje", "techniki_retoryczne"):
            if (sid, dim) in conflicts:
                labels_a, labels_b = conflicts[(sid, dim)]
            else:
                agreed = {lab for lab in gold[sid] if dimension_of(lab) == dim}
                labels_a = labels_b = agreed
            for lab in labels_a:
                a[i, label_idx[lab]] = 1
            for lab in labels_b:
                b[i, label_idx[lab]] = 1

    # Per-label agreement.
    rows = []
    for label in ALL_LABELS:
        j = label_idx[label]
        col_a, col_b = a[:, j], b[:, j]
        observed = float((col_a == col_b).mean())
        if col_a.sum() == 0 and col_b.sum() == 0:
            kappa = float("nan")  # label never used by either annotator
        else:
            kappa = float(cohen_kappa_score(col_a, col_b))
        rows.append({
            "label": label,
            "dimension": dimension_of(label),
            "kappa": kappa,
            "percent_agreement": observed,
            "annotator_1_count": int(col_a.sum()),
            "annotator_2_count": int(col_b.sum()),
        })

    valid = [r for r in rows if not np.isnan(r["kappa"])]
    macro_kappa = float(np.mean([r["kappa"] for r in valid]))
    emo_kappas = [r["kappa"] for r in valid if r["dimension"] == "emocje"]
    rhe_kappas = [r["kappa"] for r in valid if r["dimension"] == "techniki_retoryczne"]

    conflicted_speeches = {sid for sid, _ in conflicts if sid in gold}

    def interp(k: float) -> str:
        if np.isnan(k):
            return "n/d"
        if k < 0.2:
            return "słaba"
        if k < 0.4:
            return "dostateczna"
        if k < 0.6:
            return "umiarkowana"
        if k < 0.8:
            return "dobra"
        return "bardzo dobra"

    table = Table(title="Zgodność między anotatorami (emocje + techniki retoryczne)")
    table.add_column("Etykieta", style="cyan")
    table.add_column("Kappa", justify="right")
    table.add_column("Zgodność %", justify="right")
    table.add_column("Anot. 1", justify="right")
    table.add_column("Anot. 2", justify="right")
    table.add_column("Interpretacja")
    for r in rows:
        table.add_row(
            r["label"],
            "n/d" if np.isnan(r["kappa"]) else f"{r['kappa']:.3f}",
            f"{r['percent_agreement'] * 100:.1f}%",
            str(r["annotator_1_count"]),
            str(r["annotator_2_count"]),
            interp(r["kappa"]),
        )
    console.print(table)
    console.print(
        f"\nŚrednia kappa (macro): [bold]{macro_kappa:.3f}[/bold] ({interp(macro_kappa)})"
    )
    console.print(
        f"  emocje: {np.mean(emo_kappas):.3f}   "
        f"techniki retoryczne: {np.mean(rhe_kappas):.3f}"
    )
    console.print(
        f"Wypowiedzi z co najmniej jednym konfliktem: {len(conflicted_speeches)}/{len(speech_ids)} "
        f"({len(conflicted_speeches) / len(speech_ids) * 100:.1f}%)"
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"\nZapisano: {OUT_PATH}")


if __name__ == "__main__":
    main()
