#!/usr/bin/env python3
"""
Współczynnik zgodności między adnotatorami (Inter-Annotator Agreement)
dla KATEGORII TEMATYCZNYCH
======================================================================
Porównuje dwa pliki CSV z adnotacjami w formacie:
    id;kategorie
    ID;KATEGORIA1,KATEGORIA2,KATEGORIA3

Oblicza:
  - Cohen's Kappa (per etykieta + średnia)
  - Krippendorff's Alpha (per etykieta + średnia)
  - Procent zgodności (per etykieta + średnia)
  - Raport rozbieżności

Użycie:
    python main_tematyczne.py plik_annotator1.csv plik_annotator2.csv
"""

import sys
import csv
from collections import defaultdict

# ---------------------------------------------------------------------------
# Metryki – implementacja bez zewnętrznych zależności (sklearn/krippendorff)
# ---------------------------------------------------------------------------

def cohens_kappa(y1: list[int], y2: list[int]) -> float:
    """Cohen's Kappa dla dwóch list binarnych etykiet."""
    assert len(y1) == len(y2), "Listy muszą mieć tę samą długość"
    n = len(y1)
    if n == 0:
        return float("nan")

    # Macierz pomyłek 2×2
    a = sum(1 for a, b in zip(y1, y2) if a == 1 and b == 1)  # oba TAK
    b = sum(1 for a, b in zip(y1, y2) if a == 1 and b == 0)  # A=TAK, B=NIE
    c = sum(1 for a, b in zip(y1, y2) if a == 0 and b == 1)  # A=NIE, B=TAK
    d = sum(1 for a, b in zip(y1, y2) if a == 0 and b == 0)  # oba NIE

    po = (a + d) / n  # obserwowana zgodność
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)  # oczekiwana

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def krippendorffs_alpha_binary(coders: list[list[int]]) -> float:
    """
    Krippendorff's Alpha dla danych binarnych (nominalnych, 2 kodery).
    coders: lista list – coders[c][i] = wartość kodera c dla elementu i.
    """
    n_items = len(coders[0])
    n_coders = len(coders)
    if n_items == 0:
        return float("nan")

    # Macierz zbieżności (coincidence matrix) dla wartości {0, 1}
    values = [0, 1]
    coincidence = defaultdict(float)
    for i in range(n_items):
        codes = [coders[c][i] for c in range(n_coders)]
        m = len(codes)  # koderzy na ten element
        if m < 2:
            continue
        for ci in range(m):
            for cj in range(m):
                if ci != cj:
                    coincidence[(codes[ci], codes[cj])] += 1.0 / (m - 1)

    total = sum(coincidence.values())
    if total == 0:
        return 1.0

    # Marginesy
    margin = defaultdict(float)
    for (v1, v2), val in coincidence.items():
        margin[v1] += val

    # Do – obserwowana niezgodność, De – oczekiwana niezgodność
    do_val = 0.0
    de_val = 0.0
    for v1 in values:
        for v2 in values:
            if v1 != v2:
                do_val += coincidence.get((v1, v2), 0.0)
                de_val += margin.get(v1, 0.0) * margin.get(v2, 0.0)

    if de_val == 0:
        return 1.0

    de_val /= (total - 1) if total > 1 else 1
    return 1.0 - (do_val / de_val) if de_val != 0 else 1.0


def percent_agreement(y1: list[int], y2: list[int]) -> float:
    """Prosty procent zgodności."""
    if len(y1) == 0:
        return float("nan")
    return sum(a == b for a, b in zip(y1, y2)) / len(y1)


# ---------------------------------------------------------------------------
# Parsowanie plików
# ---------------------------------------------------------------------------

def load_annotations(filepath: str) -> dict[str, set]:
    """
    Wczytuje plik CSV i zwraca dict: id → set kategorii tematycznych.
    Format pliku: id;KATEGORIA1,KATEGORIA2,KATEGORIA3
    """
    annotations = {}
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader, None)  # pomijamy nagłówek
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            uid = row[0].strip()
            raw_categories = row[1].strip()

            # Parsuj kategorie rozdzielone przecinkami
            if raw_categories:
                categories = set(cat.strip() for cat in raw_categories.split(",") if cat.strip())
            else:
                categories = set()

            annotations[uid] = categories
    return annotations


# ---------------------------------------------------------------------------
# Główna logika
# ---------------------------------------------------------------------------

def build_binary_vectors(ann1: dict[str, set], ann2: dict[str, set]):
    """
    Tworzy binarne wektory per etykieta.
    Zwraca: {label: (vec_a1, vec_a2)}, common_ids
    """
    common_ids = sorted(set(ann1.keys()) & set(ann2.keys()))
    if not common_ids:
        return {}, common_ids

    # Zbierz wszystkie możliwe etykiety
    all_labels = set()
    for uid in common_ids:
        all_labels |= ann1[uid]
        all_labels |= ann2[uid]

    vectors = {}
    for label in sorted(all_labels):
        v1 = [1 if label in ann1[uid] else 0 for uid in common_ids]
        v2 = [1 if label in ann2[uid] else 0 for uid in common_ids]
        vectors[label] = (v1, v2)

    return vectors, common_ids


def find_disagreements(ann1: dict[str, set], ann2: dict[str, set], common_ids: list):
    """Zwraca listę (id, tylko_w_ann1, tylko_w_ann2) gdzie wystąpiły rozbieżności."""
    disagreements = []
    for uid in common_ids:
        s1 = ann1[uid]
        s2 = ann2[uid]
        if s1 != s2:
            disagreements.append((uid, s1 - s2, s2 - s1))
    return disagreements


def interpret_kappa(k: float) -> str:
    if k < 0:    return "brak zgodności"
    if k < 0.20: return "słaba"
    if k < 0.40: return "dostateczna"
    if k < 0.60: return "umiarkowana"
    if k < 0.80: return "dobra"
    return "bardzo dobra / doskonała"


def main():
    if len(sys.argv) < 3:
        print("Użycie: python main_tematyczne.py <plik1.csv> <plik2.csv>")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    print(f"{'='*72}")
    print("  WSPÓŁCZYNNIK ZGODNOŚCI MIĘDZY ADNOTATORAMI")
    print("  (Kategorie tematyczne)")
    print(f"{'='*72}")
    print(f"  Plik annotatora 1: {file1}")
    print(f"  Plik annotatora 2: {file2}")
    print()

    ann1 = load_annotations(file1)
    ann2 = load_annotations(file2)

    ids1 = set(ann1.keys())
    ids2 = set(ann2.keys())
    common = ids1 & ids2
    only1 = ids1 - ids2
    only2 = ids2 - ids1

    print(f"  Elementy w pliku 1:        {len(ids1)}")
    print(f"  Elementy w pliku 2:        {len(ids2)}")
    print(f"  Wspólne elementy (do porównania): {len(common)}")
    if only1:
        print(f"  Tylko w pliku 1: {len(only1)}  (np. {list(only1)[:3]})")
    if only2:
        print(f"  Tylko w pliku 2: {len(only2)}  (np. {list(only2)[:3]})")
    print()

    if not common:
        print("  ❌ Brak wspólnych ID – nie można obliczyć zgodności.")
        sys.exit(1)

    print(f"{'─'*72}")
    print(f"  KATEGORIE TEMATYCZNE")
    print(f"{'─'*72}")

    vectors, common_ids = build_binary_vectors(ann1, ann2)

    if not vectors:
        print("  Brak etykiet w tej kategorii.\n")
        sys.exit(0)

    kappas = []
    alphas = []
    agreements = []

    print(f"  {'Etykieta':<40} {'Kappa':>7} {'Alpha':>7} {'Zgod.%':>7}  Interpretacja")
    print(f"  {'─'*40} {'─'*7} {'─'*7} {'─'*7}  {'─'*20}")

    for label in sorted(vectors.keys()):
        v1, v2 = vectors[label]
        k = cohens_kappa(v1, v2)
        a = krippendorffs_alpha_binary([v1, v2])
        p = percent_agreement(v1, v2)

        kappas.append(k)
        alphas.append(a)
        agreements.append(p)

        interp = interpret_kappa(k)
        print(f"  {label:<40} {k:>7.3f} {a:>7.3f} {p*100:>6.1f}%  {interp}")

    # Średnie
    avg_k = sum(kappas) / len(kappas) if kappas else float("nan")
    avg_a = sum(alphas) / len(alphas) if alphas else float("nan")
    avg_p = sum(agreements) / len(agreements) if agreements else float("nan")

    print(f"  {'─'*40} {'─'*7} {'─'*7} {'─'*7}")
    print(f"  {'ŚREDNIA':<40} {avg_k:>7.3f} {avg_a:>7.3f} {avg_p*100:>6.1f}%  {interpret_kappa(avg_k)}")
    print()

    # Ocena ogólna
    if avg_k >= 0.7:
        print(f"  ✅ Zgodność WYSOKA (κ={avg_k:.3f} ≥ 0.7) → zbiór WIARYGODNY")
    elif avg_k >= 0.4:
        print(f"  ⚠️  Zgodność UMIARKOWANA (κ={avg_k:.3f}) → zalecana rewizja rozbieżności")
    else:
        print(f"  ❌ Zgodność NISKA (κ={avg_k:.3f} < 0.4) → wymagana ponowna adnotacja")
    print()

    # Rozbieżności
    disagreements = find_disagreements(ann1, ann2, sorted(common))
    if disagreements:
        print(f"  📋 Rozbieżności ({len(disagreements)} z {len(common)} elementów, "
              f"{len(disagreements)/len(common)*100:.1f}%):")
        print()
        shown = 0
        for uid, only_a1, only_a2 in disagreements:
            if shown >= 30:
                remaining = len(disagreements) - shown
                print(f"     ... i {remaining} więcej rozbieżności (pełna lista w eksporcie)")
                break
            print(f"    ID: {uid}")
            if only_a1:
                print(f"      Tylko annotator 1: {', '.join(sorted(only_a1))}")
            if only_a2:
                print(f"      Tylko annotator 2: {', '.join(sorted(only_a2))}")
            shown += 1
        print()

    # ---------------------------------------------------------------------------
    # Eksport rozbieżności do CSV
    # ---------------------------------------------------------------------------
    out_path = "rozbieznosci_tematyczne.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "annotator_1", "annotator_2",
                     "tylko_annotator_1", "tylko_annotator_2"])
        for uid in sorted(common):
            s1 = ann1[uid]
            s2 = ann2[uid]
            if s1 != s2:
                w.writerow([
                    uid,
                    ", ".join(sorted(s1)),
                    ", ".join(sorted(s2)),
                    ", ".join(sorted(s1 - s2)),
                    ", ".join(sorted(s2 - s1)),
                ])
    print(f"  📄 Pełna lista rozbieżności zapisana do: {out_path}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()

