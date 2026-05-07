"""
Ewaluacja baseline – klasyfikacja wieloetykietowa wypowiedzi sejmowych.
Użycie:
    python ewaluacja.py --predykcje predykcje_llama3.1_8b.tsv
"""

import argparse
import sys
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score

# ── Argumenty ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--predykcje",      default="predykcje_llama3.1_8b.tsv",
                    help="Plik TSV z predykcjami modelu (kolumny: id, predykcja)")
parser.add_argument("--zloty_standard", default="data/złoty_standard.txt",
                    help="Plik TSV ze złotym standardem (kolumny: id, Decyzja)")
args = parser.parse_args()

print(f"Plik predykcji:      {args.predykcje}")
print(f"Złoty standard:      {args.zloty_standard}")
print()

# ── Wczytanie plików z jawnym UTF-8 ──────────────────────────────────────────
def wczytaj_tsv(sciezka: str, opis: str) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "cp1250"):
        try:
            df = pd.read_csv(sciezka, sep="\t", encoding=enc)
            print(f"[OK] {opis}: {len(df)} wierszy, kodowanie {enc}")
            print(f"     Kolumny: {df.columns.tolist()}")
            print(f"     Pierwsze wartości: {df.iloc[0].tolist()}")
            print()
            return df
        except Exception:
            continue
    print(f"[BŁĄD] Nie można wczytać pliku: {sciezka}")
    sys.exit(1)

predykcje = wczytaj_tsv(args.predykcje,      "Predykcje modelu")
prawdziwe = wczytaj_tsv(args.zloty_standard, "Złoty standard")

# ── Automatyczne wykrycie kolumny z etykietami ────────────────────────────────
def znajdz_kolumne_etykiet(df: pd.DataFrame, kandydaci: list[str]) -> str:
    for k in kandydaci:
        if k in df.columns:
            return k
    print(f"[BŁĄD] Nie znaleziono kolumny etykiet. Dostępne: {df.columns.tolist()}")
    sys.exit(1)

kol_pred  = znajdz_kolumne_etykiet(predykcje, ["predykcja", "kategorie", "labels"])
kol_true  = znajdz_kolumne_etykiet(prawdziwe, ["Decyzja", "kategorie", "labels", "predykcja"])

# ── Łączenie po ID ────────────────────────────────────────────────────────────
predykcje["id"] = predykcje["id"].str.strip()
prawdziwe["id"] = prawdziwe["id"].str.strip()

df = predykcje[["id", kol_pred]].merge(
    prawdziwe[["id", kol_true]], on="id", how="inner"
)

print(f"Dopasowanych rekordów: {len(df)} "
      f"(predykcje: {len(predykcje)}, złoty standard: {len(prawdziwe)})")

if len(df) == 0:
    print("\n[BŁĄD] Brak wspólnych ID! Sprawdź czy oba pliki dotyczą tych samych danych.")
    print("Przykład ID z predykcji:     ", predykcje["id"].iloc[0])
    print("Przykład ID ze złotego std.: ", prawdziwe["id"].iloc[0])
    sys.exit(1)

# ── Parsowanie etykiet ────────────────────────────────────────────────────────
def split_kat(s) -> list[str]:
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [k.strip().upper() for k in str(s).split(",") if k.strip()]

y_pred = df[kol_pred].apply(split_kat).tolist()
y_true = df[kol_true].apply(split_kat).tolist()

# ── Diagnostyka przed ewaluacją ───────────────────────────────────────────────
wszystkie_pred  = sorted(set(k for row in y_pred for k in row))
wszystkie_true  = sorted(set(k for row in y_true for k in row))
tylko_w_pred    = set(wszystkie_pred) - set(wszystkie_true)
tylko_w_true    = set(wszystkie_true) - set(wszystkie_pred)

print(f"\nEtykiety w złotym standardzie ({len(wszystkie_true)}): {wszystkie_true}")
print(f"Etykiety w predykcjach ({len(wszystkie_pred)}):        {wszystkie_pred}")
if tylko_w_pred:
    print(f"\n[UWAGA] Etykiety TYLKO w predykcjach (nie ma ich w złotym std.): {tylko_w_pred}")
if tylko_w_true:
    print(f"[UWAGA] Etykiety TYLKO w złotym std. (model ich nie przewidział): {tylko_w_true}")

# ── Binaryzacja i metryki ─────────────────────────────────────────────────────
mlb = MultiLabelBinarizer()
mlb.fit(y_true + y_pred)
Y_true = mlb.transform(y_true)
Y_pred = mlb.transform(y_pred)

print("\n" + "=" * 60)
print("WYNIKI EWALUACJI BASELINE")
print("=" * 60)
print(f"\nModel / plik: {args.predykcje}")
print(f"Próbka:       {len(df)} wypowiedzi\n")
print(f"Micro F1:  {f1_score(Y_true, Y_pred, average='micro', zero_division=0):.3f}")
print(f"Macro F1:  {f1_score(Y_true, Y_pred, average='macro', zero_division=0):.3f}")
print(f"\nSzczegóły per kategoria:")
print(classification_report(
    Y_true, Y_pred,
    target_names=mlb.classes_,
    zero_division=0
))