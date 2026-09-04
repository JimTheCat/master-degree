"""
Przygotowanie danych do aplikacji przeglądowej.

Łączy anotacje z results/wyniki.jsonl z metadanymi ParlaMint i zapisuje dwa
pliki Parquet:

  dane/wypowiedzi.parquet   — jeden wiersz na wypowiedź (szeroki, do filtrów
                              i tabel); kategorie jako lista
  dane/kategorie.parquet    — jeden wiersz na parę wypowiedź–kategoria
                              (długi, do wykresów)

Uruchomienie:
    python przygotuj_dane.py
    python przygotuj_dane.py --parse-fail pomin     # zamiast domyślnego INNE
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PLIK_ANOTACJI = "results/wyniki.jsonl"
PLIK_METADANYCH = "../data/merged_all.tsv"
KATALOG_WY = "dane"

# Kolumny bez treści (Meeting, Agenda) lub stałe (Lang) — nie trafiają do wyniku.
KOLUMNY_DO_POMINIECIA = ["Meeting", "Agenda", "Lang", "Title"]

# W ParlaMint brak danych zapisany jest myślnikiem.
BRAK = "-"

NAZWY = {
    "ID": "id",
    "Text_ID": "posiedzenie_id",
    "Date": "data",
    "Body": "izba",
    "Term": "kadencja",
    "Session": "sesja",
    "Sitting": "dzien_sesji",
    "Subcorpus": "podkorpus",
    "Speaker_role": "rola",
    "Speaker_MP": "czy_posel",
    "Speaker_minister": "czy_minister",
    "Speaker_party": "partia",
    "Speaker_party_name": "partia_nazwa",
    "Party_status": "status_partii",
    "Party_orientation": "orientacja_partii",
    "Speaker_ID": "mowca_id",
    "Speaker_name": "mowca",
    "Speaker_gender": "plec",
    "Speaker_birth": "rok_urodzenia",
    "Topic": "temat_parlamint",
}

KUBELKI = [(0, 121), (121, 300), (300, 700), (700, 1200),
           (1200, 2000), (2000, 3500), (3500, 10 ** 9)]


def etykieta_kubelka(n: int) -> str:
    for lo, hi in KUBELKI:
        if lo <= n < hi:
            return f"{lo}–{hi - 1}" if hi < 10 ** 9 else f"{lo}+"
    return "?"


def wczytaj_anotacje(sciezka: str, parse_fail: str) -> pd.DataFrame:
    wiersze = []
    with open(sciezka, encoding="utf-8") as f:
        for linia in f:
            linia = linia.strip()
            if not linia:
                continue
            try:
                r = json.loads(linia)
            except json.JSONDecodeError:
                continue
            if not r.get("id"):
                continue  # nagłówek
            status = r.get("status", "OK")
            kategorie = list(r.get("kategorie") or [])

            if status == "PARSE_FAIL":
                if parse_fail == "inne":
                    kategorie = ["INNE"]
                elif parse_fail == "pomin":
                    continue
            elif status == "ERROR":
                continue

            wiersze.append({
                "id": r["id"],
                "kategorie": kategorie,
                "status": status,
                "znaki": r.get("znaki", 0),
                "obciety": bool(r.get("obciety", False)),
                "flagi": ",".join(r.get("flagi") or []),
            })
    return pd.DataFrame(wiersze)


def wczytaj_metadane(sciezka: str) -> pd.DataFrame:
    df = pd.read_csv(sciezka, sep="\t", dtype=str,
                     keep_default_na=False, na_values=[])
    df.columns = [c.strip() for c in df.columns]
    df = df.drop(columns=[c for c in KOLUMNY_DO_POMINIECIA if c in df.columns])
    df = df.rename(columns=NAZWY)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    return df


def wzbogac(df: pd.DataFrame) -> pd.DataFrame:
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["rok"] = df["data"].dt.year
    df["miesiac"] = df["data"].dt.to_period("M").dt.to_timestamp()
    df["kwartal"] = df["data"].dt.to_period("Q").dt.to_timestamp()

    # Brak danych: myślnik -> None, żeby wykresy nie traktowały go jak kategorii
    for c in ["partia", "partia_nazwa", "status_partii", "orientacja_partii",
              "rok_urodzenia"]:
        if c in df.columns:
            df[c] = df[c].replace(BRAK, None)

    df["rok_urodzenia"] = pd.to_numeric(df["rok_urodzenia"], errors="coerce")
    df["wiek"] = df["rok"] - df["rok_urodzenia"]

    # Term miesza izby (8./9. kadencja Sejmu, 9./10. Senatu) — łączymy jawnie.
    df["izba_kadencja"] = df["izba"] + " · " + df["kadencja"]

    df["prowadzi_obrady"] = df["rola"].eq("Przewodniczący")
    df["dlugosc"] = df["znaki"].map(etykieta_kubelka)
    df["n_kategorii"] = df["kategorie"].map(len)

    # Podkorpus 'COVID,War' zawiera oba oznaczenia — rozbijamy na flagi.
    df["okres_covid"] = df["podkorpus"].str.contains("COVID", na=False)
    df["okres_wojny"] = df["podkorpus"].str.contains("War", na=False)

    kolejnosc = [
        "id", "posiedzenie_id", "data", "rok", "miesiac", "kwartal",
        "izba", "kadencja", "izba_kadencja", "sesja", "dzien_sesji",
        "podkorpus", "okres_covid", "okres_wojny",
        "mowca_id", "mowca", "plec", "rok_urodzenia", "wiek",
        "rola", "prowadzi_obrady", "czy_posel", "czy_minister",
        "partia", "partia_nazwa", "status_partii", "orientacja_partii",
        "temat_parlamint",
        "kategorie", "n_kategorii", "status", "znaki", "dlugosc",
        "obciety", "flagi",
    ]
    return df[[c for c in kolejnosc if c in df.columns]]


def na_kategorie(df: pd.DataFrame) -> pd.DataFrame:
    """Format długi: jeden wiersz na parę wypowiedź–kategoria."""
    dlugi = df.explode("kategorie").rename(columns={"kategorie": "kategoria"})
    return dlugi[dlugi["kategoria"].notna()]


def skompresuj(df: pd.DataFrame) -> pd.DataFrame:
    """Kolumny o małej liczbie wartości -> category (mniejszy plik, szybsze filtry)."""
    for c in df.columns:
        if df[c].dtype == object and c not in ("id", "posiedzenie_id", "kategorie"):
            if df[c].nunique(dropna=True) < len(df) / 50:
                df[c] = df[c].astype("category")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--anotacje", default=PLIK_ANOTACJI)
    p.add_argument("--metadane", default=PLIK_METADANYCH)
    p.add_argument("--wyjscie", default=KATALOG_WY)
    p.add_argument("--parse-fail", choices=["inne", "pomin", "zostaw"],
                   default="inne",
                   help="co zrobić z rekordami PARSE_FAIL (domyślnie: INNE)")
    args = p.parse_args()

    print("Wczytuję anotacje…")
    anot = wczytaj_anotacje(args.anotacje, args.parse_fail)
    print(f"  rekordów: {len(anot):,}")
    print(f"  statusy : {dict(anot['status'].value_counts())}")

    print("Wczytuję metadane…")
    meta = wczytaj_metadane(args.metadane)
    print(f"  rekordów: {len(meta):,}")

    tylko_anot = set(anot["id"]) - set(meta["id"])
    tylko_meta = set(meta["id"]) - set(anot["id"])
    if tylko_anot:
        print(f"  [UWAGA] {len(tylko_anot):,} anotacji bez metadanych, np. "
              f"{list(tylko_anot)[:2]}")
    if tylko_meta:
        print(f"  [UWAGA] {len(tylko_meta):,} metadanych bez anotacji, np. "
              f"{list(tylko_meta)[:2]}")

    df = anot.merge(meta, on="id", how="inner")
    print(f"Po złączeniu: {len(df):,} wypowiedzi")

    df = wzbogac(df)
    dlugi = na_kategorie(df)

    kat_do_zapisu = df.copy()
    kat_do_zapisu["kategorie"] = kat_do_zapisu["kategorie"].map(
        lambda x: ",".join(x))

    wy = Path(args.wyjscie)
    wy.mkdir(parents=True, exist_ok=True)
    skompresuj(kat_do_zapisu).to_parquet(wy / "wypowiedzi.parquet",
                                         index=False, compression="zstd")
    skompresuj(dlugi.drop(columns=["flagi"])).to_parquet(
        wy / "kategorie.parquet", index=False, compression="zstd")

    print("\n" + "=" * 62)
    print(f"  wypowiedzi.parquet : {len(df):,} wierszy, "
          f"{(wy / 'wypowiedzi.parquet').stat().st_size / 1e6:.1f} MB")
    print(f"  kategorie.parquet  : {len(dlugi):,} wierszy, "
          f"{(wy / 'kategorie.parquet').stat().st_size / 1e6:.1f} MB")
    print("=" * 62)

    print(f"\n  zakres dat        : {df['data'].min().date()} … "
          f"{df['data'].max().date()}  ({df['miesiac'].nunique()} miesięcy)")
    print(f"  mówców            : {df['mowca_id'].nunique():,} "
          f"(≥100 wypowiedzi: {(df['mowca_id'].value_counts() >= 100).sum()})")
    print(f"  śr. etykiet       : {df['n_kategorii'].mean():.2f}")
    print(f"  prowadzący obrady : {df['prowadzi_obrady'].mean() * 100:.1f}% wypowiedzi")

    print("\n  udział INNE wg roli mówcy:")
    tylko_inne = df["kategorie"].map(lambda x: x == ["INNE"])
    for rola, udzial in df.assign(i=tylko_inne).groupby(
            "rola", observed=True)["i"].mean().sort_values(ascending=False).items():
        print(f"    {rola:<18} {udzial * 100:5.1f}%")

    print("\n  10 najczęstszych kategorii:")
    for kat, n in dlugi["kategoria"].value_counts().head(10).items():
        print(f"    {kat:<26} {n:>7,}  ({n / len(df) * 100:4.1f}% wypowiedzi)")


if __name__ == "__main__":
    main()
