"""
Anotacja korpusu — przebieg produkcyjny.

Założenia (ustalone przed napisaniem, nie zmieniaj bez powodu):
  - przetwarzanie SEKWENCYJNE; benchmark wykazał, że LM Studio serializuje
    żądania, a współbieżność zmienia 3–9% anotacji przy zerowym zysku tempa
  - rozdzielone statusy OK / PARSE_FAIL / ERROR — brak cichego fallbacku do INNE
  - wznawianie po ID; PARSE_FAIL uznaje się za zrobione (temp. 0 → ta sama
    odpowiedź), ERROR wraca do kolejki
  - flush + fsync po każdym rekordzie; batch to rytm raportowania, nie trwałości
  - MLflow: jeden run, agregaty co batch, wszystko nieblokujące
  - Dysk Google: progress.json co batch, spakowany wynik co BATCH_BACKUP

Użycie:
    python klasyfikator.py                 # przebieg właściwy (wznawia sam)
    python klasyfikator.py --benchmark --limit 200
    python klasyfikator.py --eksport
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import re
import signal
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APITimeoutError

from kategorie import KATEGORIE, KATEGORIE_DEF, suma_kontrolna_promptu
from drive_sync import DriveSync
from mlflow_sink import MLflowSink

load_dotenv()


def _env(klucz: str, domyslna, typ=str):
    wartosc = os.environ.get(klucz)
    if wartosc is None or wartosc == "":
        return domyslna
    try:
        return typ(wartosc)
    except (TypeError, ValueError):
        return domyslna


# ─────────────────────────────────────────────────────────────── konfiguracja

MODEL = _env("AI_MODEL", "qwen3.5-9b@q6_k")
LM_STUDIO_URL = _env("LM_STUDIO_URL", "http://localhost:1234/v1")
METODA = _env("METODA_PROMPTU", "one_shot")
SEED = _env("SEED", 42, int)
TEMPERATURA = _env("TEMPERATURA", 0.0, float)
LIMIT_ZNAKOW = _env("LIMIT_ZNAKOW", 6000, int)
MAX_TOKENOW = _env("MAX_TOKENOW", 500, int)
MAX_PROB = _env("MAX_PROB", 4, int)

BATCH_RAPORT = _env("BATCH_RAPORT", 500, int)
BATCH_BACKUP = _env("BATCH_BACKUP", 5000, int)

PLIK_TEKSTOW = _env("PLIK_TEKSTOW", "./data/merged_all.txt")
PLIK_WYNIKOW = _env("PLIK_WYNIKOW", "results/wyniki.jsonl")
PLIK_BLEDOW = _env("PLIK_BLEDOW", "results/bledy.jsonl")
PLIK_POSTEPU = _env("PLIK_POSTEPU", "results/progress.json")
PLIK_EKSPORTU = _env("PLIK_EKSPORTU", "results/anotacja.txt")
PLIK_LOGU = _env("PLIK_LOGU", "results/klasyfikator.log")

MLFLOW_URI = _env("MLFLOW_URI", None)
MLFLOW_EKSPERYMENT = _env("MLFLOW_EKSPERYMENT", "anotacja_korpusu")
GDRIVE_SA_JSON = _env("GDRIVE_SA_JSON", None)
GDRIVE_FOLDER_ID = _env("GDRIVE_FOLDER_ID", None)
OCZEKIWANA_SUMA = _env("OCZEKIWANA_SUMA", None)

ODSTEP_SONDY = 30.0      # sekund między próbami wykrycia powrotu LM Studio
LOG_CZEKANIA_CO = 300.0  # sekund między wpisami w logu podczas czekania

# Kubełki długości. Pierwszy odpowiada stratum, którego nie ma w złotym
# korpusie (najkrótszy tekst ewaluowany miał 121 znaków) — jego statystyki
# raportujemy osobno.
KUBELKI = [(0, 121), (121, 300), (300, 700), (700, 1200),
           (1200, 2000), (2000, 3500), (3500, 10 ** 9)]

PRZERWANIE = False


# ───────────────────────────────────────────────────────────────── narzędzia

def teraz() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def bez_ogonkow(s: str) -> str:
    """Nazwy metryk MLflow bywają wrażliwe na znaki diakrytyczne."""
    mapa = str.maketrans("ĄĆĘŁŃÓŚŹŻąćęłńóśźż", "ACELNOSZZacelnoszz")
    return re.sub(r"[^A-Za-z0-9_.\-/]", "_", s.translate(mapa))


def kubelek(n: int) -> str:
    for lo, hi in KUBELKI:
        if lo <= n < hi:
            return f"{lo}_{hi}" if hi < 10 ** 9 else f"{lo}_plus"
    return "nieznany"


def ustaw_logowanie(sciezka: str) -> None:
    Path(sciezka).parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)-12s %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    plik = logging.FileHandler(sciezka, encoding="utf-8")
    plik.setFormatter(fmt)
    konsola = logging.StreamHandler(sys.stdout)
    konsola.setFormatter(fmt)
    # Konsola Windows potrafi paść na polskim znaku — nie pozwalamy jej ubić procesu
    if hasattr(konsola.stream, "reconfigure"):
        try:
            konsola.stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    root.addHandler(plik)
    root.addHandler(konsola)
    logging.getLogger("googleapiclient").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


log = logging.getLogger("klasyfikator")


# ──────────────────────────────────────────────────────────────────── prompt

def wczytaj_szablon(metoda: str, katalog: str = "../methods") -> str:
    sciezka = Path(katalog) / f"{metoda}.txt"
    if not sciezka.exists():
        raise FileNotFoundError(f"Brak pliku metody: {sciezka}")
    return sciezka.read_text(encoding="utf-8")


def buduj_prompt(tekst: str, szablon: str) -> str:
    definicje = "\n".join(f"- {k}: {v}" for k, v in KATEGORIE_DEF.items())
    return (
        szablon
        .replace("{tekst}", tekst[:LIMIT_ZNAKOW])
        .replace("{definicje}", definicje)
        .replace("{kategorie}", ", ".join(KATEGORIE))
    )


# ─────────────────────────────────────────────────────────────── parsowanie

def parsuj_odpowiedz(raw: str) -> tuple[list[str], str, list[str]]:
    """
    Zwraca (kategorie, status, flagi).

    Rdzeń jest identyczny z ewaluatorem. Dwie różnice, obie świadome:
      - brak fallbacku do INNE; pusty wynik to PARSE_FAIL
      - usuwanie bloków <think>, których ewaluator nie napotkał; gdyby się
        pojawiły, stary parser zwróciłby INNE bez śladu w logu
    """
    flagi: list[str] = []
    tekst = raw.strip()

    if "<think" in tekst.lower():
        flagi.append("think_usuniety")
        tekst = re.sub(r"(?is)<think>.*?</think>", " ", tekst)
        tekst = re.sub(r"(?is)<think>.*$", " ", tekst).strip()

    if tekst.startswith(("{", "[")):
        try:
            dane = json.loads(tekst)
            if isinstance(dane, dict):
                for v in dane.values():
                    if isinstance(v, str):
                        tekst = v
                        break
                    if isinstance(v, list):
                        tekst = ",".join(str(x) for x in v)
                        break
            elif isinstance(dane, list):
                tekst = ",".join(str(x) for x in dane)
        except json.JSONDecodeError:
            pass

    tekst = next((ln.strip() for ln in tekst.splitlines() if ln.strip()), "")

    kandydaci = [
        re.sub(r"[^A-ZĄĆĘŁŃÓŚŹŻ_]", "", tok.strip().upper())
        for tok in tekst.split(",")
    ]
    wynik = [k for k in kandydaci if k in KATEGORIE]

    bez_powtorzen = list(dict.fromkeys(wynik))
    if len(bez_powtorzen) != len(wynik):
        flagi.append("powtorzona_kategoria")
    wynik = bez_powtorzen

    if not wynik:
        return [], "PARSE_FAIL", flagi

    if "INNE" in wynik and len(wynik) > 1:
        # Zasada 3 promptu: INNE nie łączy się z niczym innym.
        wynik = [k for k in wynik if k != "INNE"]
        flagi.append("inne_z_innymi")

    return wynik, "OK", flagi


# ────────────────────────────────────────────────────────── wejście/wyjście

def wczytaj_teksty(sciezka: str, limit: int | None = None,
                   ziarno: int = 42) -> list[dict]:
    rekordy: list[dict] = []
    pominiete = 0
    with open(sciezka, encoding="utf-8") as f:
        for linia in f:
            linia = linia.rstrip("\r\n")
            if not linia.strip():
                pominiete += 1
                continue
            czesci = linia.split("\t", 1)
            if len(czesci) < 2:
                pominiete += 1
                continue
            rekordy.append({"id": czesci[0].strip(), "tekst": czesci[1].strip()})

    if pominiete:
        log.warning("Pominięto %d linii bez tabulatora lub pustych", pominiete)

    licznik = Counter(r["id"] for r in rekordy)
    duplikaty = {k: v for k, v in licznik.items() if v > 1}
    if duplikaty:
        raise SystemExit(
            f"BŁĄD: {len(duplikaty)} zduplikowanych ID w {sciezka} "
            f"(np. {list(duplikaty)[:3]}). Wznawianie po ID wymaga unikalności "
            f"— napraw plik wejściowy przed uruchomieniem."
        )

    if limit is not None and limit < len(rekordy):
        # losowanie z CAŁOŚCI, nie permutacja początku pliku
        rng = random.Random(ziarno)
        rekordy = rng.sample(rekordy, limit)

    return rekordy


def wczytaj_zrobione(sciezka: str) -> tuple[set[str], set[str]]:
    """
    Zwraca (zrobione, do_powtorzenia).

    Liczy się OSTATNI status danego ID w pliku — rekord, który raz padł,
    a przy powtórce się udał, jest zrobiony. Odwrotna kolejność (najpierw OK,
    potem ERROR) nie występuje, bo ID zrobione nie wracają do kolejki.
    """
    ostatni: dict[str, str] = {}
    if not Path(sciezka).exists():
        return set(), set()

    with open(sciezka, encoding="utf-8") as f:
        for linia in f:
            linia = linia.strip()
            if not linia:
                continue
            try:
                d = json.loads(linia)
            except json.JSONDecodeError:
                continue  # niepełna linia po twardym przerwaniu
            rid = d.get("id")
            if not rid:
                continue  # nagłówek
            ostatni[rid] = d.get("status", "OK")

    zrobione = {i for i, s in ostatni.items() if s != "ERROR"}
    bledne = {i for i, s in ostatni.items() if s == "ERROR"}
    return zrobione, bledne


class ZapisJSONL:
    """Zapis z natychmiastową trwałością — flush + fsync po każdym rekordzie."""

    def __init__(self, sciezka: str, naglowek: dict | None = None):
        Path(sciezka).parent.mkdir(parents=True, exist_ok=True)
        nowy = not Path(sciezka).exists() or Path(sciezka).stat().st_size == 0
        self._f = open(sciezka, "a", encoding="utf-8", newline="\n")
        if nowy and naglowek:
            self.zapisz(naglowek)

    def zapisz(self, rekord: dict) -> None:
        self._f.write(json.dumps(rekord, ensure_ascii=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def zamknij(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────── klient

_klient: OpenAI | None = None


def klient() -> OpenAI:
    global _klient
    if _klient is None:
        _klient = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio", timeout=300.0)
    return _klient


def czekaj_na_serwer() -> float:
    """
    Blokuje, dopóki LM Studio nie wróci. Bez limitu prób — lepiej stać
    godzinę niż przemielić resztę korpusu błędami.
    """
    t0 = time.time()
    log.error("LM Studio nie odpowiada — wstrzymuję przetwarzanie, czekam na powrót")
    ostatni_log = time.time()
    while True:
        time.sleep(ODSTEP_SONDY)
        try:
            klient().models.list()
            przerwa = time.time() - t0
            log.info("LM Studio wróciło po %.1f min — wznawiam", przerwa / 60)
            return przerwa
        except Exception:
            if time.time() - ostatni_log >= LOG_CZEKANIA_CO:
                log.warning("Nadal brak LM Studio (%.0f min)",
                            (time.time() - t0) / 60)
                ostatni_log = time.time()


def klasyfikuj(rekord: dict, szablon: str, model: str) -> dict:
    """Klasyfikuje jeden tekst. Zawsze zwraca rekord wynikowy."""
    znaki = len(rekord["tekst"])
    prompt = buduj_prompt(rekord["tekst"], szablon)
    baza = {
        "id": rekord["id"],
        "znaki": znaki,
        "obciety": znaki > LIMIT_ZNAKOW,
        "ts": teraz(),
    }
    ostatni_blad = None
    czas_przestoju = 0.0

    for proba in range(MAX_PROB):
        t0 = time.time()
        try:
            odp = klient().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURA,
                seed=SEED,
                max_tokens=MAX_TOKENOW,
            )
            wybor = odp.choices[0]
            surowa = (wybor.message.content or "").strip()
            kategorie, status, flagi = parsuj_odpowiedz(surowa)
            if wybor.finish_reason == "length":
                flagi.append("odpowiedz_ucieta")
            u = odp.usage

            wynik = {
                **baza,
                "status": status,
                "kategorie": kategorie,
                "finish_reason": wybor.finish_reason,
                "input_tokens": u.prompt_tokens if u else 0,
                "output_tokens": u.completion_tokens if u else 0,
                "latencja_s": round(time.time() - t0, 3),
                "proba": proba + 1,
                "flagi": flagi,
            }
            if status != "OK":
                wynik["raw"] = surowa[:2000]
            if czas_przestoju:
                wynik["przestoj_s"] = round(czas_przestoju, 1)
            return wynik

        except (APIConnectionError, APITimeoutError) as exc:
            # Serwer padł: czekamy na powrót i próbujemy ten sam tekst jeszcze raz.
            ostatni_blad = f"{type(exc).__name__}: {exc}"
            czas_przestoju += czekaj_na_serwer()

        except Exception as exc:
            ostatni_blad = f"{type(exc).__name__}: {exc}"
            if proba < MAX_PROB - 1:
                time.sleep((2 ** proba) + random.random())

    return {
        **baza,
        "status": "ERROR",
        "kategorie": [],
        "raw": None,
        "blad": ostatni_blad,
        "proba": MAX_PROB,
        "flagi": [],
        "latencja_s": 0.0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


# ──────────────────────────────────────────────────────────────── agregacja

class Agregator:
    """Liczniki batcha i przebiegu. Wszystko, co idzie do MLflow i progress.json."""

    def __init__(self, do_zrobienia: int, zrobione_wczesniej: int):
        self.do_zrobienia = do_zrobienia
        self.zrobione_wczesniej = zrobione_wczesniej
        self.t_start = time.time()
        self.n = 0
        self.ok = self.parse_fail = self.error = 0
        self.obciete = 0
        self.utracone_znaki = 0
        self.przestoj_s = 0.0
        self.kategorie = Counter()
        self.flagi = Counter()
        self.kubelki = Counter()
        self.kubelki_inne = Counter()
        self.etykiety = 0
        self._reset_batch()

    def _reset_batch(self) -> None:
        self.b_n = 0
        self.b_ok = self.b_parse_fail = self.b_error = 0
        self.b_latencje: list[float] = []
        self.b_tok_in: list[int] = []
        self.b_tok_out: list[int] = []
        self.b_etykiety = 0
        self.b_kategorie = Counter()
        self.b_t0 = time.time()

    def dodaj(self, w: dict) -> None:
        self.n += 1
        self.b_n += 1
        status = w["status"]

        if status == "OK":
            self.ok += 1
            self.b_ok += 1
        elif status == "PARSE_FAIL":
            self.parse_fail += 1
            self.b_parse_fail += 1
        else:
            self.error += 1
            self.b_error += 1

        if w.get("obciety"):
            self.obciete += 1
            self.utracone_znaki += w["znaki"] - LIMIT_ZNAKOW
        self.przestoj_s += w.get("przestoj_s", 0.0)

        for f in w.get("flagi", []):
            self.flagi[f] += 1

        if status != "ERROR":
            self.b_latencje.append(w["latencja_s"])
            self.b_tok_in.append(w["input_tokens"])
            self.b_tok_out.append(w["output_tokens"])

        kat = w.get("kategorie") or []
        self.kategorie.update(kat)
        self.b_kategorie.update(kat)
        self.etykiety += len(kat)
        self.b_etykiety += len(kat)

        k = kubelek(w["znaki"])
        self.kubelki[k] += 1
        if kat == ["INNE"]:
            self.kubelki_inne[k] += 1

    # ------------------------------------------------------------- raportowanie

    @property
    def tempo_srednie(self) -> float:
        elapsed = time.time() - self.t_start - self.przestoj_s
        return self.n / elapsed if elapsed > 0 else 0.0

    @property
    def eta_h(self) -> float:
        t = self.tempo_srednie
        return (self.do_zrobienia - self.n) / t / 3600 if t > 0 else 0.0

    def metryki(self) -> dict:
        czas_b = max(time.time() - self.b_t0, 1e-9)
        m = {
            "przetworzone": self.n,
            "tempo_batch": self.b_n / czas_b,
            "tempo_srednie": self.tempo_srednie,
            "eta_h": self.eta_h,
            "odsetek_ok": self.ok / self.n if self.n else 0.0,
            "odsetek_parse_fail": self.parse_fail / self.n if self.n else 0.0,
            "odsetek_error": self.error / self.n if self.n else 0.0,
            "parse_fail_batch": self.b_parse_fail,
            "error_batch": self.b_error,
            "sr_etykiet": self.b_etykiety / self.b_n if self.b_n else 0.0,
            "obciete_skum": self.obciete,
            "przestoj_min": self.przestoj_s / 60,
        }
        if self.b_latencje:
            m["latencja_mediana"] = statistics.median(self.b_latencje)
            m["latencja_srednia"] = statistics.fmean(self.b_latencje)
            m["tok_in_sr"] = statistics.fmean(self.b_tok_in)
            m["tok_out_sr"] = statistics.fmean(self.b_tok_out)
        for kat in KATEGORIE:
            udzial = self.b_kategorie.get(kat, 0) / self.b_n if self.b_n else 0.0
            m[f"kat/{bez_ogonkow(kat)}"] = udzial
        for lo, hi in KUBELKI:
            k = f"{lo}_{hi}" if hi < 10 ** 9 else f"{lo}_plus"
            n_k = self.kubelki.get(k, 0)
            m[f"stratum/{k}/n"] = n_k
            m[f"stratum/{k}/inne"] = (self.kubelki_inne.get(k, 0) / n_k
                                      if n_k else 0.0)
        return m

    def postep(self, meta: dict) -> dict:
        return {
            "znacznik_czasu": teraz(),
            "meta": meta,
            "postep": {
                "zrobione_wczesniej": self.zrobione_wczesniej,
                "przetworzone_w_tym_uruchomieniu": self.n,
                "do_zrobienia": self.do_zrobienia,
                "odsetek": round(self.n / self.do_zrobienia * 100, 2)
                if self.do_zrobienia else 100.0,
            },
            "statusy": {"OK": self.ok, "PARSE_FAIL": self.parse_fail,
                        "ERROR": self.error},
            "tempo": {
                "tekst_na_s": round(self.tempo_srednie, 3),
                "eta_h": round(self.eta_h, 2),
                "czas_pracy_h": round((time.time() - self.t_start) / 3600, 2),
                "przestoj_min": round(self.przestoj_s / 60, 1),
            },
            "obciecia": {"liczba": self.obciete,
                         "utracone_znaki": self.utracone_znaki},
            "flagi": dict(self.flagi),
            "rozklad_kategorii": dict(self.kategorie.most_common()),
            "stratum_dlugosci": {
                k: {"n": v, "tylko_inne": self.kubelki_inne.get(k, 0)}
                for k, v in sorted(self.kubelki.items())
            },
        }


# ───────────────────────────────────────────────────────────────── przebieg

def obsluz_przerwanie(signum, ramka):  # noqa: ARG001
    global PRZERWANIE
    if PRZERWANIE:
        log.warning("Drugie przerwanie — wychodzę natychmiast")
        sys.exit(130)
    PRZERWANIE = True
    log.warning("Przerwanie: kończę bieżący tekst i zamykam pliki")


def przetworz(teksty: list[dict], szablon: str, meta: dict,
              drive: DriveSync, mlf: MLflowSink) -> Agregator:
    zapis = ZapisJSONL(PLIK_WYNIKOW, naglowek={"_naglowek": meta})
    zapis_bledow = ZapisJSONL(PLIK_BLEDOW)
    agg = Agregator(len(teksty), meta.get("zrobione_wczesniej", 0))
    numer_batcha = 0

    try:
        for rekord in teksty:
            if PRZERWANIE:
                break

            wynik = klasyfikuj(rekord, szablon, meta["model"])
            zapis.zapisz(wynik)
            if wynik["status"] == "ERROR":
                zapis_bledow.zapisz({"id": wynik["id"], "ts": wynik["ts"],
                                     "blad": wynik.get("blad")})
            agg.dodaj(wynik)

            if agg.n % BATCH_RAPORT == 0:
                numer_batcha += 1
                raportuj(agg, numer_batcha, meta, drive, mlf)
                agg._reset_batch()

            if agg.n % BATCH_BACKUP == 0:
                drive.zlec("wyniki.jsonl.gz", PLIK_WYNIKOW, spakuj=True)
                log.info("Zlecono kopię pełnych wyników na Dysk")
    finally:
        numer_batcha += 1
        raportuj(agg, numer_batcha, meta, drive, mlf)
        drive.zlec("wyniki.jsonl.gz", PLIK_WYNIKOW, spakuj=True)
        zapis.zamknij()
        zapis_bledow.zamknij()

    return agg


def raportuj(agg: Agregator, numer: int, meta: dict,
             drive: DriveSync, mlf: MLflowSink) -> None:
    if agg.n == 0:
        return

    mlf.loguj(agg.metryki(), krok=numer)

    postep = agg.postep(meta)
    try:
        Path(PLIK_POSTEPU).parent.mkdir(parents=True, exist_ok=True)
        tmp = PLIK_POSTEPU + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(postep, f, ensure_ascii=False, indent=2)
        os.replace(tmp, PLIK_POSTEPU)
        drive.zlec("progress.json", PLIK_POSTEPU)
    except Exception as exc:
        log.warning("Zapis progress.json nieudany: %s", exc)

    log.info(
        "%7d/%d | %.2f tekst/s | OK %d  PARSE_FAIL %d  ERROR %d | pozostało ~%.1f h",
        agg.n, agg.do_zrobienia, agg.tempo_srednie,
        agg.ok, agg.parse_fail, agg.error, agg.eta_h,
    )


# ──────────────────────────────────────────────────────────────── benchmark

def benchmark(teksty: list[dict], szablon: str, model: str,
              warianty: list[int]) -> None:
    """Mierzy przepustowość dla różnych liczb wątków. Nie zapisuje wyników."""
    from concurrent.futures import ThreadPoolExecutor

    log.info("Rozgrzewka (%d tekstów)…", min(10, len(teksty)))
    for r in teksty[:10]:
        klasyfikuj(r, szablon, model)

    print(f"\n  {'wątki':>6}  {'tekst/s':>9}  {'czas [s]':>9}  "
          f"{'wykorzyst.':>11}  {'PARSE_FAIL':>11}  {'ERROR':>6}")
    print("  " + "-" * 62)

    for w in warianty:
        t0 = time.time()
        if w == 1:
            wyniki = [klasyfikuj(r, szablon, model) for r in teksty]
        else:
            with ThreadPoolExecutor(max_workers=w) as pool:
                wyniki = list(pool.map(
                    lambda r: klasyfikuj(r, szablon, model), teksty))
        czas = time.time() - t0
        suma_lat = sum(x["latencja_s"] for x in wyniki)
        pf = sum(1 for x in wyniki if x["status"] == "PARSE_FAIL")
        er = sum(1 for x in wyniki if x["status"] == "ERROR")
        print(f"  {w:>6}  {len(teksty)/czas:>9.3f}  {czas:>9.1f}  "
              f"{suma_lat/(czas*w):>11.3f}  {pf:>11}  {er:>6}")

    print("\n  Uwaga: przy >1 wątku wyniki bywają inne niż sekwencyjne "
          "(zmienny skład batcha).\n")


# ────────────────────────────────────────────────────────────────── eksport

def eksportuj(plik_wej: str, plik_wyj: str) -> None:
    rekordy: list[tuple[str, str]] = []
    pominiete = 0
    with open(plik_wej, encoding="utf-8") as f:
        for linia in f:
            linia = linia.strip()
            if not linia:
                continue
            try:
                d = json.loads(linia)
            except json.JSONDecodeError:
                continue
            if not d.get("id") or d.get("status") != "OK":
                pominiete += 1
                continue
            rekordy.append((d["id"], ",".join(d["kategorie"])))

    unikalne = dict(rekordy)
    Path(plik_wyj).parent.mkdir(parents=True, exist_ok=True)
    with open(plik_wyj, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        for rid in sorted(unikalne):
            w.writerow([rid, unikalne[rid]])

    print(f"  Wyeksportowano {len(unikalne)} rekordów -> {plik_wyj}")
    if pominiete:
        print(f"  Pominięto {pominiete} rekordów bez statusu OK "
              f"(nagłówek, PARSE_FAIL, ERROR).")
    if len(rekordy) != len(unikalne):
        print(f"  [UWAGA] Usunięto {len(rekordy) - len(unikalne)} duplikatów.")


# ───────────────────────────────────────────────────────────────────── main

def main() -> None:
    p = argparse.ArgumentParser(description="Anotacja korpusu — przebieg produkcyjny")
    p.add_argument("--teksty", default=PLIK_TEKSTOW)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--metoda", default=METODA)
    p.add_argument("--limit", type=int, default=None,
                   help="Losowa próbka z całości korpusu")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--eksport", action="store_true")
    p.add_argument("--test-drive", action="store_true",
                   help="Sprawdź połączenie z Dyskiem i zakończ")
    args = p.parse_args()

    ustaw_logowanie(PLIK_LOGU)
    signal.signal(signal.SIGINT, obsluz_przerwanie)

    if args.test_drive:
        print("\n  TEST POŁĄCZENIA Z DYSKIEM GOOGLE")
        print("  " + "-" * 60)
        d = DriveSync(GDRIVE_SA_JSON, GDRIVE_FOLDER_ID)
        ok = d.sprawdz(str(Path(PLIK_POSTEPU).parent))
        d.zamknij()
        print("  " + "-" * 60)
        print("  Gotowe do przebiegu.\n" if ok else
              "  Napraw powyższe przed uruchomieniem przebiegu.\n")
        return

    if args.eksport:
        eksportuj(PLIK_WYNIKOW, PLIK_EKSPORTU)
        return

    szablon = wczytaj_szablon(args.metoda)
    suma = suma_kontrolna_promptu(szablon)
    if OCZEKIWANA_SUMA and suma != OCZEKIWANA_SUMA:
        raise SystemExit(
            f"BŁĄD: suma kontrolna promptu {suma} != oczekiwana {OCZEKIWANA_SUMA}. "
            f"Prompt lub definicje kategorii rozjechały się z ewaluacją."
        )

    wszystkie = wczytaj_teksty(args.teksty, args.limit, ziarno=SEED)

    if args.benchmark:
        benchmark(wszystkie, szablon, args.model, [1, 2, 4, 8])
        return

    zrobione, do_powtorzenia = wczytaj_zrobione(PLIK_WYNIKOW)
    do_zrobienia = [r for r in wszystkie if r["id"] not in zrobione]

    meta = {
        "run": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "model": args.model,
        "metoda": args.metoda,
        "suma_kontrolna": suma,
        "seed": SEED,
        "temperatura": TEMPERATURA,
        "limit_znakow": LIMIT_ZNAKOW,
        "max_tokenow": MAX_TOKENOW,
        "lm_studio_url": LM_STUDIO_URL,
        "kategorii": len(KATEGORIE),
        "wersja_programu": "1.0",
        "zrobione_wczesniej": len(zrobione),
    }

    log.info("=" * 68)
    for k, v in meta.items():
        log.info("  %-18s: %s", k, v)
    log.info("  %-18s: %d", "tekstów w pliku", len(wszystkie))
    log.info("  %-18s: %d", "już przetworzonych", len(zrobione))
    log.info("  %-18s: %d", "do powtórzenia (ERROR)", len(do_powtorzenia))
    log.info("  %-18s: %d", "do przetworzenia", len(do_zrobienia))
    log.info("=" * 68)

    if not do_zrobienia:
        log.info("Nic do zrobienia.")
        return

    drive = DriveSync(GDRIVE_SA_JSON, GDRIVE_FOLDER_ID)
    log.info("Dysk Google: %s", drive.status)
    mlf = MLflowSink(MLFLOW_URI, MLFLOW_EKSPERYMENT, meta["run"], meta)

    agg = przetworz(do_zrobienia, szablon, meta, drive, mlf)

    log.info("-" * 68)
    log.info("  Przetworzone     : %d", agg.n)
    log.info("  OK / PARSE_FAIL / ERROR : %d / %d / %d",
             agg.ok, agg.parse_fail, agg.error)
    log.info("  Czas             : %.2f h", (time.time() - agg.t_start) / 3600)
    log.info("  Przestój serwera : %.1f min", agg.przestoj_s / 60)
    log.info("  Tempo            : %.3f tekst/s", agg.tempo_srednie)
    log.info("  Obcięte          : %d (%d znaków)", agg.obciete,
             agg.utracone_znaki)
    if agg.flagi:
        log.info("  Flagi            : %s", dict(agg.flagi))
    log.info("=" * 68)

    mlf.tag("przerwane", str(PRZERWANIE))
    mlf.artefakt(PLIK_POSTEPU)
    mlf.zakoncz("FINISHED" if not PRZERWANIE else "KILLED")
    drive.zamknij()

    if agg.error:
        log.info("Rekordy ERROR wrócą do kolejki przy kolejnym uruchomieniu.")


if __name__ == "__main__":
    main()
