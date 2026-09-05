"""
Anotacja korpusu — przebieg produkcyjny.

Różnice względem ewaluatora:
  - brak MLflow (brak złotego standardu, brak metryk do policzenia)
  - przetwarzanie współbieżne (ThreadPoolExecutor)
  - zapis przyrostowy do JSONL + wznawianie po przerwaniu
  - retry z wykładniczym opóźnieniem
  - tryb benchmark do doboru liczby wątków

UWAGA: przetwarzanie współbieżne zwiększa niedeterminizm wyników
(zmienny skład batcha => zmienna kolejność redukcji zmiennoprzecinkowej).
Do pomiarów powtarzalności używaj trybu sekwencyjnego (--workers 1).

Użycie:
    # 1. dobierz liczbę wątków
    python anotacja_korpusu.py --benchmark --limit 200

    # 2. właściwy przebieg
    python anotacja_korpusu.py --workers 4

    # 3. po przerwaniu — to samo polecenie, wznowi od miejsca zatrzymania
    python anotacja_korpusu.py --workers 4

    # 4. eksport do formatu id<TAB>KAT1,KAT2
    python anotacja_korpusu.py --eksport
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME: str = "qwen3.5-9b@q6_k"
METODA_PROMPTU: str = "one_shot"
LIMIT_ZNAKOW_TEKSTU: int = 6000
MAX_TOKENS_ODPOWIEDZI: int = 500
TEMPERATURA: float = 0.0

PLIK_TEKSTOW: str = "data/teksty.txt"
PLIK_WYJSCIOWY: str = "results/anotacja.jsonl"
PLIK_BLEDOW: str = "results/anotacja_bledy.jsonl"
PLIK_EKSPORTU: str = "results/anotacja.txt"

MAX_PROB: int = 4          # liczba prób na tekst przy błędzie
RAPORT_CO: int = 200       # co ile tekstów raportować postęp

# ─────────────────────────────────────────────────────────────────────────────
# DEFINICJE KATEGORII
# ─────────────────────────────────────────────────────────────────────────────

KATEGORIE_DEF: dict[str, str] = {
    "GOSPODARKA": (
        "budżet, podatki, inflacja, rynek pracy, sektory gospodarki. Wszystko związane z robieniem i wydawaniem piniądza, biznesem, finansami, inwestycjami, przedsiębiorczością."
    ),
    "POLITYKA_SPOŁECZNA": (
        "emerytury, ZUS, świadczenia, rodzina, pomoc społeczna, prawa społeczne obywateli, niepełnosprawności, ubóstwo, bezdomność."
    ),
    "BEZPIECZEŃSTWO": (
        "wojsko, policja, granice, cyberbezpieczeństwo, służby mundurowe, działania na obronność państwa, NATO, wydatki militarne."
    ),
    "ŚWIATOPOGLĄD": (
        # "religia, prawa kobiet, LGBT+, silne prywatne poglądy polityków, kwestie światopoglądowe, wartości, etyka, związki partnerskie."
        "religia, prawa reprodukcyjne, LGBT+, związki partnerskie, etyka, wartości — tylko gdy stanowią główny przedmiot sporu. Sama emocjonalność wypowiedzi lub odwołanie do wartości w argumentacji to za mało."
    ),
    "PRAWORZĄDNOŚĆ": (
        "sądy, konstytucja, prawo, regulacje, ustawy, projekty ustaw"
    ),
    "EDUKACJA_NAUKA": (
        "szkoły, uczelnie, nauka, system oświaty, program nauczania, nauczyciele, badania naukowe, stypendia."
    ),
    "ŚRODOWISKO": (
        "klimat, energia, odpady, przyroda."
    ),
    "SPRAWY_ZAGRANICZNE": (
        "dyplomacja, UE, NATO, relacje międzynarodowe. Nie dotyczy, gdy wspomniany kraj jest tylko przykładem w innym kontekście (np. 'jak w Niemczech...')."
    ),
    "ROLNICTWO": (
        "rolnicy, hodowla, gleba, pestycydy, łowiectwo, rybołóstwo, dopłaty dla rolników, skupy, agrobiznes."
    ),
    "POLITYKA_LOKALNA": (
        "samorząd, regiony, inwestycje lokalne."
    ),
    "ADMINISTRACJA": (
        "działanie organów władzy wykonawczej: ministerstwa, urzędy, agencje, rząd i resorty, procedury publiczne, systemy obsługi obywateli, interpelacje i odpowiedzi na nie, nadzór i kontrola. Przypisuj również jako etykietę dodatkową, ilekroć wypowiedź dotyczy tego, jak państwo coś realizuje lub wdraża — obok kategorii tematycznej. Nie dotyczy prowadzenia obrad Sejmu (→ INNE)."
        # "działanie urzędów, ministerstw, procedury publiczne, pisma, zawiadomienia. Nie dotyczy wypowiedzi organizacyjne parlamentu w trakcie trwania obrad."
    ),
    "INFRASTRUKTURA": (
        "drogi, kolej, transport publiczny, budynki publiczne (np. szpitale jako obiekty budowlane), inwestycje infrastrukturalne, poruszanie się po drogach"
    ),
    "HISTORIA_PAMIĘĆ_NARODOWA": (
        # "rocznice, IPN, pamięć historyczna, II Wojna Światowa, komunizm, bohaterowie narodowi, muzea, pomniki."
        "rocznice, IPN, pamięć historyczna, II Wojna Światowa, komunizm, bohaterowie narodowi, muzea, pomniki.tylko gdy pamięć historyczna jest przedmiotem wypowiedzi. Odwołania retoryczne, porównania do przeszłości i przywoływanie postaci historycznych w argumentacji nie kwalifikują."
    ),
    "ZDROWIE": (
        "leczenie, NFZ, lekarze, epidemie."
    ),
    "INNE": (
        # "Wypowiedzi organizacyjne parlamentu w trakcie trwania obrad. Kategoria, która nie pasuje do żadnej z powyższych lub jest zbyt ogólna/niejednoznaczna. Używaj tylko wtedy, gdy nie da się przypisać żadnej konkretnej innej kategorii."
        "Jeśli wypowiedź dotyczy prowadzenia obrad (otwarcie/zamknięcie posiedzenia, zapowiedź głosowania, udzielanie głosu, wnioski formalne, komunikaty marszałka), przypisz wyłącznie INNE — nawet jeśli pada w niej tytuł ustawy, nazwa ministerstwa czy temat merytoryczny. Sam tytuł omawianego punktu nie czyni wypowiedzi merytoryczną."
    ),
}

KATEGORIE: list[str] = list(KATEGORIE_DEF.keys())



# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────

def wczytaj_szablon(metoda: str, katalog: str = "methods") -> str:
    sciezka = Path(katalog) / f"{metoda}.txt"
    if not sciezka.exists():
        raise FileNotFoundError(f"Brak pliku metody: {sciezka}")
    return sciezka.read_text(encoding="utf-8")


def buduj_prompt(tekst: str, szablon: str) -> str:
    definicje = "\n".join(f"- {k}: {v}" for k, v in KATEGORIE_DEF.items())
    return (
        szablon
        .replace("{tekst}", tekst[:LIMIT_ZNAKOW_TEKSTU])
        .replace("{definicje}", definicje)
        .replace("{kategorie}", ", ".join(KATEGORIE))
    )


def suma_kontrolna_promptu(szablon: str) -> str:
    """
    Skrót szablonu + definicji kategorii. Zapisywany w nagłówku wyniku,
    żeby dało się później jednoznacznie ustalić, na jakiej wersji
    kategorii powstała anotacja.
    """
    import hashlib
    tresc = szablon + json.dumps(KATEGORIE_DEF, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(tresc.encode("utf-8")).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# PARSOWANIE (identyczne z ewaluatorem — nie zmieniaj bez powtórzenia ewaluacji)
# ─────────────────────────────────────────────────────────────────────────────

def parsuj_odpowiedz(raw: str) -> list[str]:
    tekst = raw.strip()

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
    return wynik or ["INNE"]


# ─────────────────────────────────────────────────────────────────────────────
# WEJŚCIE / WYJŚCIE
# ─────────────────────────────────────────────────────────────────────────────

def wczytaj_teksty(sciezka: str, limit: int | None = None) -> list[dict]:
    rekordy = []
    with open(sciezka, encoding="utf-8") as f:
        for i, linia in enumerate(f):
            if limit is not None and i >= limit:
                break
            linia = linia.rstrip("\n")
            if not linia.strip():
                continue
            czesci = linia.split("\t", 1)
            if len(czesci) < 2:
                continue
            rekordy.append({"id": czesci[0].strip(), "tekst": czesci[1].strip()})
    return rekordy


def wczytaj_zrobione(sciezka: str) -> set[str]:
    """Zbiór id już zapisanych — podstawa wznawiania."""
    if not Path(sciezka).exists():
        return set()
    zrobione = set()
    with open(sciezka, encoding="utf-8") as f:
        for linia in f:
            linia = linia.strip()
            if not linia:
                continue
            try:
                zrobione.add(json.loads(linia)["id"])
            except (json.JSONDecodeError, KeyError):
                continue  # niepełna linia po twardym przerwaniu — pomiń
    return zrobione


class ZapisPrzyrostowy:
    """Bezpieczny wątkowo zapis JSONL z natychmiastowym flush."""

    def __init__(self, sciezka: str):
        Path(sciezka).parent.mkdir(parents=True, exist_ok=True)
        self._f = open(sciezka, "a", encoding="utf-8")
        self._lock = threading.Lock()

    def zapisz(self, rekord: dict) -> None:
        with self._lock:
            self._f.write(json.dumps(rekord, ensure_ascii=False) + "\n")
            self._f.flush()
            os.fsync(self._f.fileno())

    def zamknij(self) -> None:
        self._f.close()


# ─────────────────────────────────────────────────────────────────────────────
# KLIENT
# ─────────────────────────────────────────────────────────────────────────────

_thread_local = threading.local()


def klient() -> OpenAI:
    """Klient per wątek — unika współdzielenia stanu połączenia."""
    if not hasattr(_thread_local, "client"):
        base_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
        _thread_local.client = OpenAI(base_url=base_url, api_key="lm-studio")
    return _thread_local.client


def klasyfikuj(rekord: dict, szablon: str, model: str) -> dict:
    """Klasyfikuje jeden tekst. Zwraca rekord wynikowy lub rekord błędu."""
    prompt = buduj_prompt(rekord["tekst"], szablon)
    ostatni_blad = None

    for proba in range(MAX_PROB):
        t0 = time.time()
        try:
            odp = klient().chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURA,
                max_tokens=MAX_TOKENS_ODPOWIEDZI,
            )
            surowa = odp.choices[0].message.content.strip()
            u = odp.usage
            return {
                "id": rekord["id"],
                "kategorie": parsuj_odpowiedz(surowa),
                "raw": surowa,
                "latencja_s": round(time.time() - t0, 3),
                "input_tokens": u.prompt_tokens if u else 0,
                "output_tokens": u.completion_tokens if u else 0,
                "proba": proba + 1,
            }
        except Exception as exc:
            ostatni_blad = str(exc)
            if proba < MAX_PROB - 1:
                # wykładnicze opóźnienie z jitterem
                time.sleep((2 ** proba) + random.random())

    return {"id": rekord["id"], "blad": ostatni_blad}


# ─────────────────────────────────────────────────────────────────────────────
# PRZEBIEG
# ─────────────────────────────────────────────────────────────────────────────

def przetworz(
    teksty: list[dict],
    szablon: str,
    model: str,
    workers: int,
    plik_wyj: str,
    plik_bled: str,
    cicho: bool = False,
) -> dict:
    """Przetwarza listę tekstów. Zwraca statystyki przebiegu."""
    zapis = ZapisPrzyrostowy(plik_wyj)
    zapis_bledow = ZapisPrzyrostowy(plik_bled)

    n = len(teksty)
    ok = err = 0
    suma_latencji = 0.0
    suma_tok_in = 0
    t_start = time.time()

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(klasyfikuj, r, szablon, model): r["id"]
                for r in teksty
            }
            for i, fut in enumerate(as_completed(futures), 1):
                wynik = fut.result()

                if "blad" in wynik:
                    err += 1
                    zapis_bledow.zapisz(wynik)
                else:
                    ok += 1
                    suma_latencji += wynik["latencja_s"]
                    suma_tok_in += wynik["input_tokens"]
                    zapis.zapisz({
                        "id": wynik["id"],
                        "kategorie": wynik["kategorie"],
                        "input_tokens": wynik["input_tokens"],
                        "output_tokens": wynik["output_tokens"],
                        "latencja_s": wynik["latencja_s"],
                    })

                if not cicho and i % RAPORT_CO == 0:
                    elapsed = time.time() - t_start
                    tempo = i / elapsed
                    pozostalo = (n - i) / tempo if tempo > 0 else 0
                    print(
                        f"  {i:>7}/{n}  |  {tempo:5.2f} tekst/s  |  "
                        f"blędy: {err}  |  pozostało ~{pozostalo/3600:.1f} h"
                    )
    finally:
        zapis.zamknij()
        zapis_bledow.zamknij()

    czas = time.time() - t_start
    return {
        "n": n,
        "ok": ok,
        "err": err,
        "czas_s": round(czas, 1),
        "tekst_na_s": round(ok / czas, 3) if czas > 0 else 0,
        "suma_latencji_s": round(suma_latencji, 1),
        # Kluczowa diagnostyka: ile czasu zegarowego przypada na czas
        # spędzony w wywołaniach modelu. Przy przetwarzaniu sekwencyjnym
        # wartość < 1 oznacza narzut poza modelem.
        "wykorzystanie": round(suma_latencji / (czas * workers), 3) if czas > 0 else 0,
        "sr_tok_in": round(suma_tok_in / ok, 1) if ok else 0,
    }


def benchmark(teksty: list[dict], szablon: str, model: str, warianty: list[int]) -> None:
    """Mierzy przepustowość dla różnych liczb wątków."""
    print("\n  BENCHMARK ZRÓWNOLEGLENIA")
    print("  " + "-" * 68)
    print(f"  {'wątki':>6}  {'tekst/s':>9}  {'czas [s]':>9}  {'wykorzyst.':>11}  {'błędy':>6}")
    print("  " + "-" * 68)

    najlepszy = (0, 0.0)
    for w in warianty:
        tmp_ok = f"results/_bench_{w}.jsonl"
        tmp_err = f"results/_bench_{w}_err.jsonl"
        # for p in (tmp_ok, tmp_err):
        #     Path(p).unlink(missing_ok=True)

        st = przetworz(teksty, szablon, model, w, tmp_ok, tmp_err, cicho=True)
        print(
            f"  {w:>6}  {st['tekst_na_s']:>9.3f}  {st['czas_s']:>9.1f}  "
            f"{st['wykorzystanie']:>11.3f}  {st['err']:>6}"
        )
        if st["tekst_na_s"] > najlepszy[1]:
            najlepszy = (w, st["tekst_na_s"])

        # for p in (tmp_ok, tmp_err):
            # Path(p).unlink(missing_ok=True)

    w, tempo = najlepszy
    print("  " + "-" * 68)
    print(f"  Najlepszy wariant: {w} wątków, {tempo:.3f} tekst/s")
    print(f"  Szacowany czas dla 250 000 tekstów: {250_000/tempo/3600:.1f} h "
          f"({250_000/tempo/86400:.1f} dni)")
    print()
    print("  Kolumna 'wykorzyst.' = suma latencji / (czas zegarowy * wątki).")
    print("  Wartość bliska 1 przy 1 wątku => narzut poza modelem jest mały.")
    print("  Wartość wyraźnie < 1 przy 1 wątku => czas idzie poza model.")
    print("  Spadek wraz ze wzrostem wątków => GPU jest wysycone.")


def eksportuj(plik_wej: str, plik_wyj: str) -> None:
    """Konwertuje JSONL na format id<TAB>KAT1,KAT2 posortowany po id."""
    rekordy = []
    with open(plik_wej, encoding="utf-8") as f:
        for linia in f:
            linia = linia.strip()
            if not linia:
                continue
            try:
                d = json.loads(linia)
                rekordy.append((d["id"], ",".join(d["kategorie"])))
            except (json.JSONDecodeError, KeyError):
                continue

    # deduplikacja (na wypadek podwójnego przetworzenia przy wznowieniu)
    unikalne = dict(rekordy)
    Path(plik_wyj).parent.mkdir(parents=True, exist_ok=True)
    with open(plik_wyj, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for rid in sorted(unikalne):
            w.writerow([rid, unikalne[rid]])

    print(f"  Wyeksportowano {len(unikalne)} rekordów -> {plik_wyj}")
    if len(rekordy) != len(unikalne):
        print(f"  [UWAGA] Usunięto {len(rekordy) - len(unikalne)} duplikatów.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Anotacja korpusu — przebieg produkcyjny")
    p.add_argument("--teksty", default=PLIK_TEKSTOW)
    p.add_argument("--wyjscie", default=PLIK_WYJSCIOWY)
    p.add_argument("--bledy", default=PLIK_BLEDOW)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--metoda", default=METODA_PROMPTU)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--benchmark", action="store_true",
                   help="Zmierz przepustowość dla 1/2/4/8 wątków i zakończ")
    p.add_argument("--eksport", action="store_true",
                   help="Przekonwertuj JSONL na TSV i zakończ")
    args = p.parse_args()

    load_dotenv()

    if args.eksport:
        eksportuj(args.wyjscie, PLIK_EKSPORTU)
        return

    szablon = wczytaj_szablon(args.metoda)
    suma = suma_kontrolna_promptu(szablon)

    print("=" * 72)
    print("  ANOTACJA KORPUSU")
    print("=" * 72)
    print(f"  Model            : {args.model}")
    print(f"  Metoda           : {args.metoda}")
    print(f"  Suma kontrolna   : {suma}")
    print(f"  Kategorii        : {len(KATEGORIE)}")

    wszystkie = wczytaj_teksty(args.teksty, args.limit)
    if args.limit:
        random.seed(42)
        wszystkie = random.sample(wszystkie, min(args.limit, len(wszystkie)))
    print(f"  Tekstów w pliku  : {len(wszystkie)}")

    if args.benchmark:
        benchmark(wszystkie, szablon, args.model, [1, 2, 4, 8])
        return

    zrobione = wczytaj_zrobione(args.wyjscie)
    do_zrobienia = [r for r in wszystkie if r["id"] not in zrobione]
    print(f"  Już przetworzone : {len(zrobione)}")
    print(f"  Do przetworzenia : {len(do_zrobienia)}")
    print(f"  Wątki            : {args.workers}")
    print("-" * 72)

    if not do_zrobienia:
        print("  Nic do zrobienia.")
        return

    st = przetworz(
        do_zrobienia, szablon, args.model, args.workers,
        args.wyjscie, args.bledy,
    )

    print("-" * 72)
    print(f"  Sukces           : {st['ok']}")
    print(f"  Błędy            : {st['err']}")
    print(f"  Czas             : {st['czas_s']/3600:.2f} h")
    print(f"  Tempo            : {st['tekst_na_s']:.3f} tekst/s")
    print(f"  Suma latencji    : {st['suma_latencji_s']/3600:.2f} h")
    print(f"  Wykorzystanie    : {st['wykorzystanie']:.3f}")
    print(f"  Śr. tokenów wej. : {st['sr_tok_in']:.1f}")
    print("=" * 72)
    if st["err"]:
        print(f"  Błędy zapisane w {args.bledy} — uruchom ponownie, "
              f"żeby przetworzyć brakujące.")


if __name__ == "__main__":
    main()
