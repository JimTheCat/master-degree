"""
Ewaluator wypowiedzi sejmowych — klasyfikacja tematyczna z użyciem LLM.

WERSJA BEZ MLFLOW — identyczna logika i wyniki, bez raportowania do MLflow.
Zapis wynikow i tokenow do results/ dziala jak dotychczas.

Obsługiwane providery: OpenAI | Anthropic | Gemini | LM Studio (lokalny)
Śledzenie:            brak (wersja bez MLflow)
Koszty:               liczenie tokenów i szacowanie kosztu USD per tekst i sumarycznie
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTY
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import csv
import datetime
import io
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import uuid

from dotenv import load_dotenv


# Sledzenie LangSmith usuniete calkowicie. Sama zmienna srodowiskowa nie
# wystarczala — biblioteka honoruje kilka nazw naraz, a load_dotenv() moglo
# dociagnac wlasna wartosc z .env. Pusty dekorator wyklucza jakikolwiek ruch
# sieciowy niezaleznie od konfiguracji.
def traceable(*args, **kwargs):
    """Zaslepka po @traceable z LangSmith — zwraca funkcje bez zmian."""
    if args and callable(args[0]):
        return args[0]

    def dekorator(fn):
        return fn

    return dekorator


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#   SEKCJA KONFIGURACJI — ZMIEŃ PRZED URUCHOMIENIEM
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# Liczba pierwszych wierszy do wysłania do LLM.
# Zacznij od 10 żeby sprawdzić działanie, potem zwiększaj.
LIMIT_WIERSZY: int = 500          # <- ZMIEŃ TU  (10 | 100 | 500)

# Temperatura modelu — 0.0 daje deterministyczne, powtarzalne wyniki.
# Zalecana wartość dla zadań klasyfikacyjnych.
TEMPERATURA: float = 0.0         # <- ZMIEŃ TU  (0.0 zalecane dla klasyfikacji)

# Aktywny provider LLM.
WYBRANY_PROVIDER: str = "lm_studio" # <- ZMIEŃ TU  (openai | anthropic | gemini | lm_studio)

LIMIT_ZNAKÓW_TEKSTU: int = 6000  # <- dostosuj do najsłabszego modelu w eksperymencie

# ──────────────────────────────────────────────────────────────────────────────
#   WYBÓR METODY PROMPTOWANIA
#   Wskaż nazwę pliku z katalogu methods/ bez rozszerzenia .txt
#   Dostępne pliki: zero_shot | one_shot | few_shot | many_shot
# ──────────────────────────────────────────────────────────────────────────        ────

METODA_PROMPTU: str = ("one_shot")  # <- ZMIEŃ TU  (zero_shot | one_shot | few_shot)

# ──────────────────────────────────      ───────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA MODELI
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIG: dict[str, dict] = {
    # GPT-4o — najlepsza obsługa języka polskiego w ofercie OpenAI
    "openai": {
        "model_name": "gpt-4o",
    },
    # Claude Sonnet — dobry balans jakości i kosztów, świetny polski
    "anthropic": {
        "model_name": "claude-sonnet-4-5",
    },
    # Gemini 2.5 Flash — szybki i ekonomiczny
    "gemini": {
        "model_name": "gemini-3.5-flash",
    },
    # LM Studio — lokalny serwer z OpenAI-kompatybilnym API.
    # Nadpisywalne przez LM_STUDIO_MODEL i LM_STUDIO_URL w .env
    # lub przez --model w CLI.
    "lm_studio": {
        "model_name": "google/gemma-4-26b-a4b",  # <- ZMIEŃ na nazwę z LM Studio -> Local Server
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CENNIK MODELI (USD za 1 MILION tokenów)
# ─────────────────────────────────────────────────────────────────────────────
#
# Ceny standardowe (bez batch API, bez prompt cachingu), stan na czerwiec 2026.
# Źródła: oficjalne strony cenowe providerów. Warto okresowo weryfikować —
# ceny LLM-ów zmieniają się często, a literówka tu zafałszuje cały raport kosztów.
#
# LM Studio jest lokalny — koszt API wynosi $0 (jedyny "koszt" to prąd/sprzęt,
# czego ten skrypt nie liczy).
#
CENNIK_USD_PER_MILION: dict[str, dict[str, float]] = {
    # provider -> model_name -> {"input": ..., "output": ...}
    "openai": {
        "gpt-4o": {"input": 2.50, "output": 10.00},
    },
    "anthropic": {
        "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    },
    "gemini": {
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
        "gemini-3.5-flash": {"input": 1.50,  "output": 9.00},
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    },
    "lm_studio": {
        # Model lokalny — brak kosztu per token.
        "*": {"input": 0.0, "output": 0.0},
    },
}


def pobierz_cennik(provider: str, model_name: str) -> dict[str, float]:
    """
    Zwraca {"input": cena_per_mln, "output": cena_per_mln} dla danego providera/modelu.

    Jeśli model nie jest w cenniku (np. nadpisany przez --model), zwraca zera
    i wypisuje ostrzeżenie — koszt zostanie policzony jako 0.0 zamiast
    rzucić wyjątkiem, żeby nie przerywać ewaluacji.
    """
    cennik_providera = CENNIK_USD_PER_MILION.get(provider, {})

    if model_name in cennik_providera:
        return cennik_providera[model_name]

    # LM Studio — dowolna nazwa modelu lokalnego mapuje się na zerowy koszt
    if "*" in cennik_providera:
        return cennik_providera["*"]

    print(
        f"  [OSTRZEZENIE] Brak cennika dla '{provider}/{model_name}' — "
        f"koszt bedzie liczony jako $0.00. Dodaj wpis do CENNIK_USD_PER_MILION."
    )
    return {"input": 0.0, "output": 0.0}


def oblicz_koszt(tokeny: dict[str, int], cennik: dict[str, float]) -> float:
    """Oblicza koszt w USD na podstawie zużytych tokenów i cennika per milion."""
    koszt_input  = (tokeny.get("input_tokens", 0)  / 1_000_000) * cennik["input"]
    koszt_output = (tokeny.get("output_tokens", 0) / 1_000_000) * cennik["output"]
    return koszt_input + koszt_output


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
    ),
    "INFRASTRUKTURA": (
        "drogi, kolej, transport publiczny, budynki publiczne (np. szpitale jako obiekty budowlane), inwestycje infrastrukturalne, poruszanie się po drogach"
    ),
    "HISTORIA_PAMIĘĆ_NARODOWA": (
        "rocznice, IPN, pamięć historyczna, II Wojna Światowa, komunizm, bohaterowie narodowi, muzea, pomniki.tylko gdy pamięć historyczna jest przedmiotem wypowiedzi. Odwołania retoryczne, porównania do przeszłości i przywoływanie postaci historycznych w argumentacji nie kwalifikują."
    ),
    "ZDROWIE": (
        "leczenie, NFZ, lekarze, epidemie."
    ),
    "INNE": (
        "Jeśli wypowiedź dotyczy prowadzenia obrad (otwarcie/zamknięcie posiedzenia, zapowiedź głosowania, udzielanie głosu, wnioski formalne, komunikaty marszałka), przypisz wyłącznie INNE — nawet jeśli pada w niej tytuł ustawy, nazwa ministerstwa czy temat merytoryczny. Sam tytuł omawianego punktu nie czyni wypowiedzi merytoryczną."
    ),
}
#896 634
# KATEGORIE_DEF: dict[str, str] = {
#     "GOSPODARKA": (
#         "budżet, podatki, inflacja, rynek pracy, sektory gospodarki. "
#         "Wszystko związane z robieniem i wydawaniem piniądza, biznesem, "
#         "finansami, inwestycjami, przedsiębiorczością."
#     ),
#     "POLITYKA_SPOŁECZNA": (
#         "emerytury, ZUS, świadczenia, rodzina, pomoc społeczna, prawa "
#         "społeczne obywateli, niepełnosprawności, ubóstwo, bezdomność."
#     ),
#     "BEZPIECZEŃSTWO": (
#         "wojsko, policja, granice, cyberbezpieczeństwo, służby mundurowe, "
#         "działania na obronność państwa, NATO, wydatki militarne."
#     ),
#     "ŚWIATOPOGLĄD": (
#         "religia, prawa kobiet, LGBT+, silne prywatne poglądy polityków, "
#         "kwestie światopoglądowe, wartości, etyka, związki partnerskie."
#     ),
#     "PRAWORZĄDNOŚĆ": (
#         "sądy, konstytucja, prawo, regulacje, ustawy, projekty ustaw"
#     ),
#     "EDUKACJA_NAUKA": (
#         "szkoły, uczelnie, nauka, system oświaty, program nauczania, "
#         "nauczyciele, badania naukowe, stypendia."
#     ),
#     "ŚRODOWISKO": (
#         "klimat, energia, odpady, przyroda."
#     ),
#     "SPRAWY_ZAGRANICZNE": (
#         "dyplomacja, UE, NATO, relacje międzynarodowe. Nie dotyczy, gdy "
#         "wspomniany kraj jest tylko przykładem w innym kontekście "
#         "(np. 'jak w Niemczech...')."
#     ),
#     "ROLNICTWO": (
#         "rolnicy, hodowla, gleba, pestycydy, łowiectwo, rybołóstwo, "
#         "dopłaty dla rolników, skupy, agrobiznes."
#     ),
#     "POLITYKA_LOKALNA": (
#         "samorząd, regiony, inwestycje lokalne."
#     ),
#     "ADMINISTRACJA": (
#         "działanie urzędów, ministerstw, procedury publiczne, pisma, "
#         "zawiadomienia. Nie dotyczy wypowiedzi organizacyjne parlamentu "
#         "w trakcie trwania obrad."
#     ),
#     "INFRASTRUKTURA": (
#         "drogi, kolej, transport publiczny, budynki publiczne (np. szpitale "
#         "jako obiekty budowlane), inwestycje infrastrukturalne, poruszanie "
#         "się po drogach"
#     ),
#     "HISTORIA_PAMIĘĆ_NARODOWA": (
#         "rocznice, IPN, pamięć historyczna, II Wojna Światowa, komunizm, "
#         "bohaterowie narodowi, muzea, pomniki."
#     ),
#     "ZDROWIE": (
#         "leczenie, NFZ, lekarze, epidemie."
#     ),
#     "INNE": (
#         "Wypowiedzi organizacyjne parlamentu w trakcie trwania obrad. "
#         "Kategoria, która nie pasuje do żadnej z powyższych lub jest zbyt "
#         "ogólna/niejednoznaczna. Używaj tylko wtedy, gdy nie da się "
#         "przypisać żadnej konkretnej innej kategorii."
#     ),
# }

KATEGORIE: list[str] = list(KATEGORIE_DEF.keys())


# ─────────────────────────────────────────────────────────────────────────────
# ŁADOWANIE SZABLONU PROMPTU Z PLIKU
# ─────────────────────────────────────────────────────────────────────────────

def wczytaj_szablon_promptu(metoda: str, katalog: str = "../methods") -> str:
    """
    Wczytuje szablon promptu z pliku methods/<metoda>.txt.

    Plik może zawierać następujące znaczniki, które zostaną automatycznie
    zastąpione przez buduj_prompt():
      {tekst}      — treść klasyfikowanej wypowiedzi (max 1200 znaków)
      {definicje}  — blok definicji wszystkich kategorii
      {kategorie}  — lista nazw kategorii oddzielona przecinkami

    Raises:
        FileNotFoundError: gdy plik metody nie istnieje w katalogu methods/.
    """
    sciezka = Path(katalog) / f"{metoda}.txt"
    if not sciezka.exists():
        dostepne = (
            [p.stem for p in Path(katalog).glob("*.txt")]
            if Path(katalog).exists()
            else []
        )
        raise FileNotFoundError(
            f"Nie znaleziono pliku metody: '{sciezka}'\n"
            f"Dostępne metody w katalogu '{katalog}/': {dostepne or '(brak plików .txt)'}"
        )
    return sciezka.read_text(encoding="utf-8")


def buduj_prompt(tekst: str, szablon: str) -> str:
    """
    Buduje finalny prompt przez wstawienie treści wypowiedzi do szablonu.

    Zastępuje znaczniki {tekst}, {definicje}, {kategorie} odpowiednimi wartościami.
    """
    definicje = "\n".join(
        f"- {kat}: {opis}" for kat, opis in KATEGORIE_DEF.items()
    )
    return (
        szablon
        .replace("{tekst}", tekst[:LIMIT_ZNAKÓW_TEKSTU])
        .replace("{definicje}", definicje)
        .replace("{kategorie}", ", ".join(KATEGORIE))
    )


# ─────────────────────────────────────────────────────────────────────────────
# KLIENCI LLM
# ─────────────────────────────────────────────────────────────────────────────
#
# Każda funkcja wywolaj_* zwraca tuple (odpowiedz_tekstowa, tokeny), gdzie
# tokeny to dict {"input_tokens": int, "output_tokens": int, "total_tokens": int}.
# Dzięki temu zużycie tokenów jest dostępne od razu przy każdym wywołaniu,
# bez dodatkowego liczenia po fakcie (np. przez tiktoken).
#

# ─────────────────────────────────────────────────────────────────────────────
# KLIENCI HTTP — TWORZENI RAZ NA PRZEBIEG
# ─────────────────────────────────────────────────────────────────────────────
#
# Tworzenie klienta wewnatrz funkcji wywolujacej model oznacza nowy obiekt
# HTTP i nowe polaczenie TCP przy kazdym z 500 zapytan. Zamkniete polaczenia
# zostaja na Windowsie w stanie TIME_WAIT, przez co narzut rosnie w trakcie
# przebiegu. Klient trzymany w cache jest tworzony raz i utrzymuje pule
# polaczen przez caly czas dzialania programu.

_klienci: dict[str, object] = {}


def klient_lm_studio():
    if "lm_studio" not in _klienci:
        from openai import OpenAI
        base_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
        # LM Studio nie weryfikuje klucza — wymagana niepusta wartosc
        _klienci["lm_studio"] = OpenAI(base_url=base_url, api_key="lm-studio")
    return _klienci["lm_studio"]


def klient_openai():
    if "openai" not in _klienci:
        from openai import OpenAI
        _klienci["openai"] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _klienci["openai"]


def klient_anthropic():
    if "anthropic" not in _klienci:
        import anthropic
        _klienci["anthropic"] = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"])
    return _klienci["anthropic"]


@traceable(name="llm_openai", run_type="llm")
def wywolaj_openai(prompt: str, model_name: str, temperature: float) -> tuple[str, dict]:
    """Wywołuje OpenAI Chat API. Zwraca (odpowiedz, tokeny)."""
    response = klient_openai().chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=500,
    )
    tokeny = {
        "input_tokens":  response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens":  response.usage.total_tokens,
    }
    return response.choices[0].message.content.strip(), tokeny


@traceable(name="llm_anthropic", run_type="llm")
def wywolaj_anthropic(prompt: str, model_name: str, temperature: float) -> tuple[str, dict]:
    """Wywołuje Anthropic Messages API. Zwraca (odpowiedz, tokeny)."""
    message = klient_anthropic().messages.create(
        model=model_name,
        max_tokens=500,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    tokeny = {
        "input_tokens":  message.usage.input_tokens,
        "output_tokens": message.usage.output_tokens,
        "total_tokens":  message.usage.input_tokens + message.usage.output_tokens,
    }
    return message.content[0].text.strip(), tokeny


@traceable(name="llm_gemini", run_type="llm")
def wywolaj_gemini(prompt: str, model_name: str, temperature: float) -> tuple[str, dict]:
    """Wywołuje Google Gemini API (SDK google-genai). Zwraca (odpowiedz, tokeny)."""
    from google import genai
    from google.genai import types

    # Modele dostępne tylko przez v1beta (thinking/Pro modele)
    MODELE_BETA = {"gemini-2.5-pro"}
    api_version = "v1beta" if model_name in MODELE_BETA else "v1"

    client = genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"],
        http_options={"api_version": api_version},
    )
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=500
            # Zatrzymuje generowanie po nowej linii — zapobiega JSON-owi i komentarzom
            # stop_sequences=["\n"]#,
            # response_mime_type="text/plain",
        ),
    )

    meta = response.usage_metadata
    # candidates_token_count bywa None, gdy odpowiedź jest pusta/zablokowana
    # przez filtry bezpieczeństwa — wtedy traktujemy ją jako 0.
    tokeny = {
        "input_tokens":  meta.prompt_token_count or 0,
        "output_tokens": meta.candidates_token_count or 0,
        "total_tokens":  meta.total_token_count or 0,
    }
    return response.text.strip(), tokeny


@traceable(name="llm_lm_studio", run_type="llm")
def wywolaj_lm_studio(prompt: str, model_name: str, temperature: float) -> tuple[str, dict]:
    """
    Wywołuje lokalny model przez LM Studio (OpenAI-kompatybilne API).

    Konfiguracja w .env:
      LM_STUDIO_URL   — adres serwera (domyślnie: http://localhost:1234/v1)
      LM_STUDIO_MODEL — opcjonalne nadpisanie nazwy modelu bez edycji kodu

    Uwaga: nie każdy backend serwowany przez LM Studio zwraca pole `usage`
    (zależy od silnika inferencji). Jeśli go brak, zwracamy zera zamiast
    rzucać wyjątkiem.
    """
    model_final = os.environ.get("LM_STUDIO_MODEL") or model_name

    response = klient_lm_studio().chat.completions.create(
        model=model_final,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=500,
    )

    if response.usage is not None:
        tokeny = {
            "input_tokens":  response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens":  response.usage.total_tokens,
        }
    else:
        print("  [OSTRZEZENIE] LM Studio nie zwrocilo pola 'usage' — tokeny = 0.")
        tokeny = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    return response.choices[0].message.content.strip(), tokeny


# Rejestr providerów — dodaj tu nowy wpis, żeby rozszerzyć o kolejny model
PROVIDER_DISPATCH: dict[str, callable] = {
    "openai":    wywolaj_openai,
    "anthropic": wywolaj_anthropic,
    "gemini":    wywolaj_gemini,
    "lm_studio": wywolaj_lm_studio,
}


@traceable(name="klasyfikuj_tekst", run_type="chain")
def wywolaj_llm(prompt: str, provider: str, model_name: str, temperature: float) -> tuple[str, dict]:
    """Centralny dispatcher — wybiera właściwego klienta LLM i wywołuje go. Zwraca (odpowiedz, tokeny)."""
    fn = PROVIDER_DISPATCH.get(provider)
    if fn is None:
        raise ValueError(
            f"Nieznany provider: '{provider}'. "
            f"Dozwolone: {list(PROVIDER_DISPATCH)}"
        )
    return fn(prompt, model_name, temperature)


# ─────────────────────────────────────────────────────────────────────────────
# PARSOWANIE ODPOWIEDZI
# ─────────────────────────────────────────────────────────────────────────────

def parsuj_odpowiedz(raw: str) -> list[str]:
    """
    Parsuje surową odpowiedź LLM na listę znanych kategorii.

    Odporna na typowe defekty modeli:
      - owinięcie w JSON: {"output": "KAT1,KAT2"} lub {"categories": [...]}
      - trailing przecinek: "KAT1,KAT2,"
      - cudzysłowy i nawiasy kwadratowe: ["KAT1", "KAT2"]
      - obcięte nazwy (np. "PRAWORZĄ") — odrzucane jako nieznane
      - dodatkowe linie lub komentarze po odpowiedzi
    """
    tekst = raw.strip()

    # Wyodrębnij wartość z JSON jeśli model opakował odpowiedź
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
            pass  # nie jest poprawnym JSON — traktuj as-is

    # Weź tylko pierwszą niepustą linię (modele czasem dodają wyjaśnienia)
    tekst = next((ln.strip() for ln in tekst.splitlines() if ln.strip()), "")

    # Podziel po przecinku, wyczyść każdy token z niealfabetycznych znaków
    kandydaci = [
        re.sub(r"[^A-ZĄĆĘŁŃÓŚŹŻ_]", "", tok.strip().upper())
        for tok in tekst.split(",")
    ]

    # Zostaw tylko pełne, znane nazwy kategorii (obcięte odpadają)
    wynik = [k for k in kandydaci if k in KATEGORIE]
    return wynik or ["INNE"]


# ─────────────────────────────────────────────────────────────────────────────
# WCZYTYWANIE DANYCH
# ─────────────────────────────────────────────────────────────────────────────

def wczytaj_teksty(sciezka: str, limit: int) -> list[dict]:
    """
    Wczytuje plik z tekstami w formacie: id<TAB>tekst
    Zwraca listę słowników {id, tekst}, ograniczoną do `limit` wierszy.
    """
    rekordy = []
    with open(sciezka, encoding="utf-8") as f:
        for i, linia in enumerate(f):
            if i >= limit:
                break
            linia = linia.rstrip("\n")
            if not linia.strip():
                continue
            czesci = linia.split("\t", 1)
            if len(czesci) < 2:
                print(f"  [OSTRZEZENIE] Linia {i + 1} nie ma tabulatora — pomijam.")
                continue
            rekordy.append({"id": czesci[0].strip(), "tekst": czesci[1].strip()})
    return rekordy


def wczytaj_zloty_standard(sciezka: str) -> dict[str, list[str]]:
    """
    Wczytuje zloty standard w formacie: id<TAB>KAT1,KAT2,...
    Zwraca slownik {id -> [kategorie]}.
    """
    standard = {}
    with open(sciezka, encoding="utf-8") as f:
        for linia in f:
            linia = linia.rstrip("\n")
            if not linia.strip():
                continue
            czesci = linia.split("\t", 1)
            if len(czesci) < 2:
                continue
            rec_id = czesci[0].strip()
            kats   = [k.strip().upper() for k in czesci[1].split(",") if k.strip()]
            standard[rec_id] = kats
    return standard


# ─────────────────────────────────────────────────────────────────────────────
# OBLICZANIE METRYK
# ─────────────────────────────────────────────────────────────────────────────

def oblicz_metryki(
    wyniki:   dict[str, list[str]],
    standard: dict[str, list[str]],
) -> dict:
    """
    Oblicza pelny zestaw metryk dla multi-label classification.

    Metryki globalne:
      - exact match ratio    — % probek z identycznym zbiorem etykiet
      - hamming loss         — % blednie przypisanych etykiet (FP+FN / n*k)
      - precision micro      — srednia wazona po etykietach
      - recall micro         — srednia wazona po etykietach
      - F1 micro             — srednia wazona po etykietach
      - F1 macro             — srednia niewazona po kategoriach

    Metryki per kategoria:
      - precision, recall, F1, support

    Confusion pairs:
      - FP / FN per kategoria
      - TOP-15 par najczesciej mylonych (model powiedzial A zamiast B)
    """
    import numpy as np
    from sklearn.metrics import (
        f1_score,
        hamming_loss,
        precision_score,
        recall_score,
    )

    wspolne_id = [rid for rid in wyniki if rid in standard]
    n = len(wspolne_id)
    if n == 0:
        print("[BLAD] Brak wspolnych id miedzy wynikami a zlotym standardem.")
        return {}

    def binaryzuj(kats: list[str]) -> list[int]:
        return [1 if kat in kats else 0 for kat in KATEGORIE]

    y_true = np.array([binaryzuj(standard[rid]) for rid in wspolne_id])
    y_pred = np.array([binaryzuj(wyniki[rid])   for rid in wspolne_id])

    # ── Metryki globalne ────────────────────────────────────────────────────
    exact_match  = float(np.all(y_true == y_pred, axis=1).mean())
    hamming      = float(hamming_loss(y_true, y_pred))
    prec_micro   = float(precision_score(y_true, y_pred, average="micro",  zero_division=0))
    rec_micro    = float(recall_score(   y_true, y_pred, average="micro",  zero_division=0))
    f1_micro     = float(f1_score(       y_true, y_pred, average="micro",  zero_division=0))
    f1_macro     = float(f1_score(       y_true, y_pred, average="macro",  zero_division=0))

    # ── Metryki per kategoria ───────────────────────────────────────────────
    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per  = recall_score(   y_true, y_pred, average=None, zero_division=0)
    f1_per   = f1_score(       y_true, y_pred, average=None, zero_division=0)
    support  = y_true.sum(axis=0)

    per_kat = {
        kat: {
            "precision": round(float(prec_per[i]), 4),
            "recall":    round(float(rec_per[i]),  4),
            "f1":        round(float(f1_per[i]),   4),
            "support":   int(support[i]),
        }
        for i, kat in enumerate(KATEGORIE)
    }

    # ── Confusion pairs ─────────────────────────────────────────────────────
    #
    # FP (False Positive): model dodal kategorie, ktorej nie ma w standardzie
    # FN (False Negative): model pominil kategorie, ktora jest w standardzie
    #
    # Para mylonych (A -> B):
    #   A = kategoria przewidziana przez model (FP)
    #   B = kategoria, ktora powinna byc (FN)
    #   Dla kazdej probki: kazde A jest parowane z kazdym B.
    #
    fp_per: dict[str, int] = {}
    fn_per: dict[str, int] = {}

    for i, kat in enumerate(KATEGORIE):
        fp_per[kat] = int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum())
        fn_per[kat] = int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum())

    confusion_counter: Counter = Counter()

    for rid in wspolne_id:
        pred_set = set(wyniki[rid])
        true_set = set(standard[rid])
        fp_set   = pred_set - true_set   # model dodal blednie
        fn_set   = true_set - pred_set   # model pominil

        for fp_kat in fp_set:
            for fn_kat in fn_set:
                confusion_counter[(fp_kat, fn_kat)] += 1

    top_confusion_pairs = confusion_counter.most_common(15)

    return {
        # Globalne
        "n_probek":           n,
        "exact_match":        round(exact_match, 4),
        "hamming_loss":       round(hamming,     4),
        "precision_micro":    round(prec_micro,  4),
        "recall_micro":       round(rec_micro,   4),
        "f1_micro":           round(f1_micro,    4),
        "f1_macro":           round(f1_macro,    4),
        # Per kategoria
        "per_kategoria":      per_kat,
        # Confusion
        "fp_per":             fp_per,
        "fn_per":             fn_per,
        "top_confusion_pairs": top_confusion_pairs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WYSWIETLANIE RAPORTU
# ─────────────────────────────────────────────────────────────────────────────

def drukuj_raport(metryki: dict) -> None:
    """Wyswietla pelny raport ewaluacyjny w konsoli."""

    W   = 82
    SEP = "=" * W
    sep = "-" * W

    print(f"\n{SEP}")
    print("  RAPORT EWALUACJI -- POROWNANIE Z ZLOTYM STANDARDEM")
    print(SEP)

    # ── Metryki globalne ────────────────────────────────────────────────────
    print(f"\n  Probki do ewaluacji  : {metryki['n_probek']}")
    print()
    print(f"  {'Miara':<24}  {'Wartosc':>10}")
    print(f"  {'-'*24}  {'-'*10}")
    print(f"  {'Exact Match Ratio':<24}  {metryki['exact_match']:>10.4f}")
    print(f"  {'Hamming Loss':<24}  {metryki['hamming_loss']:>10.4f}")
    print(f"  {'Precision (micro)':<24}  {metryki['precision_micro']:>10.4f}")
    print(f"  {'Recall (micro)':<24}  {metryki['recall_micro']:>10.4f}")
    print(f"  {'F1 (micro)':<24}  {metryki['f1_micro']:>10.4f}")
    print(f"  {'F1 (macro)':<24}  {metryki['f1_macro']:>10.4f}")

    # ── Metryki per kategoria ───────────────────────────────────────────────
    C_KAT = 26
    C_NUM = 11

    print(f"\n{sep}")
    print("  SZCZEGOLY PER KATEGORIA")
    print(sep)

    print(
        f"  {'':>{C_KAT}}"
        f"  {'precision':>{C_NUM}}"
        f"  {'recall':>{C_NUM}}"
        f"  {'f1-score':>{C_NUM}}"
        f"  {'support':>{C_NUM}}"
    )
    print(f"  {'-'*C_KAT}  {'-'*C_NUM}  {'-'*C_NUM}  {'-'*C_NUM}  {'-'*C_NUM}")

    for kat in sorted(metryki["per_kategoria"]):
        s = metryki["per_kategoria"][kat]
        print(
            f"  {kat:>{C_KAT}}"
            f"  {s['precision']:>{C_NUM}.4f}"
            f"  {s['recall']:>{C_NUM}.4f}"
            f"  {s['f1']:>{C_NUM}.4f}"
            f"  {s['support']:>{C_NUM}}"
        )

    total_support = sum(v["support"] for v in metryki["per_kategoria"].values())
    print(f"  {'-'*C_KAT}  {'-'*C_NUM}  {'-'*C_NUM}  {'-'*C_NUM}  {'-'*C_NUM}")
    print(
        f"  {'micro avg':>{C_KAT}}"
        f"  {metryki['precision_micro']:>{C_NUM}.4f}"
        f"  {metryki['recall_micro']:>{C_NUM}.4f}"
        f"  {metryki['f1_micro']:>{C_NUM}.4f}"
        f"  {total_support:>{C_NUM}}"
    )
    print(
        f"  {'macro avg':>{C_KAT}}"
        f"  {'':>{C_NUM}}"
        f"  {'':>{C_NUM}}"
        f"  {metryki['f1_macro']:>{C_NUM}.4f}"
        f"  {total_support:>{C_NUM}}"
    )

    # ── Confusion pairs: FP / FN per kategoria ──────────────────────────────
    print(f"\n{sep}")
    print("  BLEDY PER KATEGORIA  (FP = dodano blednie | FN = pominieto)")
    print(sep)
    print(f"  {'':>{C_KAT}}  {'FP':>{C_NUM}}  {'FN':>{C_NUM}}")
    print(f"  {'-'*C_KAT}  {'-'*C_NUM}  {'-'*C_NUM}")

    for kat in sorted(metryki["per_kategoria"]):
        fp = metryki["fp_per"][kat]
        fn = metryki["fn_per"][kat]
        print(f"  {kat:>{C_KAT}}  {fp:>{C_NUM}}  {fn:>{C_NUM}}")

    # ── Confusion pairs: najczesciej mylone pary ────────────────────────────
    print(f"\n{sep}")
    print("  TOP MYLONYCH PAR  (model powiedzial A --> powinno byc B)")
    print(sep)

    if metryki["top_confusion_pairs"]:
        print(f"  {'Model przewidzial (A)':<28}  {'Powinno byc (B)':<28}  {'Ile razy':>8}")
        print(f"  {'-'*28}  {'-'*28}  {'-'*8}")
        for (pred_kat, true_kat), count in metryki["top_confusion_pairs"]:
            print(f"  {pred_kat:<28}  {true_kat:<28}  {count:>8}")
    else:
        print("  Brak pomylek — model nie mial zadnych blednych par kategorii.")

    print(SEP)


def drukuj_raport_kosztow(
    suma_tokenow: dict[str, int],
    suma_koszt_usd: float,
    n_tekstow: int,
    cennik: dict[str, float],
) -> None:
    """Wyswietla podsumowanie zuzycia tokenow i kosztu w konsoli."""

    W   = 82
    sep = "-" * W

    print(f"\n{sep}")
    print("  ZUZYCIE TOKENOW I KOSZT")
    print(sep)
    print(f"  Cennik (USD / 1M tok.)  : input ${cennik['input']:.3f}  |  output ${cennik['output']:.3f}")
    print()
    print(f"  {'Miara':<28}  {'Wartosc':>15}")
    print(f"  {'-'*28}  {'-'*15}")
    print(f"  {'Tokeny wejsciowe (suma)':<28}  {suma_tokenow['input_tokens']:>15,}")
    print(f"  {'Tokeny wyjsciowe (suma)':<28}  {suma_tokenow['output_tokens']:>15,}")
    print(f"  {'Tokeny razem (suma)':<28}  {suma_tokenow['total_tokens']:>15,}")
    if n_tekstow > 0:
        print(f"  {'Sr. tokenow / tekst':<28}  {suma_tokenow['total_tokens'] / n_tekstow:>15,.1f}")
    print(f"  {'-'*28}  {'-'*15}")
    print(f"  {'Koszt calkowity (USD)':<28}  ${suma_koszt_usd:>14,.4f}")
    if n_tekstow > 0:
        print(f"  {'Sr. koszt / tekst (USD)':<28}  ${suma_koszt_usd / n_tekstow:>14,.6f}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# ZAPIS WYNIKOW
# ─────────────────────────────────────────────────────────────────────────────

class _Duplikator:
    """
    Kieruje wypisywany tekst jednoczesnie na konsole i do bufora, zeby caly
    przebieg dalo sie pozniej zapisac do pliku bez powtarzania print().
    """

    def __init__(self, konsola, bufor):
        self._konsola = konsola
        self._bufor = bufor

    def write(self, tekst: str) -> int:
        self._bufor.write(tekst)
        return self._konsola.write(tekst)

    def flush(self) -> None:
        self._konsola.flush()

    def isatty(self) -> bool:
        return False


def zapisz_raport(
    tresc:   str,
    run_id:  str,
    prefiks: str = "",
    katalog: str = "results",
) -> str:
    """
    Zapisuje pelny przebieg konsoli (konfiguracja, klasyfikacje, raport
    ewaluacji, zuzycie tokenow) do osobnego pliku tekstowego.
    """
    Path(katalog).mkdir(parents=True, exist_ok=True)
    data_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    czolo    = f"{prefiks}_" if prefiks else ""
    nazwa    = f"{czolo}raport_{run_id}_{data_str}.txt"
    sciezka  = os.path.join(katalog, nazwa)

    with open(sciezka, "w", encoding="utf-8") as f:
        f.write(tresc)

    print(f"  Raport zapisany: {sciezka}")
    return sciezka


def slug_modelu(model_name: str) -> str:
    """
    Zamienia nazwe modelu na fragment bezpieczny w nazwie pliku:
    'qwen/qwen3.6-27b' -> 'qwen3.6-27b', 'qwen3.5-9b@q6_k' -> 'qwen3.5-9b-q6_k'
    """
    slug = model_name.split("/")[-1]
    return re.sub(r"[^0-9A-Za-z._-]+", "-", slug).strip("-")


def buduj_prefiks(model_name: str, metoda: str, tokeny_wejsciowe: int) -> str:
    """
    Prefiks nazw plikow wynikowych: model, metoda promptu i liczba tokenow
    wejsciowych. Ta ostatnia jednoznacznie identyfikuje wersje definicji
    kategorii — przy stalym zbiorze tekstow i stalej dlugosci czesci stalej
    promptu kazda wersja definicji daje inna sume.
    """
    return f"{slug_modelu(model_name)}_{metoda}_{tokeny_wejsciowe}"


def zapisz_wyniki(
    wyniki:  dict[str, list[str]],
    run_id:  str,
    prefiks: str = "",
    katalog: str = "results",
) -> str:
    """
    Zapisuje wyniki w formacie zlotego standardu: id<TAB>KAT1,KAT2,...
    Nazwa pliku: <model>_<metoda>_<tokeny>_wynik_ewaluacji_<run_id>_<data>.txt
    """
    Path(katalog).mkdir(parents=True, exist_ok=True)
    data_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    czolo    = f"{prefiks}_" if prefiks else ""
    nazwa    = f"{czolo}wynik_ewaluacji_{run_id}_{data_str}.txt"
    sciezka  = os.path.join(katalog, nazwa)

    with open(sciezka, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for rid, kats in wyniki.items():
            writer.writerow([rid, ",".join(kats)])

    print(f"  Wyniki zapisane: {sciezka}")
    return sciezka


def zapisz_tokeny_i_koszty(
    tokeny_per_tekst: dict[str, dict],
    cennik:           dict[str, float],
    run_id:           str,
    prefiks:          str = "",
    katalog:          str = "results",
) -> str:
    """
    Zapisuje zuzycie tokenow i koszt per tekst do pliku CSV:
      id, input_tokens, output_tokens, total_tokens, koszt_usd

    Plik pozwala na pozniejsza analize (np. przeliczenie kosztu innym
    cennikiem albo zsumowanie po batchach).
    """
    Path(katalog).mkdir(parents=True, exist_ok=True)
    data_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    czolo    = f"{prefiks}_" if prefiks else ""
    nazwa    = f"{czolo}tokeny_koszty_{run_id}_{data_str}.csv"
    sciezka  = os.path.join(katalog, nazwa)

    with open(sciezka, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "input_tokens", "output_tokens", "total_tokens", "koszt_usd"])
        for rid, t in tokeny_per_tekst.items():
            koszt = oblicz_koszt(t, cennik)
            writer.writerow([
                rid,
                t["input_tokens"],
                t["output_tokens"],
                t["total_tokens"],
                round(koszt, 6),
            ])

    print(f"  Tokeny i koszty zapisane: {sciezka}")
    return sciezka


# ─────────────────────────────────────────────────────────────────────────────
# GLOWNA FUNKCJA
# ─────────────────────────────────────────────────────────────────────────────

def main(
    sciezka_teksty:   str,
    sciezka_standard: str,
    provider:         str            = WYBRANY_PROVIDER,
    model_override:   Optional[str]  = None,
    metoda:           str            = METODA_PROMPTU,
    limit:            int            = LIMIT_WIERSZY,
    temperatura:      float          = TEMPERATURA,
) -> tuple[str, str]:
    """
    Glowna petla ewaluacyjna: laduje dane, klasyfikuje, ewaluuje, zapisuje.
    Zwraca (run_id, prefiks) do nazwania pliku z raportem.
    """

    # ── Inicjalizacja srodowiska ─────────────────────────────────────────────
    load_dotenv()

    print("=" * 80)
    print("  EWALUATOR WYPOWIEDZI SEJMOWYCH  (bez MLflow)")
    print("=" * 80)

    # ── Wybor i walidacja modelu ─────────────────────────────────────────────
    if provider not in MODEL_CONFIG:
        raise ValueError(
            f"Nieznany provider: '{provider}'. "
            f"Dozwolone: {list(MODEL_CONFIG)}"
        )

    cfg = MODEL_CONFIG[provider]

    if provider == "lm_studio":
        model_name = model_override or os.environ.get("LM_STUDIO_MODEL") or cfg["model_name"]
        lm_url     = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
        print(f"  LM Studio URL      : {lm_url}")
    else:
        model_name = model_override or cfg["model_name"]

    cennik = pobierz_cennik(provider, model_name)

    print(f"  Provider           : {provider}")
    print(f"  Model              : {model_name}")
    print(f"  Temperatura        : {temperatura}")
    print(f"  Limit wierszy      : {limit}")
    print(f"  Metoda promptu     : {metoda}")
    print(f"  Cennik (per 1M tok): input ${cennik['input']:.3f} / output ${cennik['output']:.3f}")

    # ── Wczytanie szablonu promptu ───────────────────────────────────────────
    szablon = wczytaj_szablon_promptu(metoda)
    print(f"  Szablon            : methods/{metoda}.txt  [OK]")

    # ── Wczytanie danych ─────────────────────────────────────────────────────
    print()
    print(f"  Teksty             : {sciezka_teksty}")
    teksty = wczytaj_teksty(sciezka_teksty, limit)
    print(f"  Wczytano rekordow  : {len(teksty)}")

    print(f"  Zloty standard     : {sciezka_standard}")
    standard = wczytaj_zloty_standard(sciezka_standard)
    print(f"  Rekordow w std.    : {len(standard)}")
    print()

    # ── Przebieg ─────────────────────────────────────────────────────────────
    wyniki: dict[str, list[str]] = {}
    tokeny_per_tekst: dict[str, dict] = {}

    # Identyfikator przebiegu zastepuje run_id z MLflow — sluzy wylacznie do
    # nazywania plikow w results/, wiec ma ten sam format co dotychczas.
    run_id = uuid.uuid4().hex
    print(f"  Run ID             : {run_id}")
    print("-" * 80)

    sukces = bledy = 0
    suma_tokenow = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    suma_koszt_usd = 0.0
    t_start = time.time()

    # ── Petla klasyfikacji — kazdy tekst = osobne zapytanie do LLM ───────────
    for i, rekord in enumerate(teksty):
        rid    = rekord["id"]
        tekst  = rekord["tekst"]
        prompt = buduj_prompt(tekst, szablon)

        print(f"  [{i + 1:>3}/{len(teksty)}] {rid} ...", end=" ", flush=True)

        t0 = time.time()
        try:
            surowa, tokeny = wywolaj_llm(prompt, provider, model_name, temperatura)
            kategorie      = parsuj_odpowiedz(surowa)
            latencja       = round(time.time() - t0, 3)
            koszt          = oblicz_koszt(tokeny, cennik)

            wyniki[rid]           = kategorie
            tokeny_per_tekst[rid] = tokeny
            sukces += 1

            for k in suma_tokenow:
                suma_tokenow[k] += tokeny.get(k, 0)
            suma_koszt_usd += koszt

            print(
                f"-> {','.join(kategorie)}  "
                f"({latencja}s, {tokeny['total_tokens']} tok., ${koszt:.6f})"
            )

        except Exception as exc:
            bledy      += 1
            wyniki[rid] = ["INNE"]
            tokeny_per_tekst[rid] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            print(f"[BLAD] {exc}")

    czas_calkowity = time.time() - t_start

    # ── Zapis wynikow ────────────────────────────────────────────────────────
    print()
    prefiks = buduj_prefiks(model_name, metoda, suma_tokenow["input_tokens"])
    zapisz_wyniki(wyniki, run_id, prefiks)
    zapisz_tokeny_i_koszty(tokeny_per_tekst, cennik, run_id, prefiks)

    # ── Ewaluacja ────────────────────────────────────────────────────────────
    metryki = oblicz_metryki(wyniki, standard)
    if metryki:
        drukuj_raport(metryki)

    # ── Raport tokenow i kosztow ─────────────────────────────────────────────
    drukuj_raport_kosztow(suma_tokenow, suma_koszt_usd, sukces, cennik)

    print(f"\n  Sukces: {sukces}  |  Bledy: {bledy}")
    print(f"  Czas   : {czas_calkowity / 60:.1f} min "
          f"({sukces / czas_calkowity:.3f} tekst/s)")
    print("=" * 80)

    return run_id, prefiks


# ─────────────────────────────────────────────────────────────────────────────
# PUNKT WEJSCIA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ewaluacja klasyfikacji tematycznej wypowiedzi sejmowych",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--teksty",
        default="../data/teksty.txt",
        help="Plik z tekstami (id<TAB>tekst)",
    )
    parser.add_argument(
        "--standard",
        default="../data/standard.txt",
        help="Zloty standard (id<TAB>KAT1,KAT2,...)",
    )
    parser.add_argument(
        "--provider",
        default=WYBRANY_PROVIDER,
        choices=list(MODEL_CONFIG),
        help="Provider LLM",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Nadpisuje model_name z MODEL_CONFIG (przydatne dla lm_studio)",
    )
    parser.add_argument(
        "--metoda",
        default=METODA_PROMPTU,
        help="Metoda promptu: nazwa pliku z methods/ bez rozszerzenia (np. zero_shot)",
    )
    parser.add_argument(
        "--limit",
        default=LIMIT_WIERSZY,
        type=int,
        help="Liczba pierwszych rekordow do przetworzenia",
    )
    parser.add_argument(
        "--temp",
        default=TEMPERATURA,
        type=float,
        help="Temperatura modelu (0.0 = deterministyczna)",
    )

    args = parser.parse_args()

    # Caly przebieg trafia rownolegle na konsole i do bufora, zeby zapisac
    # go pozniej w pliku raportu.
    bufor = io.StringIO()
    konsola = sys.stdout
    sys.stdout = _Duplikator(konsola, bufor)
    try:
        run_id, prefiks = main(
            sciezka_teksty   = args.teksty,
            sciezka_standard = args.standard,
            provider         = args.provider,
            model_override   = args.model,
            metoda           = args.metoda,
            limit            = args.limit,
            temperatura      = args.temp,
        )
    finally:
        sys.stdout = konsola

    zapisz_raport(bufor.getvalue(), run_id, prefiks)