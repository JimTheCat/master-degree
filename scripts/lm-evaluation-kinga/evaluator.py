"""
Program ewaluacyjny wypowiedzi sejmowych z klasyfikacją tematyczną.
Obsługuje modele po API: OpenAI GPT-4, Anthropic Claude, Google Gemini
oraz modele odpalone lokalnie przez LM Studio (OpenAI-kompatybilne API).
Integracja z MLflow i LangSmith.
"""

import os
import re
import csv
import time
import datetime
import argparse
from pathlib import Path

import mlflow
from dotenv import load_dotenv
from langsmith import traceable

# ─────────────────────────────────────────────────────────────────────────────
# STAŁE KONFIGURACYJNE
# ─────────────────────────────────────────────────────────────────────────────

KATEGORIE_DEF = {
    "GOSPODARKA": (
        "Kwestie związane z budżetem państwa, podatkami, inflacją, rynkiem pracy, "
        "przedsiębiorczością, handlem. Wszystko związane z wykorzystaniem i robieniem "
        "przez państwo pieniędzy. Przykłady: VAT, PIT, Inflacja, Miejsca pracy, Budżet, "
        "Nasze firmy, PKB, Podatki"
    ),
    "POLITYKA_SPOŁECZNA": (
        "Kwestie związane z systemem emerytalnym, pomocą socjalną, rodziną, "
        "niepełnosprawnościami, ochroną praw społecznych, kwestie dotyczące wsparcie "
        "obywateli. Przykłady: Emerytury, 500 plus, ZUS, Pomoc dla rodzin, Urlop "
        "rodzicielski, Ubóstwo, Programy społeczne, Bezdomność"
    ),
    "BEZPIECZEŃSTWO": (
        "Kwestie związane z obronnością, wojskiem, policją, służbami specjalnymi, "
        "bezpieczeństwem granic, cyberbezpieczeństwa, polityką zagraniczną w kontekście "
        "militarnym. Przykłady: Armia, Wojsko, Wydatki na obronność, Sojusze, NATO, "
        "Policja, Ataki hakerskie, Cyberbezpieczeństwo"
    ),
    "ŚWIATOPOGLĄD": (
        "Kwestie związane z prawami reprodukcyjnymi, prawami mniejszości, religią, "
        "kulturą, wartościami, silnymi prywatnymi poglądami. Przykłady: Aborcja, Prawa "
        "kobiet, Związki partnerskie, Religia w szkole, Kościół, Wiara, Etyka, Kultura, LGBT+"
    ),
    "PRAWORZĄDNOŚĆ": (
        "Teksty związane z prawem, wymiarem sprawiedliwości, ustawodawstwem, regulacjami, "
        "sądami i procedurami prawnymi. Przykłady: Konstytucja, Wolne sądy, Trybunał, "
        "Praworządność, Neo-sędziowie, prawo, egzekwowanie, regulacje"
    ),
    "EDUKACJA_NAUKA": (
        "Kwestie związane z systemem oświaty (szkoły, przedszkola), szkolnictwem wyższym, "
        "badaniami naukowymi, programami kształcenia i stypendiami. Przykłady: Nauczyciele, "
        "Szkoły, Uniwersytety, Podstawa programowa, Uczeń, Program nauczania, Badania, Nauka"
    ),
    "ŚRODOWISKO": (
        "Kwestie związane z ochroną klimatu, transformacją energetyczną, OZE, smogiem, "
        "ochroną przyrody, gospodarką odpadami, ochrona dziedzictwa i zasobów. Przykłady: "
        "Klimat, Węgiel, OZE, Smog, Parki narodowe, Odpady, Spalarnie śmieci, "
        "Zrównoważony rozwój, Natura2000, Odnawialne źródła energii"
    ),
    "SPRAWY_ZAGRANICZNE": (
        "Kwestie dotyczące polityki międzynarodowej, dyplomacji, współpracy "
        "międzynarodowej, traktatów i konfliktów zagranicznych. Przykłady: Dyplomacja, "
        "Unia Europejska, Stany Zjednoczone, NATO, ONZ, Strefa Shengen, Uchodźcy, "
        "Ambasada, Rosja, Ukraina, Sankcje"
    ),
    "ROLNICTWO": (
        "Kwestie związane z rolnictwem, z uprawą roli, ochroną gleby, hodowlą zwierząt "
        "gospodarczych, gospodarką wodną, skupy, pestycydy. Przykłady: Pestycydy, Nawóz, "
        "Melioracje, Zrównoważony rozwój, GMO, Hodowla, Gospodarstwo rolne, Rolnicy, Rybołówstwo"
    ),
    "POLITYKA_LOKALNA": (
        "Odniesienia do specyficznych problemów regionów, samorządów, inwestycji lokalnych. "
        "Przykłady: Samorządy, Prezydent miasta, Drogi lokalne, Inwestycje w..., Prawo "
        "administracyjne, Udział społeczny, Gospodarka przestrzenna"
    ),
    "ADMINISTRACJA": (
        "Opis faktycznego działania urzędów, przepisów administracyjnych, procedur, "
        "finansowania, kompetencji instytucji publicznych, kwestii formalno-prawnych. "
        "Przykłady: Urząd, Ministerstwo, Procedura, Regulacja administracyjna, Budżet urzędu"
    ),
    "INFRASTRUKTURA": (
        "Zabudowa krajowa i lokalna, Transport Publiczny i drogowy, drogi, autostrady, "
        "szpitale, budynki publiczne, poruszanie się po drogach. Przykłady: Autostrada, "
        "Most, Ulica, Ruch drogowy, Transport publiczny, Budowa szpitali, PKP, "
        "Korytarz życia, Inwestycja drogowa"
    ),
    "HISTORIA_PAMIĘĆ_NARODOWA": (
        "Teksty dotyczące historii, pamięci narodowej, obchodów rocznic, bohaterów "
        "narodowych, muzeów. Przykłady: Historia, IPN, Rocznica, Bohater narodowy, "
        "Pamięć narodowa, komunizm, II Wojna Światowa, Powstanie, Pomnik"
    ),
    "ZDROWIE": (
        "Teksty dotyczące ochrony zdrowia, systemu opieki zdrowotnej, chorób, leczenia "
        "i profilaktyki, lekarzy i szpitali. Przykłady: Szpital, Pacjent, Choroba, "
        "Lekarz, NFZ, Zdrowie publiczne, Epidemia, Szczepienie"
    ),
    "INNE": (
        "Teksty, które nie mieszczą się w żadnej z powyższych kategorii, np. Wypowiedzi "
        "organizacyjne marszałka, komentarze polityczne bez merytorycznego tematu. "
        "Przykłady: Sprawozdanie komisji, Głosowanie, Komentarz, Dyskusja, Przemówienie, "
        "Proszę o zabranie głosu"
    ),
}

KATEGORIE = list(KATEGORIE_DEF.keys())

# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#   SEKCJA KONFIGURACJI TESTOWEJ — ZMIEŃ TE WARTOŚCI PRZED URUCHOMIENIEM
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

# Ile pierwszych linii tekstu wysłać do LLM.
# Zmień na 100 lub 500 do większych testów, na 500 do produkcji.
LIMIT_WIERSZY = 100  # ← ZMIEŃ TU (10 | 100 | 500)

# Temperatura modelu. Dla ewaluacji klasyfikacyjnej zalecana wartość to 0.0
# (deterministyczna, powtarzalna odpowiedź). Zmień na wyższą dla kreatywności.
TEMPERATURA = 0.3  # ← ZMIEŃ TU (zalecana dla klasyfikacji: 0.0)

# Wybór modelu: "openai" | "anthropic" | "gemini" | "lm_studio"
WYBRANY_PROVIDER = "lm_studio"  # ← ZMIEŃ TU

# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURACJA MODELI — dobrane pod kątem obsługi języka polskiego
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CONFIG = {
    # GPT-4o — najlepszy model OpenAI do polskiego, szybszy i tańszy niż o1
    "openai": {
        # "model_name": "gpt-3.5-turbo",
        "model_name": "gpt-4o",
        "provider": "openai",
    },
    # Claude Sonnet 4 — bardzo dobra obsługa języka polskiego, dobry balans
    # kosztów i jakości w ofercie Anthropic
    "anthropic": {
        "model_name": "claude-sonnet-4-5",
        "provider": "anthropic",
    },
    # Gemini 2.0 Flash — szybki, tani, dobry dla polskiego (Thinking opcjonalnie)
    "gemini": {
        "model_name": "gemini-2.5-flash",  # aktualny stabilny model dostępny dla nowych użytkowników
        "provider": "gemini",
    },
    # LM Studio — lokalny serwer z OpenAI-kompatybilnym API.
    # model_name: identyfikator modelu widoczny w LM Studio (zakładka "Local Server").
    # Domyślnie ustawiony na gemma; zmień na "bielik" lub inny załadowany model.
    # Nadpisywalne przez zmienne środowiskowe LM_STUDIO_MODEL i LM_STUDIO_URL w .env
    "lm_studio": {
        "model_name": "gemma-3-12b-it",  # ← ZMIEŃ na nazwę modelu z LM Studio
        "provider": "lm_studio",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# BUDOWANIE PROMPTU
# ─────────────────────────────────────────────────────────────────────────────

def buduj_prompt(tekst: str) -> str:
    """Buduje prompt klasyfikacyjny dla danego tekstu sejmowego."""
    definicje = "\n".join(
        f"- {kat}: {opis}" for kat, opis in KATEGORIE_DEF.items()
    )
    return f"""Jesteś ekspertem od analizy polskiego dyskursu parlamentarnego.
Twoim zadaniem jest przypisanie kategorii tematycznych do poniższej wypowiedzi sejmowej.

DEFINICJE KATEGORII:
{definicje}

ZASADY:
1. Wybierz TYLKO kategorie z powyższej listy (dokładna pisownia).
2. Możesz wybrać kilka kategorii, jeśli tekst dotyczy wielu tematów.
3. Wskaż kategorię INNE TYLKO i wyłącznie, gdy inne kategorie nie występują.
4. Zwróć WYŁĄCZNIE nazwy kategorii oddzielone przecinkami. Żadnego komentarza, żadnych wyjaśnień.
5. Przykład poprawnej odpowiedzi: GOSPODARKA,POLITYKA_SPOŁECZNA

WYPOWIEDŹ:
{tekst[:1200]}

ODPOWIEDŹ (tylko kategorie):"""


# ─────────────────────────────────────────────────────────────────────────────
# KLIENTY LLM — każdy provider ma osobną funkcję wywołującą API
# Dekorator @traceable rejestruje każde wywołanie w LangSmith niezależnie
# od użytego providera — widoczne jako osobny span z promptem i odpowiedzią.
# ─────────────────────────────────────────────────────────────────────────────

@traceable(name="llm_openai", run_type="llm")
def wywolaj_openai(prompt: str, model_name: str, temperature: float) -> str:
    """Wywołuje OpenAI API i zwraca surową odpowiedź tekstową."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2000,
    )
    return response.choices[0].message.content.strip()


@traceable(name="llm_anthropic", run_type="llm")
def wywolaj_anthropic(prompt: str, model_name: str, temperature: float) -> str:
    """Wywołuje Anthropic API i zwraca surową odpowiedź tekstową."""
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=model_name,
        max_tokens=2000,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()


@traceable(name="llm_gemini", run_type="llm")
def wywolaj_gemini(prompt: str, model_name: str, temperature: float) -> str:
    """Wywołuje Google Gemini API (nowe SDK google-genai) i zwraca surową odpowiedź."""
    from google import genai
    from google.genai import types
    client = genai.Client(
        api_key=os.environ["GOOGLE_API_KEY"],
        http_options={"api_version": "v1"}
    )
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=2000,
        ),
    )
    return response.text.strip()


@traceable(name="llm_lm_studio", run_type="llm")
def wywolaj_lm_studio(prompt: str, model_name: str, temperature: float) -> str:
    """
    Wywołuje lokalny model przez LM Studio używając OpenAI-kompatybilnego API.

    Konfiguracja w .env:
      LM_STUDIO_URL   — adres serwera LM Studio (domyślnie http://localhost:1234/v1)
      LM_STUDIO_MODEL — opcjonalne nadpisanie nazwy modelu bez edycji kodu
                        (jeśli ustawione, ma priorytet nad model_name z MODEL_CONFIG)

    Jak sprawdzić nazwę modelu w LM Studio:
      Otwórz LM Studio → zakładka "Local Server" → pole "Model" — skopiuj identyfikator.
    """
    from openai import OpenAI

    # Odczyt URL i opcjonalnego nadpisania modelu ze zmiennych środowiskowych
    base_url    = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
    model_final = os.environ.get("LM_STUDIO_MODEL", model_name)

    # LM Studio nie wymaga prawdziwego klucza API — wymagana jest jednak
    # niepusta wartość, żeby klient OpenAI nie rzucił błędu walidacji.
    client = OpenAI(base_url=base_url, api_key="lm-studio")

    response = client.chat.completions.create(
        model=model_final,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


# Dispatcher — wybór providera na podstawie konfiguracji
PROVIDER_DISPATCH = {
    "openai":    wywolaj_openai,
    "anthropic": wywolaj_anthropic,
    "gemini":    wywolaj_gemini,
    "lm_studio": wywolaj_lm_studio,
}


@traceable(name="klasyfikuj_tekst", run_type="chain")
def wywolaj_llm(prompt: str, provider: str, model_name: str, temperature: float) -> str:
    """Centralny punkt wywołania LLM — deleguje do właściwego klienta.
    Owinięty w @traceable jako nadrzędny span łańcucha w LangSmith."""
    fn = PROVIDER_DISPATCH.get(provider)
    if fn is None:
        raise ValueError(f"Nieznany provider: {provider}. Dozwolone: {list(PROVIDER_DISPATCH)}")
    return fn(prompt, model_name, temperature)


# ─────────────────────────────────────────────────────────────────────────────
# PARSOWANIE ODPOWIEDZI
# ─────────────────────────────────────────────────────────────────────────────

def parsuj_odpowiedz(raw: str) -> list[str]:
    """
    Parsuje odpowiedź modelu i zwraca listę znanych kategorii.
    Usuwa śmieci (białe znaki, cudzysłowy, komentarze po przecinku).
    """
    kandydaci = [tok.strip().upper() for tok in raw.split(",")]
    # Wyciągnij tylko alphanumeryczne + podkreślnik (usuwa ewentualne brudne znaki)
    kandydaci = [re.sub(r"[^A-ZĄĆĘŁŃÓŚŹŻ_]", "", k) for k in kandydaci]
    # Filtruj tylko znane kategorie
    return [k for k in kandydaci if k in KATEGORIE] or ["INNE"]


# ─────────────────────────────────────────────────────────────────────────────
# WCZYTYWANIE DANYCH
# ─────────────────────────────────────────────────────────────────────────────

def wczytaj_teksty(sciezka: str, limit: int) -> list[dict]:
    """
    Wczytuje plik z tekstami w formacie 'id<TAB>tekst'.
    Zwraca listę słowników {id, tekst}.
    Ogranicza do `limit` pierwszych wierszy.
    """
    rekordy = []
    with open(sciezka, encoding="utf-8") as f:
        for i, linia in enumerate(f):
            if i >= limit:
                break
            linia = linia.rstrip("\n")
            if not linia.strip():
                continue
            # Podział po pierwszej tabulacji
            czesci = linia.split("\t", 1)
            if len(czesci) < 2:
                print(f"  [OSTRZEŻENIE] Linia {i+1} nie ma tabulacji: {linia[:60]}")
                continue
            rekordy.append({"id": czesci[0].strip(), "tekst": czesci[1].strip()})
    return rekordy


def wczytaj_zloty_standard(sciezka: str) -> dict[str, list[str]]:
    """
    Wczytuje złoty standard w formacie 'id<TAB>KAT1,KAT2,...'.
    Zwraca słownik {id -> [kategorie]}.
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
            kats = [k.strip().upper() for k in czesci[1].split(",") if k.strip()]
            standard[rec_id] = kats
    return standard


# ─────────────────────────────────────────────────────────────────────────────
# EWALUACJA — miary jakości
# ─────────────────────────────────────────────────────────────────────────────

def oblicz_metryki(wyniki: dict[str, list[str]], standard: dict[str, list[str]]) -> dict:
    """
    Oblicza miary ewaluacyjne dla multi-label classification:
    - zgodność (exact match)
    - precision, recall, F1 (micro i macro)
    - precision, recall, F1, support, kappa, alpha per kategoria
    - Cohen's Kappa (binary per kategoria, uśredniony)
    - Krippendorff's Alpha (uproszczony, nominal, per kategoria i uśredniony)
    """
    from sklearn.metrics import precision_score, recall_score, f1_score, cohen_kappa_score
    import numpy as np

    # Zbieramy tylko id, które są w obu zbiorach
    wspolne_id = [rid for rid in wyniki if rid in standard]
    n = len(wspolne_id)
    if n == 0:
        print("[BŁĄD] Brak wspólnych id między wynikami a złotym standardem.")
        return {}

    # Budujemy macierze binarne: każda kolumna = kategoria, każdy wiersz = próbka
    def binaryzuj(id_lista):
        """Zamienia listę kategorii na wektor binarny."""
        return [1 if kat in id_lista else 0 for kat in KATEGORIE]

    y_true = np.array([binaryzuj(standard[rid]) for rid in wspolne_id])
    y_pred = np.array([binaryzuj(wyniki[rid])   for rid in wspolne_id])

    # ── 1. Zgodność (exact match) ──────────────────────────────────────────
    zgodnosc = float(np.all(y_true == y_pred, axis=1).mean())

    # ── 2. Precision, Recall, F1 — micro i macro ──────────────────────────
    prec_micro  = precision_score(y_true, y_pred, average="micro",  zero_division=0)
    rec_micro   = recall_score(  y_true, y_pred, average="micro",  zero_division=0)
    f1_micro    = f1_score(      y_true, y_pred, average="micro",  zero_division=0)
    prec_macro  = precision_score(y_true, y_pred, average="macro",  zero_division=0)
    rec_macro   = recall_score(  y_true, y_pred, average="macro",  zero_division=0)
    f1_macro    = f1_score(      y_true, y_pred, average="macro",  zero_division=0)

    # ── 3. Metryki per kategoria ────────────────────────────────────────────
    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per  = recall_score(   y_true, y_pred, average=None, zero_division=0)
    f1_per   = f1_score(       y_true, y_pred, average=None, zero_division=0)
    # support = liczba prawdziwie pozytywnych próbek w złotym standardzie
    support  = y_true.sum(axis=0)

    # ── 4. Cohen's Kappa per kategoria ────────────────────────────────────
    kappa_per = []
    for i in range(len(KATEGORIE)):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        if len(set(col_true)) == 1 and len(set(col_pred)) == 1:
            kappa_per.append(1.0 if col_true[0] == col_pred[0] else 0.0)
        else:
            kappa_per.append(cohen_kappa_score(col_true, col_pred))
    kappa_srednia = float(np.mean(kappa_per))

    # ── 5. Krippendorff's Alpha per kategoria ─────────────────────────────
    def kripp_alpha_binary(v1, v2):
        """Uproszczona alpha Krippendorffa dla danych binarnych (2 annotatorów)."""
        n_ = len(v1)
        observed_disagreement = sum(v1[j] != v2[j] for j in range(n_)) / n_
        p1 = sum(v1) / n_
        p2 = sum(v2) / n_
        expected_disagreement = p1 * (1 - p2) + (1 - p1) * p2
        if expected_disagreement == 0:
            return 1.0 if observed_disagreement == 0 else 0.0
        return 1.0 - observed_disagreement / expected_disagreement

    alpha_per = []
    for i in range(len(KATEGORIE)):
        col_true = list(y_true[:, i])
        col_pred = list(y_pred[:, i])
        alpha_per.append(kripp_alpha_binary(col_true, col_pred))
    alpha_srednia = float(np.mean(alpha_per))

    # ── Składamy słownik per kategoria ────────────────────────────────────
    per_kat = {}
    for i, kat in enumerate(KATEGORIE):
        per_kat[kat] = {
            "precision": round(float(prec_per[i]), 4),
            "recall":    round(float(rec_per[i]),  4),
            "f1":        round(float(f1_per[i]),   4),
            "support":   int(support[i]),
            "kappa":     round(float(kappa_per[i]),  4),
            "alpha":     round(float(alpha_per[i]),  4),
        }

    return {
        "n_probek":        n,
        "zgodnosc":        round(zgodnosc,    4),
        "precision_micro": round(prec_micro,  4),
        "recall_micro":    round(rec_micro,   4),
        "f1_micro":        round(f1_micro,    4),
        "precision_macro": round(prec_macro,  4),
        "recall_macro":    round(rec_macro,   4),
        "f1_macro":        round(f1_macro,    4),
        "kappa_srednia":   round(kappa_srednia, 4),
        "alpha_kripp":     round(alpha_srednia, 4),
        "per_kategoria":   per_kat,
    }


def drukuj_metryki(metryki: dict) -> None:
    """Wyświetla pełny raport ewaluacyjny w konsoli w czytelnym formacie tabelarycznym."""
    if not metryki:
        return

    W = 86  # szerokość tabeli

    print("\n" + "═" * W)
    print("📏  RAPORT EWALUACJI — PORÓWNANIE Z ZŁOTYM STANDARDEM")
    print("═" * W)

    # ── Metryki zbiorcze ──────────────────────────────────────────────────
    print(f"\n  Próbek do ewaluacji   : {metryki['n_probek']}")
    print(f"  Zgodność (exact match): {metryki['zgodnosc']:.4f}")
    print()
    print(f"  {'Miara':<22}  {'micro':>8}  {'macro':>8}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}")
    print(f"  {'Precision':<22}  {metryki['precision_micro']:>8.4f}  {metryki['precision_macro']:>8.4f}")
    print(f"  {'Recall':<22}  {metryki['recall_micro']:>8.4f}  {metryki['recall_macro']:>8.4f}")
    print(f"  {'F1-score':<22}  {metryki['f1_micro']:>8.4f}  {metryki['f1_macro']:>8.4f}")
    print()
    print(f"  Cohen's Kappa (śr.)   : {metryki['kappa_srednia']:.4f}")
    print(f"  Krippendorff Alpha (śr.): {metryki['alpha_kripp']:.4f}")

    # ── Tabela per kategoria ──────────────────────────────────────────────
    print()
    print("  Szczegóły per kategoria:")
    print()

    # Nagłówek — dopasowany do szerokości nazw kategorii
    COL_KAT  = 26   # szerokość kolumny nazwy kategorii
    COL_NUM  = 10   # szerokość kolumn numerycznych
    header = (
        f"  {'':>{COL_KAT}}"
        f"  {'precision':>{COL_NUM}}"
        f"  {'recall':>{COL_NUM}}"
        f"  {'f1-score':>{COL_NUM}}"
        f"  {'support':>{COL_NUM}}"
        f"  {'kappa':>{COL_NUM}}"
        f"  {'alpha':>{COL_NUM}}"
    )
    separator = "  " + "-" * (COL_KAT + 6 * (COL_NUM + 2))
    print(header)
    print(separator)

    # Wiersze kategorii posortowane alfabetycznie
    for kat in sorted(metryki["per_kategoria"]):
        s = metryki["per_kategoria"][kat]
        print(
            f"  {kat:>{COL_KAT}}"
            f"  {s['precision']:>{COL_NUM}.4f}"
            f"  {s['recall']:>{COL_NUM}.4f}"
            f"  {s['f1']:>{COL_NUM}.4f}"
            f"  {s['support']:>{COL_NUM}}"
            f"  {s['kappa']:>{COL_NUM}.4f}"
            f"  {s['alpha']:>{COL_NUM}.4f}"
        )

    print(separator)

    # Wiersz sumaryczny (micro)
    total_support = sum(v["support"] for v in metryki["per_kategoria"].values())
    print(
        f"  {'micro avg':>{COL_KAT}}"
        f"  {metryki['precision_micro']:>{COL_NUM}.4f}"
        f"  {metryki['recall_micro']:>{COL_NUM}.4f}"
        f"  {metryki['f1_micro']:>{COL_NUM}.4f}"
        f"  {total_support:>{COL_NUM}}"
        f"  {metryki['kappa_srednia']:>{COL_NUM}.4f}"
        f"  {metryki['alpha_kripp']:>{COL_NUM}.4f}"
    )
    print(
        f"  {'macro avg':>{COL_KAT}}"
        f"  {metryki['precision_macro']:>{COL_NUM}.4f}"
        f"  {metryki['recall_macro']:>{COL_NUM}.4f}"
        f"  {metryki['f1_macro']:>{COL_NUM}.4f}"
        f"  {total_support:>{COL_NUM}}"
        f"  {'':>{COL_NUM}}"
        f"  {'':>{COL_NUM}}"
    )
    print("═" * W)


# ─────────────────────────────────────────────────────────────────────────────
# ZAPIS WYNIKÓW
# ─────────────────────────────────────────────────────────────────────────────

def zapisz_wyniki(wyniki: dict[str, list[str]], run_id: str, katalog: str = "results") -> str:
    """
    Zapisuje wyniki w formacie złotego standardu:
    id<TAB>KAT1,KAT2,...
    Plik: wynik_ewaluacji_<model>_<run_id>_<data>.txt
    """
    Path(katalog).mkdir(parents=True, exist_ok=True)
    data_str  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nazwa     = f"wynik_ewaluacji_{WYBRANY_PROVIDER}_{run_id}_{data_str}.txt"
    sciezka   = os.path.join(katalog, nazwa)

    with open(sciezka, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for rid, kats in wyniki.items():
            writer.writerow([rid, ",".join(kats)])

    print(f"\n✅ Wyniki zapisane: {sciezka}")
    return sciezka


# ─────────────────────────────────────────────────────────────────────────────
# GŁÓWNA PĘTLA EWALUACYJNA
# ─────────────────────────────────────────────────────────────────────────────

def main(
    sciezka_teksty: str,
    sciezka_standard: str,
    provider: str        = WYBRANY_PROVIDER,
    model_override: str  = None,   # jeśli podane, zastępuje model_name z MODEL_CONFIG
    limit: int           = LIMIT_WIERSZY,
    temperatura: float   = TEMPERATURA,
):
    # ── Konfiguracja środowiska ──────────────────────────────────────────────
    load_dotenv()

    # LangSmith — automatyczne śledzenie przez zmienne środowiskowe
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    langsmith_project = os.environ.get("LANGCHAIN_PROJECT", "EwaluacjaSejmowa")
    print(f"🔍 LangSmith projekt: {langsmith_project}")

    # MLflow — tracking URI i eksperyment
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(langsmith_project)
    print(f"📊 MLflow URI: {mlflow_uri}")

    # ── Wybór modelu ─────────────────────────────────────────────────────────
    if provider not in MODEL_CONFIG:
        raise ValueError(f"Nieznany provider: {provider}. Dozwolone: {list(MODEL_CONFIG)}")
    cfg        = MODEL_CONFIG[provider]
    # model_override (z --model CLI lub LM_STUDIO_MODEL w .env) ma priorytet
    # nad wartością z MODEL_CONFIG — bez potrzeby edycji kodu przy zmianie modelu
    model_name = model_override or os.environ.get("LM_STUDIO_MODEL") if provider == "lm_studio" else None
    model_name = model_name or cfg["model_name"]
    print(f"🤖 Model: {provider} / {model_name} | temperatura: {temperatura} | limit: {limit}")
    if provider == "lm_studio":
        lm_url = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
        print(f"   LM Studio URL: {lm_url}")

    # ── Wczytanie danych ─────────────────────────────────────────────────────
    print(f"\n📂 Wczytywanie tekstów: {sciezka_teksty}")
    teksty = wczytaj_teksty(sciezka_teksty, limit)
    print(f"   Wczytano {len(teksty)} rekordów.")

    print(f"📂 Wczytywanie złotego standardu: {sciezka_standard}")
    standard = wczytaj_zloty_standard(sciezka_standard)
    print(f"   Złoty standard: {len(standard)} rekordów.")

    # ── Główny run MLflow ────────────────────────────────────────────────────
    wyniki: dict[str, list[str]] = {}

    with mlflow.start_run(run_name=f"eval_{provider}_{datetime.datetime.now():%Y%m%d_%H%M}") as run:
        run_id = run.info.run_id
        print(f"\n🏃 MLflow run ID: {run_id}")

        # Logowanie parametrów globalnych
        mlflow.log_params({
            "provider":   provider,
            "model_name": model_name,
            "temperatura": temperatura,
            "limit_wierszy": limit,
        })

        bledy   = 0
        sukces  = 0

        # ── Pętla po tekstach — każdy rekord = osobne zapytanie do LLM ───────
        for i, rekord in enumerate(teksty):
            rid   = rekord["id"]
            tekst = rekord["tekst"]
            prompt = buduj_prompt(tekst)

            print(f"  [{i+1}/{len(teksty)}] ID={rid} ...", end=" ", flush=True)

            # Każdy tekst dostaje własny zagnieżdżony run MLflow
            with mlflow.start_run(run_name=f"tekst_{rid}", nested=True):
                t_start = time.time()
                try:
                    surowa_odpowiedz = wywolaj_llm(prompt, provider, model_name, temperatura)
                    kategorie        = parsuj_odpowiedz(surowa_odpowiedz)
                    latencja         = round(time.time() - t_start, 3)

                    wyniki[rid] = kategorie
                    sukces += 1

                    print(f"→ {','.join(kategorie)} ({latencja}s)")

                    # Logowanie szczegółów per rekord
                    mlflow.log_params({
                        "rekord_id":     rid,
                        "tekst_skrot":   tekst[:80],
                    })
                    mlflow.log_metrics({
                        "latencja_s":    latencja,
                        "n_kategorii":   len(kategorie),
                    })
                    mlflow.log_text(surowa_odpowiedz, f"odpowiedzi/raw_{rid}.txt")
                    mlflow.log_text(",".join(kategorie), f"odpowiedzi/parsed_{rid}.txt")

                except Exception as e:
                    latencja = round(time.time() - t_start, 3)
                    bledy   += 1
                    wyniki[rid] = ["INNE"]
                    print(f"❌ BŁĄD: {e}")
                    mlflow.log_text(str(e), f"bledy/error_{rid}.txt")

        # ── Zapis pliku wynikowego ─────────────────────────────────────────
        plik_wynikowy = zapisz_wyniki(wyniki, run_id)

        # ── Ewaluacja ze złotym standardem ───────────────────────────────
        print("\n" + "═" * 60)
        print("📏 EWALUACJA ZE ZŁOTYM STANDARDEM")
        print("═" * 60)

        metryki = oblicz_metryki(wyniki, standard)

        if metryki:
            # Wyświetlenie pełnej tabeli w konsoli
            drukuj_metryki(metryki)

            # Logowanie metryk zbiorczych do MLflow
            metryki_mlflow = {k: v for k, v in metryki.items() if k != "per_kategoria"}
            mlflow.log_metrics(metryki_mlflow)

            # Logowanie metryk per kategoria do MLflow
            for kat, vals in metryki["per_kategoria"].items():
                kat_slug = kat.lower()
                mlflow.log_metrics({
                    f"prec_{kat_slug}":    vals["precision"],
                    f"rec_{kat_slug}":     vals["recall"],
                    f"f1_{kat_slug}":      vals["f1"],
                    f"kappa_{kat_slug}":   vals["kappa"],
                    f"alpha_{kat_slug}":   vals["alpha"],
                })

        # Logowanie pliku wynikowego jako artefaktu MLflow
        mlflow.log_artifact(plik_wynikowy)

        # Logowanie statystyk końcowych
        mlflow.log_metrics({"n_sukces": sukces, "n_bledy": bledy})

        print("\n" + "═" * 60)
        print(f"✅ Sukces: {sukces} | ❌ Błędy: {bledy}")
        print(f"🔗 MLflow run: {mlflow_uri}/#/experiments/")
        print(f"🔍 LangSmith: {os.environ.get('LANGCHAIN_ENDPOINT','')}")
        print("═" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# PUNKT WEJŚCIA
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ewaluacja klasyfikacji wypowiedzi sejmowych")
    parser.add_argument("--teksty",   default="data/teksty.txt",   help="Plik z tekstami (id<TAB>tekst)")
    parser.add_argument("--standard", default="data/standard.txt", help="Złoty standard (id<TAB>KAT1,KAT2,...)")
    parser.add_argument("--provider", default=WYBRANY_PROVIDER,
                        help="openai | anthropic | gemini | lm_studio")
    parser.add_argument("--model",    default=None,
                        help="Nadpisuje model_name z MODEL_CONFIG (np. dla lm_studio: 'bielik-11b')")
    parser.add_argument("--limit",    default=LIMIT_WIERSZY, type=int,
                        help="Liczba pierwszych wierszy do przetworzenia")
    parser.add_argument("--temp",     default=TEMPERATURA,  type=float,
                        help="Temperatura modelu")
    args = parser.parse_args()

    main(
        sciezka_teksty  = args.teksty,
        sciezka_standard= args.standard,
        provider        = args.provider,
        model_override  = args.model,
        limit           = args.limit,
        temperatura     = args.temp,
    )