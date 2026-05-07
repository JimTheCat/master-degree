import pandas as pd
import requests
import re
from tqdm import tqdm

# ── Kategorie z definicjami ───────────────────────────────────────────────────
KATEGORIE_DEF = {
    "GOSPODARKA": "Budżet państwa, podatki (VAT, PIT, CIT), inflacja, rynek pracy, przedsiębiorczość, PKB, handel, inwestycje gospodarcze, deficyt budżetowy.",
    "POLITYKA_SPOŁECZNA": "System emerytalny, pomoc socjalna, rodzina, niepełnosprawności, świadczenia (500+, becikowe), ubóstwo, bezdomność, prawa społeczne obywateli.",
    "BEZPIECZEŃSTWO": "Obronność, wojsko, armia, policja, służby specjalne, bezpieczeństwo granic, cyberbezpieczeństwo, NATO, wydatki militarne.",
    "ŚWIATOPOGLĄD": "Prawa reprodukcyjne (aborcja), prawa mniejszości (LGBT+), religia w przestrzeni publicznej, wartości, etyka, związki partnerskie, kwestie światopoglądowe.",
    "PRAWORZĄDNOŚĆ": "Konstytucja, wymiar sprawiedliwości, sądy, Trybunał Konstytucyjny, ustawodawstwo, regulacje prawne, procedury prawne, praworządność jako zasada ustrojowa.",
    "EDUKACJA_NAUKA": "System oświaty (szkoły, przedszkola), szkolnictwo wyższe, badania naukowe, programy nauczania, nauczyciele, stypendia, podstawa programowa.",
    "ŚRODOWISKO": "Ochrona klimatu, transformacja energetyczna, OZE, smog, ochrona przyrody, gospodarka odpadami, zrównoważony rozwój, parki narodowe.",
    "SPRAWY_ZAGRANICZNE": "Polityka międzynarodowa, dyplomacja, Unia Europejska, traktaty, stosunki z innymi państwami (Rosja, Ukraina, USA), uchodźcy, ONZ, sankcje.",
    "ROLNICTWO": "Uprawa roli, hodowla zwierząt, ochrona gleby, gospodarka wodna, pestycydy, GMO, dopłaty dla rolników, skupy, agrobiznes.",
    "POLITYKA_LOKALNA": "Samorządy, problemy regionalne, inwestycje lokalne, prezydenci miast, drogi lokalne, gospodarka przestrzenna, udział społeczny.",
    "ADMINISTRACJA": "Działanie urzędów i ministerstw, przepisy administracyjne, procedury biurokratyczne, finansowanie instytucji publicznych, kompetencje organów państwa.",
    "INFRASTRUKTURA": "Drogi, autostrady, mosty, transport publiczny, PKP, budynki publiczne, szpitale jako obiekty budowlane, inwestycje drogowe.",
    "HISTORIA_PAMIĘĆ_NARODOWA": "Historia Polski, pamięć narodowa, rocznice, bohaterowie narodowi, IPN, muzea, pomniki, II Wojna Światowa, komunizm.",
    "ZDROWIE": "Ochrona zdrowia, system opieki zdrowotnej (NFZ), choroby, leczenie, profilaktyka, lekarze, szpitale jako instytucje medyczne, epidemie, szczepienia.",
    "INNE": "Tylko jeśli tekst dotyczy wyłącznie procedur obrad (głosowania, wnioski formalne, powitania, komentarze bez treści merytorycznej). Używaj tej kategorii rzadko.",
}

KATEGORIE = list(KATEGORIE_DEF.keys())

# ── Budowanie promptu ─────────────────────────────────────────────────────────
def buduj_prompt(tekst: str) -> str:
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
3. Kategorię INNE stosuj wyłącznie, gdy tekst nie ma żadnej treści nie spełnia żadnej innej kategorii (zgodnie z definicją), np. "Dzień dobry państwu. Wznawiam obrady."
4. Zwróć WYŁĄCZNIE nazwy kategorii oddzielone przecinkami. Żadnego komentarza, żadnych wyjaśnień.
5. Przykład poprawnej odpowiedzi: GOSPODARKA,POLITYKA_SPOŁECZNA

WYPOWIEDŹ:
{tekst[:1200]}

ODPOWIEDŹ (tylko kategorie):"""


def parsuj_odpowiedz(tekst_odpowiedzi: str) -> list[str]:
    """
    Wyciąga tylko znane etykiety z odpowiedzi modelu.
    Odporne na dodatkowy tekst, małe litery, spacje.
    """
    tekst_upper = tekst_odpowiedzi.upper()
    znalezione = []
    for kat in KATEGORIE:
        # szukamy etykiety jako osobnego tokenu (nie jako fragment innego słowa)
        pattern = r'(?<![A-ZŁŚŹŻĆŃÓ])' + re.escape(kat) + r'(?![A-ZŁŚŹŻĆŃÓ_])'
        if re.search(pattern, tekst_upper):
            znalezione.append(kat)
    return znalezione if znalezione else ["INNE"]


def zapytaj_model(tekst: str, model: str = "llama3.1:8b") -> list[str]:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": buduj_prompt(tekst),
                "stream": False,
                "options": {
                    "temperature": 0,
                    "top_p": 1,
                    "repeat_penalty": 1.0,
                    "num_predict": 60,
                },
            },
            timeout=120,
        )
        odpowiedz = response.json()["response"].strip()
        return parsuj_odpowiedz(odpowiedz)
    except Exception as e:
        print(f"  [BŁĄD] {e}")
        return ["INNE"]


def main():
    MODEL = "llama3.1:8b"
    PLIK_DANYCH    = "data/teksty.txt"
    PLIK_WYNIKOW   = f"predykcje_{MODEL.replace(':', '_')}.tsv"

    print(f"Model: {MODEL}")
    dane = pd.read_csv(PLIK_DANYCH, sep="\t")
    print(f"Wczytano {len(dane)} tekstów")

    wyniki = []
    for _, row in tqdm(dane.iterrows(), total=len(dane), desc="Klasyfikacja"):
        kategorie = zapytaj_model(str(row["tekst"]), model=MODEL)
        wyniki.append({
            "id": row["id"],
            "predykcja": ",".join(kategorie),
        })

    pd.DataFrame(wyniki).to_csv(PLIK_WYNIKOW, sep="\t", index=False, encoding="utf-8-sig")

    print(f"\nGotowe! Wyniki zapisane w: {PLIK_WYNIKOW}")
    print(f"Uruchom teraz: python ewaluacja_claude.py --predykcje {PLIK_WYNIKOW}")


if __name__ == "__main__":
    main()