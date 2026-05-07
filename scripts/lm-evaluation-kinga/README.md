# 🏛️ Ewaluator Wypowiedzi Sejmowych

Program do automatycznej klasyfikacji tematycznej wypowiedzi sejmowych z wykorzystaniem LLM (OpenAI, Anthropic, Gemini, lokalnych modeli z LMStudio), śledzeniem w **MLflow** i **LangSmith** oraz pełną ewaluacją jakości klasyfikacji.

---

## 📁 Struktura projektu

```
lm-evaluation-kinga/
├── evaluator.py        # Główny skrypt ewaluacyjny
├── requirements.txt    # Zależności Python
├── .env                # Klucze API (uzupełnij!)
├── .gitignore
├── data/
│   ├── teksty.tsv      # Plik wejściowy z tekstami
│   └── standard.tsv    # Złoty standard
└── results/            # Wyniki ewaluacji (tworzy się automatycznie)
```

---

## ⚙️ Wymagania wstępne

- Python 3.11+
- Dostęp do przynajmniej jednego API: OpenAI, Anthropic lub Google (lub lokalny model z LMStudio)
- Opcjonalnie: konto LangSmith (https://smith.langchain.com)

---

## 🚀 Krok po kroku: Uruchomienie

### 1. Sklonuj / skopiuj projekt i przejdź do katalogu

```bash
cd lm-evaluation-kinga
```

### 2. Utwórz i aktywuj środowisko wirtualne

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Zainstaluj zależności

```bash
pip install -r requirements.txt
```

### 4. Skonfiguruj plik `.env`

Otwórz plik `.env` i uzupełnij wartości swoimi kluczami API:

```env
OPENAI_API_KEY=sk-proj-...          # klucz OpenAI
ANTHROPIC_API_KEY=sk-ant-...        # klucz Anthropic
GOOGLE_API_KEY=AIza-...             # klucz Google AI Studio
LANGCHAIN_API_KEY=lsv2_pt_...       # klucz LangSmith
LANGCHAIN_PROJECT=EwaluacjaSejmowa  # nazwa projektu w LangSmith
MLFLOW_TRACKING_URI=http://127.0.0.1:5001
LM_STUDIO_URL=http://localhost:1234/v1
```

> 💡 Klucze możesz pobrać z:
> - OpenAI: https://platform.openai.com/api-keys
> - Anthropic: https://console.anthropic.com/settings/keys
> - Google: https://aistudio.google.com/apikey
> - LangSmith: https://smith.langchain.com → Settings → API Keys

---

### 5. Przygotuj pliki danych

#### Plik z tekstami (`data/teksty.txt`)
Format: `id<TAB>tekst` — jedna wypowiedź per linia:

```
001	Panie Marszałku! Wysoka Izbo! Chciałbym zabrać głos w sprawie budżetu na rok 2025...
002	Dziękuję. Przechodzimy do kolejnego punktu porządku obrad.
003	W kwestii ochrony zdrowia pragnę wskazać, że nakłady na NFZ...
```

#### Złoty standard (`data/standard.txt`)
Format: `id<TAB>KAT1,KAT2,...` — każde id może mieć wiele kategorii:

```
001	GOSPODARKA,POLITYKA_SPOŁECZNA
002	INNE
003	ZDROWIE
```

> ⚠️ Separator kolumn to **tabulator** (`\t`), nie spacja.

---

### 6. Uruchom serwer MLflow

W **osobnym** oknie terminala uruchom serwer MLflow:

```bash
python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Po uruchomieniu interfejs MLflow będzie dostępny pod adresem:
**http://127.0.0.1:5001**

---

### 7. Uruchom program

#### Tryb testowy (domyślnie 10 rekordów — tanie testowanie)

Przed uruchomieniem na 500 rekordach zmień `LIMIT_WIERSZY` w `evaluator.py`:

```python
LIMIT_WIERSZY = 10   # ← szybki test
LIMIT_WIERSZY = 100  # ← pośredni test
LIMIT_WIERSZY = 500  # ← pełna ewaluacja
```

Lub podaj limit jako argument:

```bash
# 10 rekordów, OpenAI
python evaluator.py --limit 10 --provider openai

# 500 rekordów, Anthropic Claude
python evaluator.py --limit 500 --provider anthropic

# 100 rekordów, Gemini
python evaluator.py --limit 100 --provider gemini

# Niestandardowe pliki wejściowe
python evaluator.py --teksty moje_teksty.tsv --standard moj_standard.tsv --provider openai
```

#### Wszystkie parametry CLI

| Parametr     | Domyślnie           | Opis                                                       |
|--------------|---------------------|------------------------------------------------------------|
| `--teksty`   | `data/teksty.txt`   | Ścieżka do pliku z tekstami                                |
| `--standard` | `data/standard.txt` | Ścieżka do złotego standardu                               |
| `--provider` | `openai`            | Provider LLM: `openai`, `anthropic`, `gemini`, `lm_studio` |
| `--limit`    | `10`                | Liczba rekordów do przetworzenia                           |
| `--temp`     | `0.0`               | Temperatura modelu (0.0 = deterministyczna)                |

---

## 🤖 Dostępne modele

| Provider    | Model               | Uwagi                                     |
|-------------|---------------------|-------------------------------------------|
| `openai`    | `gpt-4o`            | Najlepszy model OpenAI do języka polskiego |
| `anthropic` | `claude-sonnet-4-5` | Dobry balans jakości i kosztów            |
| `gemini`    | `gemini-2.0-flash`  | Szybki i ekonomiczny                      |
| `lm_studio` |                     | Lokalny model z LMStudio                  |
Modele możesz zmienić w sekcji `MODEL_CONFIG` w pliku `evaluator.py`.

---

## 📊 Wyniki i artefakty

### Plik wynikowy
Zapisywany automatycznie w katalogu `results/`:
```
results/wynik_ewaluacji_<run_id>_<data>.tsv
```
Format identyczny ze złotym standardem: `id<TAB>KAT1,KAT2,...`

### Metryki wyświetlane w konsoli i logowane do MLflow

| Metryka               | Opis                                                  |
|-----------------------|-------------------------------------------------------|
| **Zgodność (exact)**  | % rekordów z identycznym zbiorem kategorii            |
| **Precision (micro)** | Precyzja uśredniona po wszystkich kategoriach         |
| **Recall (micro)**    | Czułość uśredniona po wszystkich kategoriach          |
| **F1 (micro)**        | Miara F1 uśredniona po wszystkich kategoriach         |
| **Kappa (śr.)**       | Cohen's Kappa uśrednione po kategoriach               |
| **Alpha Kripp.**      | Krippendorff's Alpha (niezawodność annotatorów)       |
| **F1 per kategoria**  | F1 osobno dla każdej z 15 kategorii                   |

---

## 🔍 Monitoring w LangSmith

Po uruchomieniu programu każde zapytanie do LLM jest automatycznie śledzone w LangSmith. Wejdź na:

**https://smith.langchain.com** → wybierz projekt `EwaluacjaSejmowa`

Zobaczysz:
- Każde osobne zapytanie z treścią promptu i odpowiedzią
- Czasy odpowiedzi
- Zużycie tokenów (jeśli obsługiwane przez provider)

---

## 🗂️ Kategorie tematyczne

Program klasyfikuje wypowiedzi do 15 kategorii:

`ADMINISTRACJA`, `BEZPIECZEŃSTWO`, `EDUKACJA_NAUKA`, `GOSPODARKA`,
`HISTORIA_PAMIĘĆ_NARODOWA`, `INFRASTRUKTURA`, `INNE`, `POLITYKA_LOKALNA`,
`POLITYKA_SPOŁECZNA`, `PRAWORZĄDNOŚĆ`, `ROLNICTWO`, `SPRAWY_ZAGRANICZNE`,
`ZDROWIE`, `ŚRODOWISKO`, `ŚWIATOPOGLĄD`

---

## 🐛 Rozwiązywanie problemów

**`ModuleNotFoundError`** → Sprawdź czy środowisko wirtualne jest aktywowane i uruchom `pip install -r requirements.txt`

**`AuthenticationError`** → Sprawdź czy klucze API w `.env` są poprawne

**MLflow nie odpowiada** → Uruchom `mlflow ui --port 5001` w osobnym terminalu

**LangSmith nie rejestruje** → Sprawdź `LANGCHAIN_TRACING_V2=true` i `LANGCHAIN_API_KEY` w `.env`

**Błąd parsowania pliku wejściowego** → Upewnij się, że separator to tabulator (`\t`), a plik jest w kodowaniu UTF-8
