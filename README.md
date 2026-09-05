# Master's Degree — Polish Parliamentary Speech Analysis

Research repository for two master's theses built on the same source material:
speeches from the Polish Sejm and Senate (ParlaMint-PL / Sejm website). The
shared pipeline goes from crawling transcripts, through dual-annotator manual
labelling and inter-annotator agreement analysis, to automatic classification
with classical models, fine-tuned transformers and LLMs.

Two annotation tracks run on top of that pipeline:

- **Emotions and rhetorical techniques** (`data-classification-patryk`) —
  13 labels, multi-label.
- **Thematic categories** (`data-classification-kinga`,
  `gold-standard-experiments-kinga`) — 15 topical categories, multi-label.

## Repository Structure

```
master-degree/
├── dataset/                            # Gold standard annotations (500 samples per annotator)
│   ├── zloty-standard-badanie-kinga.txt
│   └── zloty-standard-badanie-patryk.txt
│
└── scripts/
    ├── data-crawler/                   # Sejm transcript scraper
    ├── ground-truth-kinga/             # Annotation app (annotator 1)
    ├── ground-truth-patryk/            # Annotation app (annotator 2)
    ├── inter-annotator-agreement/      # Agreement statistics
    ├── NLP-Benchmark-API/              # Hate-speech detection benchmark API
    ├── data-classification-patryk/     # Emotions + rhetorical techniques pipeline
    ├── data-classification-kinga/      # Thematic classification (production annotation run)
    ├── gold-standard-experiments-kinga/# Thematic classification (LLM evaluation harness)
    └── evaluation-presentation-kinga/  # Streamlit browser for the annotated corpus
```

## Components

### Data Crawler (`scripts/data-crawler/`)

Web scraper for Polish Sejm parliamentary speeches and deputy information.

**Features:**
- Downloads speech transcripts from the Sejm API
- Parses HTML into structured text
- Extracts speaker metadata
- Outputs TSV/JSON format

**Usage:**
```bash
cd scripts/data-crawler
pip install -r requirements.txt
python main.py
```

### Annotation Tools (`scripts/ground-truth-*/`)

Streamlit web applications for manual text annotation by two independent annotators.

**Features:**
- Text-by-text annotation interface
- Multi-category selection
- Progress tracking
- Google Drive synchronization
- Offline-first design

**Usage:**
```bash
cd scripts/ground-truth-kinga  # or ground-truth-patryk
pip install -r requirements.txt
streamlit run app.py
```

### Inter-Annotator Agreement (`scripts/inter-annotator-agreement/`)

Statistical analysis of annotation consistency between annotators.

**Metrics:**
- **Cohen's Kappa** — Agreement accounting for chance
- **Krippendorff's Alpha** — Multi-coder reliability
- **Percent Agreement** — Simple agreement percentage

**Scripts:**
- `main_emocje_techniki.py` — For emotions and rhetorical techniques
- `main_tematyczne.py` — For thematic categories

**Interpretation Scale:**
| Kappa Value | Interpretation |
|-------------|----------------|
| < 0.20      | Weak           |
| 0.20 - 0.40 | Fair           |
| 0.40 - 0.60 | Moderate       |
| 0.60 - 0.80 | Good           |
| ≥ 0.80      | Excellent      |

### Emotions & Rhetoric Classification (`scripts/data-classification-patryk/`)

Config-driven pipeline for multi-label detection of **5 emotions**
(verbal aggression, fear strategy, dehumanisation/contempt, pride & success,
moral messianism) and **8 rhetorical techniques** (us-vs-them polarisation,
ad hominem, besieged fortress, attributing bad intentions, whataboutism,
extension sophism, anecdotal evidence, appeal to unity).

**Compared approaches:**
| Approach    | Model                                            |
|-------------|--------------------------------------------------|
| Classical   | TF-IDF + one-vs-rest Logistic Regression         |
| Transformer | Fine-tuned HerBERT (`allegro/herbert-base-cased`)|
| LLM (API)   | GPT, Claude, Gemini                              |
| LLM (local) | Bielik, Gemma, Qwen via LM Studio                |

Each model is run across four prompt variants (`zeroshot`, `fewshot`,
`detailed`, `fewshot_detailed`). Experiments are YAML-configured, tracked in
MLflow + LangSmith, and the winning config is then applied to the full corpus
with checkpointing and a cost budget.

**Usage:**
```bash
cd scripts/data-classification-patryk
pip install -e ".[dev]"
cp .env.example .env                  # API keys

python scripts/smoke_test.py --config configs/model_claude.yaml
python scripts/run_llm_experiment.py --config configs/model_claude.yaml
python scripts/run_llm_sweep.py --split all --resume
python scripts/generate_corpus_analysis.py
```

See [`scripts/data-classification-patryk/CLAUDE.md`](scripts/data-classification-patryk/CLAUDE.md)
for the full command reference and architecture notes.

### Thematic Classification (`scripts/data-classification-kinga/`, `scripts/gold-standard-experiments-kinga/`)

Multi-label assignment of **15 thematic categories** (economy, social policy,
security, worldview, rule of law, education & science, environment, foreign
affairs, agriculture, and others) to parliamentary speeches.

- `gold-standard-experiments-kinga/` — LLM evaluation harness over the gold
  standard. Supports OpenAI / Anthropic / Gemini / LM Studio, tracks runs in
  MLflow + LangSmith and estimates per-text USD cost.
- `data-classification-kinga/` — the production annotation run over the whole
  corpus. Sequential processing, explicit `OK` / `PARSE_FAIL` / `ERROR`
  statuses (no silent fallback), resume-by-ID, per-record flush + fsync, and
  Google Drive backups. A prompt checksum guards that the category definitions
  match the ones the evaluation was run on.

**Usage:**
```bash
cd scripts/data-classification-kinga
pip install -r requirements.txt
python klasyfikator.py                      # production run (auto-resumes)
python klasyfikator.py --benchmark --limit 200
```

### Corpus Browser (`scripts/evaluation-presentation-kinga/`)

Streamlit application for exploring the thematically annotated corpus —
228,326 speeches from the Sejm and Senate (November 2015 – June 2022) broken
down by party, chamber, term, speaker, gender and time.

**Usage:**
```bash
cd scripts/evaluation-presentation-kinga
pip install -r requirements.txt
streamlit run app.py
```

Online: https://s24839-masters-degree-presentation-app.streamlit.app

### NLP Benchmark API (`scripts/NLP-Benchmark-API/`)

FastAPI server for evaluating multiple hate speech detection methods.

**Detection Methods (9 variants):**

| Category    | Method                | Description                                  |
|-------------|-----------------------|----------------------------------------------|
| Formal      | `formal_regex`        | Pattern matching for hate speech keywords    |
| Formal      | `formal_negation`     | Token-based detection with negation handling |
| Statistical | `stat_nb`             | Naive Bayes with TF-IDF                      |
| Statistical | `stat_svm`            | Support Vector Machine with TF-IDF           |
| Statistical | `stat_logreg`         | Logistic Regression with TF-IDF              |
| Statistical | `stat_randomforest`   | Random Forest with TF-IDF                    |
| Neural      | `neural_bert`         | HerBERT (Polish BERT) fine-tuning            |
| Neural      | `neural_lstm`         | LSTM network                                 |
| Hybrid      | `hybrid_voting`       | Ensemble of formal + statistical methods     |

**Usage:**
```bash
cd scripts/NLP-Benchmark-API
pip install -r requirements.txt
uvicorn app:app --reload
```

**API Endpoint:**
```
POST /experiments/run
{
    "method": "stat_svm",
    "dataset_path": "path/to/data",
    "params": {}
}
```

**Returns:** Classification metrics (Precision, Recall, F1, AUC, Kappa)

## Data Flow

```
Polish Sejm Website / ParlaMint-PL
              │
              ▼
        [data-crawler]
              │
              ▼
        Raw Transcripts
              │
          ┌───┴───┐
          ▼       ▼
      [kinga]  [patryk]          manual annotation (500 samples each)
          │       │
          └───┬───┘
              ▼
  [inter-annotator-agreement]
              │
              ▼
     Gold Standard Dataset
              │
      ┌───────┼────────────────────┐
      ▼       ▼                    ▼
[NLP-Benchmark]  [data-classification-patryk]   [gold-standard-experiments-kinga]
   hate speech    emotions + rhetoric                thematic categories
      │                  │                                  │
      │                  ▼                                  ▼
      │        full-corpus predictions          [data-classification-kinga]
      │                  │                                  │
      ▼                  ▼                                  ▼
 Evaluation      results/analysis/REPORT.md      [evaluation-presentation-kinga]
  Results         (party / speaker / time)          interactive browser
```

## Technology Stack

- **Web Framework:** FastAPI, Uvicorn
- **Annotation & Presentation UI:** Streamlit
- **Machine Learning:** scikit-learn, PyTorch, Transformers
- **NLP Models:** HerBERT (Polish BERT by Allegro), Bielik, Gemma, Qwen
- **LLM Access:** LangChain (OpenAI, Anthropic, Google GenAI, LM Studio)
- **Experiment Tracking:** MLflow, LangSmith
- **Web Scraping:** requests, BeautifulSoup4
- **Data Processing:** pandas, NumPy, PyArrow
- **Visualization:** matplotlib, seaborn, plotly
- **Cloud Storage:** Google Drive API

## Dataset

The `dataset/` directory contains ground truth annotations:
- 500 annotated text samples per annotator
- Dual-annotator setup for reliability measurement
- Polish parliamentary speech excerpts

The full annotated corpus (~228k speeches, 2015–2022) is produced by the
classification pipelines and lives inside the individual component
directories, not in `dataset/`.

## Language

This project focuses on the **Polish language**:
- Polish-specific models (HerBERT, Bielik)
- Annotation categories and prompts in Polish
- Data sourced from the Polish Sejm and Senate

## Requirements

- Python 3.11+ (3.8+ for the older components; see each component's own docs)
- Per-component dependencies: `requirements.txt`, or `pyproject.toml` for
  `data-classification-patryk`
- API keys for the LLM components go in a local `.env` (never committed)

## License

This project is part of a master's degree research program.
