## Project Overview

This repository implements an end-to-end pipeline for:
1. **Data Collection** - Crawling parliamentary transcripts from the Polish Sejm website
2. **Manual Annotation** - Web-based tools for dual-annotator labeling
3. **Agreement Analysis** - Statistical measures of annotation consistency
4. **Benchmark Evaluation** - Comparing 9 different hate speech detection methods

## Repository Structure

```
master-degree/
├── dataset/                          # Annotated ground truth data
│   ├── zloty-standard-badanie-kinga.txt
│   └── zloty-standard-badanie-patryk.txt
│
├── scripts/
│   ├── data-crawler/                 # Sejm transcript scraper
│   ├── ground-truth-kinga/           # Annotation app (annotator 1)
│   ├── ground-truth-patryk/          # Annotation app (annotator 2)
│   ├── inter-annotator-agreement/    # Agreement statistics
│   └── NLP-Benchmark-API/            # Detection methods API
```

## Components

### Data Crawler (`scripts/data-crawler/`)

Web scraper for Polish Sejm parliamentary speeches and deputy information.

**Features:**
- Downloads speech transcripts from Sejm API
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
- **Cohen's Kappa** - Agreement accounting for chance
- **Krippendorff's Alpha** - Multi-coder reliability
- **Percent Agreement** - Simple agreement percentage

**Scripts:**
- `main_emocje_techniki.py` - For emotions and rhetorical techniques
- `main_tematyczne.py` - For thematic categories

**Interpretation Scale:**
| Kappa Value | Interpretation |
|-------------|----------------|
| < 0.20      | Weak           |
| 0.20 - 0.40 | Fair           |
| 0.40 - 0.60 | Moderate       |
| 0.60 - 0.80 | Good           |
| ≥ 0.80      | Excellent      |

### NLP Benchmark API (`scripts/NLP-Benchmark-API/`)

FastAPI server for evaluating multiple hate speech detection methods.

**Detection Methods (9 variants):**

| Category    | Method                | Description                              |
|-------------|-----------------------|------------------------------------------|
| Formal      | `formal_regex`        | Pattern matching for hate speech keywords |
| Formal      | `formal_negation`     | Token-based detection with negation handling |
| Statistical | `stat_nb`             | Naive Bayes with TF-IDF                  |
| Statistical | `stat_svm`            | Support Vector Machine with TF-IDF       |
| Statistical | `stat_logreg`         | Logistic Regression with TF-IDF          |
| Statistical | `stat_randomforest`   | Random Forest with TF-IDF                |
| Neural      | `neural_bert`         | HerBERT (Polish BERT) fine-tuning        |
| Neural      | `neural_lstm`         | LSTM network                             |
| Hybrid      | `hybrid_voting`       | Ensemble of formal + statistical methods |

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
Polish Sejm Website
        │
        ▼
   [data-crawler]
        │
        ▼
   Raw Transcripts
        │
    ┌───┴───┐
    ▼       ▼
[kinga]  [patryk]
    │       │
    └───┬───┘
        ▼
[inter-annotator-agreement]
        │
        ▼
   Gold Standard Dataset
        │
        ▼
  [NLP-Benchmark-API]
        │
        ▼
   Evaluation Results
```

## Technology Stack

- **Web Framework:** FastAPI, Uvicorn
- **Annotation UI:** Streamlit
- **Machine Learning:** scikit-learn, PyTorch, Transformers
- **NLP Models:** HerBERT (Polish BERT by Allegro)
- **Web Scraping:** requests, BeautifulSoup4
- **Data Processing:** pandas, NumPy
- **Cloud Storage:** Google Drive API

## Dataset

The `dataset/` directory contains ground truth annotations:
- 500 annotated text samples per annotator
- Dual-annotator setup for reliability measurement
- Polish parliamentary speech excerpts

## Language

This project focuses on **Polish language** hate speech detection:
- Uses HerBERT (Polish BERT variant)
- Annotation categories in Polish
- Data sourced from Polish Sejm (parliament)

## Requirements

- Python 3.8+
- See individual `requirements.txt` files in each component directory

## License

This project is part of a master's degree research program.
