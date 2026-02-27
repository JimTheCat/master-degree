from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
import uuid
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from hate_speech.services.runner import run_method
import os

router = APIRouter()
logger = logging.getLogger('uvicorn.error')

class ExperimentRequest(BaseModel):
    method: str
    dataset_path: str
    params: dict = {}


def _replace_nan(obj):
    if isinstance(obj, dict):
        return {k: _replace_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_nan(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


@router.post('/run')
def run_experiment(req: ExperimentRequest):
    try:
        logger.info(f"Starting experiment {req.method} on {req.dataset_path}")
        df = load_dataset(req.dataset_path)
        X = df['processed_text'].tolist()
        y = df['label'].tolist()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        exp_id = str(uuid.uuid4())
        model, preds = run_method(req.method, X_train, y_train, X_test, req.params)
        report = classification_report(y_test, preds, output_dict=True)
        try:
            auc = roc_auc_score(y_test, preds)
            report['auc'] = auc
        except Exception:
            report['auc'] = None
        report = _replace_nan(report)
        logger.info(f"Experiment {exp_id} finished")
        return {'exp_id': exp_id, 'metrics': report}
    except Exception as e:
        logger.exception('Experiment failed')
        raise HTTPException(status_code=500, detail=str(e))


def load_dataset(path: str) -> pd.DataFrame:
    import json as _json
    txt_path = f"{path}/merged_all.txt"
    json_path = f"{path}/merged_all.json"
    labels_path = f"{path}/merged_labels.txt"

    # Wczytaj treści wypowiedzi
    df = pd.read_csv(txt_path, sep='\t', names=['id', 'text'], encoding='utf-8')

    # Wczytaj metadane (JSON lub TSV). Najpierw sprawdź JSON, później TSV.
    meta = None
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as fh:
            meta = _json.load(fh)
        # meta jest listą słowników oczekiwanie (jak wcześniej)
    else:
        # Spróbuj znaleźć plik TSV z metadanymi w katalogu (ignorujemy merged_all.txt i labels)
        candidates = [
            f for f in os.listdir(path)
            if f.lower().endswith('.tsv') and f not in (os.path.basename(txt_path), os.path.basename(labels_path))
        ]
        chosen = None
        if len(candidates) == 1:
            chosen = candidates[0]
        elif len(candidates) > 1:
            # Preferuj pliki zawierające 'meta' w nazwie
            meta_candidates = [c for c in candidates if 'meta' in c.lower()]
            chosen = meta_candidates[0] if meta_candidates else candidates[0]

        if chosen:
            tsv_path = os.path.join(path, chosen)
            df_meta = pd.read_csv(tsv_path, sep='\t', encoding='utf-8', dtype=str)
            # Zamień na listę słowników tak, by walidacja długości była analogiczna do JSON
            meta = df_meta.to_dict(orient='records')
            # Dołącz kolumny metadanych do głównego df (po indeksach)
            df = pd.concat([df.reset_index(drop=True), df_meta.reset_index(drop=True)], axis=1)
        else:
            # Brak pliku z metadanymi
            meta = []

    # Wczytaj etykiety z pliku labels
    with open(labels_path, 'r', encoding='utf-8') as fh:
        labels = [int(line.strip()) for line in fh if line.strip() in ('0', '1')]

    # Walidacja długości
    if not (len(df) == len(meta) == len(labels)):
        raise ValueError(
            f"Liczba rekordów w plikach nie jest równa: "
            f"texts={len(df)}, meta={len(meta)}, labels={len(labels)}"
        )

    df['label'] = labels
    # proste preprocessowanie
    df['processed_text'] = (
        df['text'].fillna('')
        .str.lower()
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )
    return df
