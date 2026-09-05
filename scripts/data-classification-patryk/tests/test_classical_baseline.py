"""Tests for the TF-IDF + LogReg classical baseline."""

import asyncio
from datetime import date

from src.models.classical_baseline import TfidfLogRegBaseline
from src.schema.labels import ALL_LABELS
from src.schema.models import (
    AnnotatedSpeech,
    GoldAnnotation,
    Prediction,
    Speech,
)


def _build_training_set() -> list[AnnotatedSpeech]:
    """Build a tiny synthetic training set covering several labels."""
    samples = [
        ("s1", "Złodzieje, hańba dla narodu! Oszuści.", ["AGRESJA_WERBALNA"]),
        ("s2", "Wielkie zwycięstwo, Polska rośnie w siłę. Sukces.", ["DUMA_I_SUKCES"]),
        ("s3", "Zniszczą nasz kraj, idą po wasze pieniądze. Strach.",
         ["STRATEGIA_STRACHU"]),
        ("s4", "Prawdziwi Polacy kontra zdrajcy. My i oni.",
         ["POLARYZACJA_MY_ONI"]),
        ("s5", "Atakują, nagonka medialna na nas trwa od lat.",
         ["OBLEZIONA_TWIERDZA"]),
        ("s6", "Lecz się człowieku, taki nie ma prawa głosu.",
         ["AD_HOMINEM"]),
        ("s7", "Złodzieje! Oszuści! Hańba! Bezprawie.",
         ["AGRESJA_WERBALNA"]),
        ("s8", "Wielki sukces, Polska świętuje, dumni jesteśmy.",
         ["DUMA_I_SUKCES"]),
    ]
    return [
        AnnotatedSpeech(
            speech=Speech(
                speech_id=sid,
                text=text,
                speaker="x",
                party="x",
                date=date(2020, 1, 1),
            ),
            gold=GoldAnnotation(speech_id=sid, labels=labels),
        )
        for sid, text, labels in samples
    ]


def test_fit_basic():
    model = TfidfLogRegBaseline(max_features=500, ngram_range=(1, 1), min_df=1)
    train = _build_training_set()
    info = model.fit(train)
    assert info["n_train"] == len(train)
    assert info["n_labels"] == len(ALL_LABELS)
    assert info["n_features"] > 0


def test_classify_batch_returns_predictions():
    model = TfidfLogRegBaseline(max_features=500, ngram_range=(1, 1), min_df=1)
    train = _build_training_set()
    model.fit(train)

    test_speeches = [s.speech for s in train[:3]]
    preds = asyncio.run(model.classify_batch(test_speeches))
    assert len(preds) == len(test_speeches)
    for p, s in zip(preds, test_speeches):
        assert isinstance(p, Prediction)
        assert p.speech_id == s.speech_id
        assert p.model_name == "tfidf-logreg"
        assert set(p.label_scores.keys()) == set(ALL_LABELS)
        for label in p.labels:
            assert label in ALL_LABELS


def test_set_thresholds_changes_active_labels():
    model = TfidfLogRegBaseline(max_features=500, ngram_range=(1, 1), min_df=1)
    train = _build_training_set()
    model.fit(train)
    test_speech = train[0].speech

    # Threshold at 0 forces every label active.
    model.set_thresholds({label: 0.0 for label in ALL_LABELS})
    pred = asyncio.run(model.classify(test_speech))
    assert set(pred.labels) == set(ALL_LABELS)

    # Threshold at 1 forces every label inactive.
    model.set_thresholds({label: 1.0 for label in ALL_LABELS})
    pred = asyncio.run(model.classify(test_speech))
    assert pred.labels == []
