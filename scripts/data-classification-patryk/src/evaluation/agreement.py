"""Inter-annotator and model-human agreement metrics."""

import numpy as np
from sklearn.metrics import cohen_kappa_score

from src.schema.labels import ALL_LABELS
from src.schema.models import GoldAnnotation, Prediction


def compute_agreement(
    annotations_a: list[GoldAnnotation],
    annotations_b: list[GoldAnnotation] | list[Prediction],
) -> dict[str, float]:
    """Compute per-label and overall Cohen's kappa between two sets of annotations.

    Returns a dict with per-label kappa and an overall macro-averaged kappa.
    """
    assert len(annotations_a) == len(annotations_b)

    label_to_idx = {label: i for i, label in enumerate(ALL_LABELS)}
    n = len(annotations_a)

    matrix_a = np.zeros((n, len(ALL_LABELS)), dtype=int)
    matrix_b = np.zeros((n, len(ALL_LABELS)), dtype=int)

    for i, ann in enumerate(annotations_a):
        for label in ann.labels:
            if label in label_to_idx:
                matrix_a[i, label_to_idx[label]] = 1

    for i, ann in enumerate(annotations_b):
        for label in ann.labels:
            if label in label_to_idx:
                matrix_b[i, label_to_idx[label]] = 1

    kappas = {}
    for j, label_name in enumerate(ALL_LABELS):
        try:
            k = float(cohen_kappa_score(matrix_a[:, j], matrix_b[:, j]))
        except Exception:
            k = 0.0
        kappas[label_name] = k

    kappas["macro_kappa"] = float(np.mean(list(kappas.values())))
    return kappas
