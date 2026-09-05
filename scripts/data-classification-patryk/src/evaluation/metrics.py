from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)

from src.schema.labels import ALL_LABELS, is_emotion, is_rhetorical
from src.schema.models import GoldAnnotation, Prediction


@dataclass
class LabelMetrics:
    precision: float
    recall: float
    f1: float
    support: int
    kappa: float


@dataclass
class EvaluationReport:
    # Per-label metrics
    per_label: dict[str, LabelMetrics] = field(default_factory=dict)

    # Aggregated metrics
    macro_f1: float = 0.0
    micro_f1: float = 0.0
    weighted_f1: float = 0.0
    hamming_loss: float = 0.0
    subset_accuracy: float = 0.0
    jaccard: float = 0.0

    # Category breakdowns
    emotions_macro_f1: float = 0.0
    rhetorical_macro_f1: float = 0.0

    # Metadata
    total_samples: int = 0
    model_name: str = ""
    prompt_variant: str = ""

    def to_dict(self) -> dict:
        """Flatten report to a dict suitable for MLflow logging."""
        result = {
            "macro_f1": self.macro_f1,
            "micro_f1": self.micro_f1,
            "weighted_f1": self.weighted_f1,
            "hamming_loss": self.hamming_loss,
            "subset_accuracy": self.subset_accuracy,
            "jaccard": self.jaccard,
            "emotions_macro_f1": self.emotions_macro_f1,
            "rhetorical_macro_f1": self.rhetorical_macro_f1,
            "total_samples": self.total_samples,
        }
        for label_name, metrics in self.per_label.items():
            result[f"f1_{label_name}"] = metrics.f1
            result[f"precision_{label_name}"] = metrics.precision
            result[f"recall_{label_name}"] = metrics.recall
            result[f"kappa_{label_name}"] = metrics.kappa
            result[f"support_{label_name}"] = metrics.support
        return result

    def to_table_rows(self) -> list[dict]:
        """Return rows for display as a Rich table."""
        rows = []
        for label_name in ALL_LABELS:
            if label_name in self.per_label:
                m = self.per_label[label_name]
                rows.append({
                    "label": label_name,
                    "precision": f"{m.precision:.3f}",
                    "recall": f"{m.recall:.3f}",
                    "f1": f"{m.f1:.3f}",
                    "kappa": f"{m.kappa:.3f}",
                    "support": str(m.support),
                })
        return rows


def _labels_to_binary_matrix(
    items: list[GoldAnnotation] | list[Prediction],
) -> np.ndarray:
    """Convert a list of annotations/predictions to a binary matrix."""
    label_to_idx = {label: i for i, label in enumerate(ALL_LABELS)}
    matrix = np.zeros((len(items), len(ALL_LABELS)), dtype=int)
    for i, item in enumerate(items):
        for label in item.labels:
            if label in label_to_idx:
                matrix[i, label_to_idx[label]] = 1
    return matrix


def evaluate(
    predictions: list[Prediction],
    gold: list[GoldAnnotation],
    model_name: str = "",
    prompt_variant: str = "",
) -> EvaluationReport:
    """Evaluate predictions against gold standard annotations.

    Both lists must be aligned by index (same speech order).
    """
    assert len(predictions) == len(gold), (
        f"Predictions ({len(predictions)}) and gold ({len(gold)}) must have same length"
    )

    y_true = _labels_to_binary_matrix(gold)
    y_pred = _labels_to_binary_matrix(predictions)

    report = EvaluationReport(
        total_samples=len(predictions),
        model_name=model_name,
        prompt_variant=prompt_variant,
    )

    # Per-label metrics
    for i, label_name in enumerate(ALL_LABELS):
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]
        support = int(true_col.sum())

        if support == 0 and pred_col.sum() == 0:
            # No positive samples in either set
            lm = LabelMetrics(precision=1.0, recall=1.0, f1=1.0, support=0, kappa=1.0)
        else:
            p = float(precision_score(true_col, pred_col, zero_division=0))
            r = float(recall_score(true_col, pred_col, zero_division=0))
            f = float(f1_score(true_col, pred_col, zero_division=0))
            try:
                k = float(cohen_kappa_score(true_col, pred_col))
            except Exception:
                k = 0.0
            lm = LabelMetrics(precision=p, recall=r, f1=f, support=support, kappa=k)

        report.per_label[label_name] = lm

    # Aggregated metrics
    report.macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report.micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    report.weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    report.hamming_loss = float(hamming_loss(y_true, y_pred))
    report.jaccard = float(jaccard_score(y_true, y_pred, average="samples", zero_division=0))

    # Subset accuracy (exact match)
    report.subset_accuracy = float(np.all(y_true == y_pred, axis=1).mean())

    # Category breakdowns
    emotion_f1s = [
        report.per_label[l].f1 for l in ALL_LABELS if is_emotion(l) and l in report.per_label
    ]
    rhetorical_f1s = [
        report.per_label[l].f1 for l in ALL_LABELS if is_rhetorical(l) and l in report.per_label
    ]
    report.emotions_macro_f1 = float(np.mean(emotion_f1s)) if emotion_f1s else 0.0
    report.rhetorical_macro_f1 = float(np.mean(rhetorical_f1s)) if rhetorical_f1s else 0.0

    return report


def optimize_thresholds(
    predictions: list[Prediction],
    gold: list[GoldAnnotation],
    thresholds_range: np.ndarray | None = None,
) -> dict[str, float]:
    """Find optimal per-label threshold maximizing F1 on a validation set.

    Requires predictions to have label_scores (confidence per label).
    """
    if thresholds_range is None:
        thresholds_range = np.arange(0.1, 0.91, 0.05)

    y_true = _labels_to_binary_matrix(gold)
    label_to_idx = {label: i for i, label in enumerate(ALL_LABELS)}

    # Build score matrix
    score_matrix = np.zeros((len(predictions), len(ALL_LABELS)))
    for i, pred in enumerate(predictions):
        for label, score in pred.label_scores.items():
            if label in label_to_idx:
                score_matrix[i, label_to_idx[label]] = score

    optimal_thresholds = {}
    for j, label_name in enumerate(ALL_LABELS):
        best_f1 = -1.0
        best_thresh = 0.5
        for thresh in thresholds_range:
            y_pred_col = (score_matrix[:, j] >= thresh).astype(int)
            f = float(f1_score(y_true[:, j], y_pred_col, zero_division=0))
            if f > best_f1:
                best_f1 = f
                best_thresh = float(thresh)
        optimal_thresholds[label_name] = best_thresh

    return optimal_thresholds
