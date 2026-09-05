"""Statistical tests for comparing groups and models."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.schema.labels import ALL_LABELS


def chi_squared_test(
    df: pd.DataFrame,
    label: str,
    group_col: str = "party",
) -> dict[str, Any]:
    """Chi-squared test for independence between group and label presence.

    Tests whether the proportion of speeches with a given label differs
    significantly across political groups.
    """
    if f"{label}_count" not in df.columns:
        return {"label": label, "group_col": group_col, "error": "count column not found"}
    if group_col not in df.columns:
        return {
            "label": label,
            "group_col": group_col,
            "error": f"group column '{group_col}' not found",
        }

    observed = np.array([
        [row[f"{label}_count"], row["total_speeches"] - row[f"{label}_count"]]
        for _, row in df.iterrows()
    ])

    # Remove groups with 0 total speeches
    mask = observed.sum(axis=1) > 0
    observed = observed[mask]

    if observed.shape[0] < 2:
        return {"label": label, "group_col": group_col, "error": "need at least 2 groups"}

    chi2, p_value, dof, expected = stats.chi2_contingency(observed)

    # Cramér's V as effect size
    n = observed.sum()
    k = min(observed.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if n * k > 0 else 0

    return {
        "label": label,
        "group_col": group_col,
        "n_groups": int(observed.shape[0]),
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "cramers_v": float(cramers_v),
        "significant": p_value < 0.05,
    }


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict[str, float]:
    """McNemar's test comparing two models on the same test set.

    Args:
        y_true: Binary array of ground truth (per-label).
        y_pred_a: Binary array of model A predictions.
        y_pred_b: Binary array of model B predictions.
    """
    # Correct by A but not B
    a_correct = (y_pred_a == y_true).astype(int)
    b_correct = (y_pred_b == y_true).astype(int)

    # Contingency: (A correct & B wrong), (A wrong & B correct)
    b01 = np.sum((a_correct == 1) & (b_correct == 0))
    c01 = np.sum((a_correct == 0) & (b_correct == 1))

    if b01 + c01 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    # McNemar with continuity correction
    statistic = (abs(b01 - c01) - 1) ** 2 / (b01 + c01)
    p_value = float(stats.chi2.sf(statistic, df=1))

    return {
        "statistic": float(statistic),
        "p_value": p_value,
        "b01": int(b01),
        "c01": int(c01),
        "significant": p_value < 0.05,
    }


def run_all_chi_squared(
    df: pd.DataFrame,
    group_col: str = "party",
) -> pd.DataFrame:
    """Run chi-squared tests for all labels."""
    results = []
    for label in ALL_LABELS:
        result = chi_squared_test(df, label, group_col)
        results.append(result)
    return pd.DataFrame(results)
