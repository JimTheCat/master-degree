"""Aggregate classification results by political group."""

from collections import defaultdict

import pandas as pd

from src.schema.labels import ALL_LABELS, is_emotion, is_rhetorical
from src.schema.models import Prediction, Speech


def aggregate_by_party(
    speeches: list[Speech],
    predictions: list[Prediction],
) -> pd.DataFrame:
    """Aggregate label frequencies by political party.

    Returns a DataFrame with parties as rows and labels as columns,
    values are the fraction of speeches with each label.
    """
    party_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    party_totals: dict[str, int] = defaultdict(int)

    for speech, pred in zip(speeches, predictions):
        party = speech.party
        party_totals[party] += 1
        for label in pred.labels:
            party_counts[party][label] += 1

    rows = []
    for party in sorted(party_totals.keys()):
        row = {"party": party, "total_speeches": party_totals[party]}
        for label in ALL_LABELS:
            count = party_counts[party].get(label, 0)
            row[label] = count / party_totals[party] if party_totals[party] > 0 else 0
            row[f"{label}_count"] = count
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_by_group(
    speeches: list[Speech],
    predictions: list[Prediction],
) -> pd.DataFrame:
    """Aggregate by political_group (coalition) instead of party."""
    group_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    group_totals: dict[str, int] = defaultdict(int)

    for speech, pred in zip(speeches, predictions):
        group = speech.political_group or speech.party
        group_totals[group] += 1
        for label in pred.labels:
            group_counts[group][label] += 1

    rows = []
    for group in sorted(group_totals.keys()):
        row = {"group": group, "total_speeches": group_totals[group]}
        for label in ALL_LABELS:
            count = group_counts[group].get(label, 0)
            row[label] = count / group_totals[group] if group_totals[group] > 0 else 0
        rows.append(row)

    return pd.DataFrame(rows)


def compute_rhetoric_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a normalized rhetorical profile per group.

    Returns emotion and rhetorical scores for each group, normalized to sum to 1.
    """
    emotion_cols = [l for l in ALL_LABELS if is_emotion(l)]
    rhetoric_cols = [l for l in ALL_LABELS if is_rhetorical(l)]

    group_col = "party" if "party" in df.columns else "group"
    profiles = []

    for _, row in df.iterrows():
        e_total = sum(row.get(c, 0) for c in emotion_cols)
        r_total = sum(row.get(c, 0) for c in rhetoric_cols)

        profile = {group_col: row[group_col]}
        for c in emotion_cols:
            profile[f"{c}_norm"] = row.get(c, 0) / e_total if e_total > 0 else 0
        for c in rhetoric_cols:
            profile[f"{c}_norm"] = row.get(c, 0) / r_total if r_total > 0 else 0
        profiles.append(profile)

    return pd.DataFrame(profiles)
