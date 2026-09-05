"""Aggregate classification results over time and over metadata dimensions.

Speech volume per year is uneven (2015 starts on 12 Nov, 2022 ends on 30 Jun),
so every aggregation here reports label *rates* — the fraction of speeches in a
period carrying a label — never raw counts. The ``<LABEL>_count`` columns are
kept only so ``statistical_tests.run_all_chi_squared`` can consume the frames.
"""

from collections import Counter, defaultdict

import pandas as pd

from src.schema.labels import ALL_LABELS
from src.schema.models import Prediction, Speech


def _rows_from_counts(
    key_name: str,
    totals: Counter[str],
    counts: dict[str, Counter[str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, total in totals.items():
        row: dict[str, object] = {key_name: key, "total_speeches": total}
        for label in ALL_LABELS:
            count = counts[key][label]
            row[label] = count / total if total else 0.0
            row[f"{label}_count"] = count
        rows.append(row)
    return rows


def aggregate_by_period(
    speeches: list[Speech],
    predictions: list[Prediction],
    freq: str = "Q",
) -> pd.DataFrame:
    """Aggregate label rates per time period.

    Args:
        freq: Pandas period frequency — "Y" (year), "Q" (quarter), "M" (month).

    Returns:
        Chronologically sorted frame with a ``period`` string column (e.g.
        "2019Q4"), ``total_speeches``, and per-label rate/count columns.
    """
    totals: Counter[str] = Counter()
    counts: dict[str, Counter[str]] = defaultdict(Counter)

    for speech, pred in zip(speeches, predictions):
        period = str(pd.Period(speech.date, freq=freq))
        totals[period] += 1
        counts[period].update(pred.labels)

    df = pd.DataFrame(_rows_from_counts("period", totals, counts))
    if df.empty:
        return df
    return df.sort_values("period").reset_index(drop=True)


def aggregate_by_period_and_party(
    speeches: list[Speech],
    predictions: list[Prediction],
    freq: str = "Q",
    parties: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate label rates per (period, party) pair.

    Args:
        parties: Restrict to these parties; None keeps all.
    """
    keep = set(parties) if parties else None
    totals: Counter[tuple[str, str]] = Counter()
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)

    for speech, pred in zip(speeches, predictions):
        if keep is not None and speech.party not in keep:
            continue
        key = (str(pd.Period(speech.date, freq=freq)), speech.party)
        totals[key] += 1
        counts[key].update(pred.labels)

    rows: list[dict[str, object]] = []
    for (period, party), total in totals.items():
        row: dict[str, object] = {
            "period": period,
            "party": party,
            "total_speeches": total,
        }
        for label in ALL_LABELS:
            count = counts[(period, party)][label]
            row[label] = count / total if total else 0.0
            row[f"{label}_count"] = count
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["period", "party"]).reset_index(drop=True)


def aggregate_by_metadata_key(
    speeches: list[Speech],
    predictions: list[Prediction],
    key: str,
) -> pd.DataFrame:
    """Aggregate label rates by an arbitrary ``speech.metadata`` field.

    Used for ``party_status`` (Coalition/Opposition), ``term`` (Sejm 8./9.,
    Senat 9./10.) and ``body`` (Sejm vs Senat). Speeches missing the field are
    skipped, so ``total_speeches`` sums to the field's coverage, not the corpus
    size. The group column is named after ``key`` so the frame can be passed
    straight to ``run_all_chi_squared(df, group_col=key)``.

    Rows are ordered by each group's earliest speech, so terms come out
    chronologically rather than alphabetically ("8. kadencja" before "10.").
    """
    totals: Counter[str] = Counter()
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    first_seen: dict[str, object] = {}
    last_seen: dict[str, object] = {}

    for speech, pred in zip(speeches, predictions):
        value = (speech.metadata or {}).get(key)
        if value is None or value == "":
            continue
        group = str(value)
        totals[group] += 1
        counts[group].update(pred.labels)
        if group not in first_seen or speech.date < first_seen[group]:
            first_seen[group] = speech.date
        if group not in last_seen or speech.date > last_seen[group]:
            last_seen[group] = speech.date

    rows = _rows_from_counts(key, totals, counts)
    for row in rows:
        row["first_date"] = str(first_seen[row[key]])
        row["last_date"] = str(last_seen[row[key]])

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["first_date", key]).reset_index(drop=True)
