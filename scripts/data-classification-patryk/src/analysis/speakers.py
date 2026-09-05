"""Aggregate classification results per speaker (politician rankings).

Two filters matter here and both are on by default:

  - ``roles``: the corpus is ~46% ``Przewodniczący`` utterances (chairing the
    sitting, mean 0.12 labels each). Without the role filter every ranking is
    a ranking of Sejm marshals by how often they open a debate.
  - ``min_chars``: labels-per-speech scales with speech length (mean 0.18 for
    <500 chars vs 2.85 for >3000), so procedural one-liners drag rates down
    for whoever happens to make many of them.
"""

from collections import Counter, defaultdict

import pandas as pd
from scipy import stats

from src.schema.labels import ALL_LABELS, is_emotion, is_rhetorical
from src.schema.models import Prediction, Speech

DEFAULT_ROLES = frozenset({"Parlamentarzysta"})

EMOTION_LABELS = [label for label in ALL_LABELS if is_emotion(label)]
RHETORIC_LABELS = [label for label in ALL_LABELS if is_rhetorical(label)]


def aggregate_by_speaker(
    speeches: list[Speech],
    predictions: list[Prediction],
    roles: frozenset[str] | None = DEFAULT_ROLES,
    min_speeches: int = 50,
    min_chars: int = 200,
) -> pd.DataFrame:
    """Aggregate label frequencies per speaker.

    Args:
        speeches: Corpus speeches, positionally aligned with ``predictions``.
        predictions: Model predictions, one per speech.
        roles: Keep only these ``metadata.speaker_role`` values; None = all roles.
        min_speeches: Drop speakers with fewer qualifying speeches.
        min_chars: Drop speeches shorter than this before aggregating.

    Returns:
        One row per speaker with ``speaker``, ``party`` (most frequent party of
        that speaker), ``total_speeches`` (qualifying speeches only),
        ``mean_labels_per_speech``, ``emotion_rate``, ``rhetoric_rate``,
        ``median_chars``, plus ``<LABEL>`` (fraction of speeches carrying it)
        and ``<LABEL>_count`` for every label.
    """
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    parties: dict[str, Counter[str]] = defaultdict(Counter)
    lengths: dict[str, list[int]] = defaultdict(list)

    for speech, pred in zip(speeches, predictions):
        if roles is not None and (speech.metadata or {}).get("speaker_role") not in roles:
            continue
        if len(speech.text) < min_chars:
            continue

        speaker = speech.speaker
        totals[speaker] += 1
        parties[speaker][speech.party] += 1
        lengths[speaker].append(len(speech.text))
        counts[speaker].update(pred.labels)

    rows = []
    for speaker, total in totals.items():
        if total < min_speeches:
            continue
        speaker_counts = counts[speaker]
        row: dict[str, object] = {
            "speaker": speaker,
            "party": parties[speaker].most_common(1)[0][0],
            "total_speeches": total,
            "mean_labels_per_speech": sum(speaker_counts.values()) / total,
            "emotion_rate": sum(speaker_counts[k] for k in EMOTION_LABELS) / total,
            "rhetoric_rate": sum(speaker_counts[k] for k in RHETORIC_LABELS) / total,
            "median_chars": float(pd.Series(lengths[speaker]).median()),
        }
        for label in ALL_LABELS:
            count = speaker_counts[label]
            row[label] = count / total
            row[f"{label}_count"] = count
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("speaker").reset_index(drop=True)


def top_speakers(df: pd.DataFrame, metric: str, n: int = 20) -> pd.DataFrame:
    """Top-N speakers by a metric column, highest first."""
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    return df.nlargest(n, metric).reset_index(drop=True)


def speaker_density(
    speeches: list[Speech],
    predictions: list[Prediction],
    roles: frozenset[str] | None = DEFAULT_ROLES,
    min_speeches: int = 50,
    min_chars: int = 200,
) -> pd.DataFrame:
    """Per-speaker label density, per speech and per 1000 characters.

    ``mean_labels_per_speech`` from ``aggregate_by_speaker`` ranks speakers
    partly by how long they talk: it correlates 0.18 with median speech length,
    and the top of that ranking is held by speakers who deliver policy
    statements of 10 000 characters. Dividing by characters removes that, but
    overcorrects in the other direction (density correlates -0.31 with median
    length, because label count grows sublinearly with length). Neither figure
    is neutral, so both are returned and meant to be read together.

    Returns:
        One row per speaker with ``speaker``, ``party``, ``total_speeches``,
        ``total_chars``, ``median_chars``, ``labels_per_speech``,
        ``labels_per_1000_chars``, and ``density_odd``/``density_even``
        (the same density computed on alternate speeches, for the split-half
        reliability check).
    """
    counts: Counter[str] = Counter()
    chars: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    halves: dict[str, list[list[int]]] = defaultdict(lambda: [[0, 0], [0, 0]])
    parties: dict[str, Counter[str]] = defaultdict(Counter)
    lengths: dict[str, list[int]] = defaultdict(list)

    for speech, pred in zip(speeches, predictions):
        if roles is not None and (speech.metadata or {}).get("speaker_role") not in roles:
            continue
        if len(speech.text) < min_chars or not speech.speaker:
            continue

        speaker = speech.speaker
        half = halves[speaker][totals[speaker] % 2]
        half[0] += len(pred.labels)
        half[1] += len(speech.text)
        totals[speaker] += 1
        counts[speaker] += len(pred.labels)
        chars[speaker] += len(speech.text)
        parties[speaker][speech.party] += 1
        lengths[speaker].append(len(speech.text))

    rows = []
    for speaker, total in totals.items():
        if total < min_speeches:
            continue
        odd, even = halves[speaker]
        rows.append({
            "speaker": speaker,
            "party": parties[speaker].most_common(1)[0][0],
            "total_speeches": total,
            "total_chars": chars[speaker],
            "median_chars": float(pd.Series(lengths[speaker]).median()),
            "labels_per_speech": counts[speaker] / total,
            "labels_per_1000_chars": counts[speaker] / chars[speaker] * 1000,
            "density_odd": odd[0] / odd[1] * 1000 if odd[1] else float("nan"),
            "density_even": even[0] / even[1] * 1000 if even[1] else float("nan"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("labels_per_1000_chars", ascending=False).reset_index(drop=True)


def split_half_reliability(df: pd.DataFrame) -> dict:
    """Spearman correlation between the two half-samples, Spearman-Brown corrected.

    Answers whether the density index measures a stable trait of the speaker or
    the accident of which speeches landed in the corpus. The correction
    estimates reliability at full sample length, since each half uses only half
    the speeches.
    """
    usable = df.dropna(subset=["density_odd", "density_even"])
    if len(usable) < 3:
        return {"n": len(usable)}
    rho = float(stats.spearmanr(usable["density_odd"], usable["density_even"]).statistic)
    return {
        "n": len(usable),
        "spearman": rho,
        "spearman_brown": 2 * rho / (1 + rho) if rho != -1 else float("nan"),
    }


def variance_share_by_party(df: pd.DataFrame, metric: str = "labels_per_1000_chars") -> dict:
    """Share of between-speaker variance in ``metric`` explained by party (eta squared).

    A high value would mean the taxonomy measures a club-level style; a low one
    that it measures individuals who happen to be grouped into clubs.
    """
    if df.empty or metric not in df.columns:
        return {}
    values = df[metric].astype(float)
    grand_mean = values.mean()
    ss_total = float(((values - grand_mean) ** 2).sum())
    ss_between = float(sum(
        len(group) * (group[metric].mean() - grand_mean) ** 2
        for _, group in df.groupby("party")
    ))
    return {
        "eta_squared": ss_between / ss_total if ss_total else 0.0,
        "n_speakers": len(df),
        "n_parties": int(df["party"].nunique()),
    }
