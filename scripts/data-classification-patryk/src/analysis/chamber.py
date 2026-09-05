"""Compare Sejm and Senat, including a within-speaker paired comparison.

The chamber split is available through ``aggregate_by_metadata_key(..., "chamber")``
like any other metadata dimension, but that comparison confounds three things:
the two chambers seat different people, in different party proportions, in
different years. This module isolates the chamber by restricting to the 56
speakers who spoke in *both* (they moved between chambers at the 2019 election),
so party and person are held constant and each speaker is their own control.

Density is measured per 1000 characters rather than per speech: label count
scales with speech length (Spearman 0.61), and the two chambers differ in
length distribution, so a per-speech figure would partly measure how long
people talk.
"""

from collections import defaultdict

import pandas as pd
from scipy import stats

from src.schema.models import Prediction, Speech

DEFAULT_ROLES = frozenset({"Parlamentarzysta"})


def speaker_chamber_density(
    speeches: list[Speech],
    predictions: list[Prediction],
    roles: frozenset[str] | None = DEFAULT_ROLES,
    min_speeches_per_chamber: int = 20,
) -> pd.DataFrame:
    """Label density per speaker per chamber, for speakers present in both.

    Args:
        speeches: Corpus speeches, positionally aligned with ``predictions``.
        predictions: Model predictions, one per speech.
        roles: Keep only these ``metadata.speaker_role`` values; None = all roles.
        min_speeches_per_chamber: Drop speakers below this count in either
            chamber. At 20 the per-speaker density is already stable; lower
            thresholds admit speakers whose Senat figure rests on two speeches.

    Returns:
        One row per speaker with ``speaker``, ``party``, ``n_sejm``, ``n_senat``,
        ``density_sejm``, ``density_senat`` (labels per 1000 characters) and
        ``senat_later`` (True when the speaker's Senat speeches are the later
        ones, i.e. they moved Sejm -> Senat).
    """
    labels: dict[tuple[str, str], int] = defaultdict(int)
    chars: dict[tuple[str, str], int] = defaultdict(int)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    dates: dict[tuple[str, str], list] = defaultdict(list)
    parties: dict[str, str] = {}

    for speech, pred in zip(speeches, predictions):
        meta = speech.metadata or {}
        if roles is not None and meta.get("speaker_role") not in roles:
            continue
        chamber = meta.get("chamber")
        if not speech.speaker or chamber not in ("sejm", "senat"):
            continue
        key = (speech.speaker, chamber)
        labels[key] += len(pred.labels)
        chars[key] += len(speech.text)
        counts[key] += 1
        dates[key].append(str(speech.date))
        parties.setdefault(speech.speaker, speech.party)

    speakers = {name for name, _ in counts}
    rows = []
    for name in sorted(speakers):
        sejm, senat = (name, "sejm"), (name, "senat")
        if counts[sejm] < min_speeches_per_chamber or counts[senat] < min_speeches_per_chamber:
            continue
        rows.append({
            "speaker": name,
            "party": parties.get(name),
            "n_sejm": counts[sejm],
            "n_senat": counts[senat],
            "density_sejm": labels[sejm] / chars[sejm] * 1000,
            "density_senat": labels[senat] / chars[senat] * 1000,
            "senat_later": _median_str(dates[senat]) > _median_str(dates[sejm]),
        })
    return pd.DataFrame(rows)


def chamber_density_summary(
    speeches: list[Speech],
    predictions: list[Prediction],
    roles: frozenset[str] | None = DEFAULT_ROLES,
    drop_top_speakers: int = 0,
) -> pd.DataFrame:
    """Labels per speech and per 1000 characters, per chamber.

    ``drop_top_speakers`` exists because ParlaMint's ``speaker_role`` leaks: a
    Marshal chairing the sitting keeps the ``Parlamentarzysta`` tag, and in the
    Senat the ten busiest speakers account for ~40% of the utterances. Dropping
    them is the robustness check for the chamber difference — if it survives,
    the difference is not an artefact of who happens to run the sitting.
    """
    labels: dict[tuple[str, str], int] = defaultdict(int)
    chars: dict[tuple[str, str], int] = defaultdict(int)
    counts: dict[tuple[str, str], int] = defaultdict(int)

    for speech, pred in zip(speeches, predictions):
        meta = speech.metadata or {}
        if roles is not None and meta.get("speaker_role") not in roles:
            continue
        chamber = meta.get("chamber")
        if chamber not in ("sejm", "senat"):
            continue
        key = (chamber, speech.speaker)
        labels[key] += len(pred.labels)
        chars[key] += len(speech.text)
        counts[key] += 1

    rows = []
    for chamber in ("sejm", "senat"):
        keys = [k for k in counts if k[0] == chamber]
        if drop_top_speakers:
            busiest = sorted(keys, key=lambda k: counts[k], reverse=True)[:drop_top_speakers]
            keys = [k for k in keys if k not in set(busiest)]
        n = sum(counts[k] for k in keys)
        if not n:
            continue
        total_labels = sum(labels[k] for k in keys)
        total_chars = sum(chars[k] for k in keys)
        rows.append({
            "chamber": chamber,
            "n_speeches": n,
            "labels_per_speech": total_labels / n,
            "labels_per_1000_chars": total_labels / total_chars * 1000,
            "dropped_speakers": drop_top_speakers,
        })
    return pd.DataFrame(rows)


def paired_chamber_test(df: pd.DataFrame) -> dict:
    """Wilcoxon signed-rank test on the paired Sejm/Senat densities.

    Returns the test result plus how many speakers moved in each direction, so
    the caller can check the effect is not the 2015-2022 upward trend in
    disguise: speakers who moved Sejm -> Senat and Senat -> Sejm are reported
    separately and should both show the drop if the chamber is what matters.
    """
    if df.empty:
        return {"n": 0}

    result: dict[str, object] = {
        "n": len(df),
        "mean_sejm": float(df["density_sejm"].mean()),
        "mean_senat": float(df["density_senat"].mean()),
        "n_lower_in_senat": int((df["density_senat"] < df["density_sejm"]).sum()),
    }
    stat, p_value = stats.wilcoxon(df["density_sejm"], df["density_senat"])
    result["wilcoxon"] = float(stat)
    result["p_value"] = float(p_value)

    for direction, subset in (("senat_later", df[df["senat_later"]]),
                              ("sejm_later", df[~df["senat_later"]])):
        entry: dict[str, object] = {
            "n": len(subset),
            "n_lower_in_senat": int((subset["density_senat"] < subset["density_sejm"]).sum()),
        }
        if len(subset) >= 6:
            stat, p_value = stats.wilcoxon(subset["density_sejm"], subset["density_senat"])
            entry["wilcoxon"] = float(stat)
            entry["p_value"] = float(p_value)
        result[direction] = entry
    return result


def _median_str(values: list[str]) -> str:
    """Median of ISO date strings; they sort lexicographically, so no parsing."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]
