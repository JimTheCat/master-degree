"""Visualization functions for experiment results and political analysis."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.analysis.labels_pl import pl_name, pl_names
from src.schema.labels import ALL_LABELS, is_emotion, is_rhetorical

# 2019-10-13 parliamentary election: boundary between Sejm term 8 and 9.
ELECTION_2019 = "2019Q4"


def plot_label_f1_heatmap(
    per_label_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Per-label F1 scores across experiments",
):
    """Heatmap of F1 scores: labels x experiments."""
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        per_label_df.astype(float),
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Label")
    ax.set_xlabel("Experiment")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_radar_chart(
    per_label_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Model comparison (per-label F1)",
):
    """Radar chart comparing models across labels."""
    labels = list(per_label_df.index)
    experiments = list(per_label_df.columns)
    num_labels = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for exp_name in experiments:
        values = per_label_df[exp_name].astype(float).tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=exp_name)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=8)
    ax.set_ylim(0, 1)
    ax.set_title(title, size=14, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_party_rhetoric_profile(
    df: pd.DataFrame,
    output_path: str | Path,
    group_col: str = "party",
    title: str = "Rhetorical profile by political group",
):
    """Stacked bar chart showing label distribution per party."""
    label_cols = [c for c in ALL_LABELS if c in df.columns]
    if not label_cols:
        return

    emotion_cols = [c for c in label_cols if is_emotion(c)]
    rhetoric_cols = [c for c in label_cols if is_rhetorical(c)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # Emotions
    if emotion_cols:
        df_emotions = df.set_index(group_col)[emotion_cols].rename(columns=pl_name)
        df_emotions.plot(kind="bar", stacked=True, ax=ax1, colormap="Set2")
        ax1.set_title("Rozkład emocji", fontsize=12)
        ax1.set_ylabel("Odsetek wypowiedzi")
        ax1.legend(fontsize=8, loc="upper right")
        ax1.tick_params(axis="x", rotation=45)

    # Rhetorical techniques
    if rhetoric_cols:
        df_rhetoric = df.set_index(group_col)[rhetoric_cols].rename(columns=pl_name)
        df_rhetoric.plot(kind="bar", stacked=True, ax=ax2, colormap="tab10")
        ax2.set_title("Rozkład technik retorycznych", fontsize=12)
        ax2.set_ylabel("Odsetek wypowiedzi")
        ax2.legend(fontsize=8, loc="upper right")
        ax2.tick_params(axis="x", rotation=45)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_cost_effectiveness(
    experiments: list[dict],
    output_path: str | Path,
    title: str = "Cost-effectiveness: F1 vs API cost",
):
    """Scatter plot of Macro-F1 vs total cost for each experiment.

    Args:
        experiments: List of dicts with keys: name, macro_f1, cost_usd.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [e["name"] for e in experiments]
    f1_scores = [e["macro_f1"] for e in experiments]
    costs = [e["cost_usd"] for e in experiments]

    ax.scatter(costs, f1_scores, s=100, zorder=5)
    for i, name in enumerate(names):
        ax.annotate(
            name,
            (costs[i], f1_scores[i]),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=9,
        )

    ax.set_xlabel("Total cost (USD)", fontsize=12)
    ax.set_ylabel("Macro-F1", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_speaker_ranking(
    df: pd.DataFrame,
    output_path: str | Path,
    metric: str,
    top_n: int = 20,
    title: str = "Ranking mówców",
    xlabel: str = "Wskaźnik na wypowiedź",
):
    """Horizontal bar ranking of speakers by a metric, coloured by party.

    Args:
        df: Frame from ``aggregate_by_speaker`` (needs ``speaker``, ``party``,
            ``total_speeches`` and the ``metric`` column).
        metric: Column to rank by, e.g. ``emotion_rate`` or ``AGRESJA_WERBALNA``.
    """
    if df.empty or metric not in df.columns:
        return

    top = df.nlargest(top_n, metric).iloc[::-1]  # reversed: #1 ends up on top
    parties = sorted(top["party"].unique())
    palette = dict(zip(parties, sns.color_palette("tab10", len(parties))))

    fig, ax = plt.subplots(figsize=(11, max(5, 0.42 * len(top))))
    ax.barh(
        range(len(top)),
        top[metric],
        color=[palette[p] for p in top["party"]],
    )
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(
        [f"{s}  (n={int(n)})" for s, n in zip(top["speaker"], top["total_speeches"])],
        fontsize=9,
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=palette[p], label=p) for p in parties],
        title="Partia",
        fontsize=9,
        loc="lower right",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_temporal_lines(
    period_df: pd.DataFrame,
    output_path: str | Path,
    labels: list[str],
    title: str = "Dynamika etykiet w czasie",
    ylabel: str = "Odsetek wypowiedzi",
    mark_period: str | None = ELECTION_2019,
):
    """Line chart of label rates over time, one line per label."""
    if period_df.empty:
        return

    periods = period_df["period"].astype(str).tolist()
    fig, ax = plt.subplots(figsize=(14, 7))

    for label in labels:
        if label in period_df.columns:
            ax.plot(periods, period_df[label], linewidth=2, marker="o",
                    markersize=3, label=pl_name(label))

    if mark_period in periods:
        ax.axvline(periods.index(mark_period), color="gray", linestyle="--",
                   linewidth=1.5)
        ax.text(periods.index(mark_period), ax.get_ylim()[1], " wybory 2019",
                color="gray", fontsize=9, va="top")

    ax.set_xlabel("Okres", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_temporal_heatmap(
    period_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Nasilenie etykiet w czasie",
    normalize: bool = True,
):
    """Heatmap of label rates: labels (rows, Polish names) x periods (columns).

    Args:
        normalize: Z-score each row. Label base rates span 2%–31%, so a shared
            colour scale renders every row except POLARYZACJA_MY_ONI flat.
            Normalizing shows *when* each label peaked, at the cost of no longer
            comparing labels against each other.
    """
    label_cols = [c for c in ALL_LABELS if c in period_df.columns]
    if period_df.empty or not label_cols:
        return

    matrix = period_df.set_index("period")[label_cols].T.astype(float)
    matrix.index = pl_names(matrix.index)

    if normalize:
        std = matrix.std(axis=1).replace(0, np.nan)
        matrix = matrix.sub(matrix.mean(axis=1), axis=0).div(std, axis=0).fillna(0)
        cmap, center, cbar_label = "RdBu_r", 0, "Odchylenie od średniej etykiety (z-score)"
    else:
        cmap, center, cbar_label = "YlOrRd", None, "Odsetek wypowiedzi"

    fig, ax = plt.subplots(figsize=(max(12, 0.45 * matrix.shape[1]), 7))
    sns.heatmap(matrix, cmap=cmap, center=center, ax=ax, annot=False,
                cbar_kws={"label": cbar_label})
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Okres", fontsize=12)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_temporal_by_party(
    df: pd.DataFrame,
    output_path: str | Path,
    label: str,
    title: str | None = None,
    ylabel: str = "Odsetek wypowiedzi",
):
    """Line chart of one label's rate over time, one line per party."""
    if df.empty or label not in df.columns:
        return

    pivot = df.pivot(index="period", columns="party", values=label).sort_index()

    fig, ax = plt.subplots(figsize=(14, 7))
    for party in pivot.columns:
        ax.plot(pivot.index.astype(str), pivot[party], linewidth=2, marker="o",
                markersize=3, label=party)

    periods = pivot.index.astype(str).tolist()
    if ELECTION_2019 in periods:
        ax.axvline(periods.index(ELECTION_2019), color="gray", linestyle="--",
                   linewidth=1.5)

    ax.set_xlabel("Okres", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title or f"{pl_name(label)} — dynamika wg partii", fontsize=14)
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Partia", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_grouped_bars(
    df: pd.DataFrame,
    output_path: str | Path,
    group_col: str,
    title: str = "Porównanie grup",
    ylabel: str = "Odsetek wypowiedzi",
    legend_title: str | None = None,
):
    """Grouped (not stacked) bars: one bar cluster per label, one bar per group.

    Stacked bars hide the pairwise differences this is meant to show, so this
    is deliberately not ``plot_party_rhetoric_profile``.
    """
    label_cols = [c for c in ALL_LABELS if c in df.columns]
    if df.empty or not label_cols:
        return

    plot_df = df.set_index(group_col)[label_cols].T.astype(float)
    plot_df.index = pl_names(plot_df.index)

    fig, ax = plt.subplots(figsize=(15, 7))
    plot_df.plot(kind="bar", ax=ax, colormap="Set2", width=0.8)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(title=legend_title or group_col, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_heatmap(
    per_label_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Label co-occurrence matrix",
):
    """Heatmap showing how often labels co-occur in predictions."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        per_label_df.corr(),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_paired_chamber(
    df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Nasycenie wypowiedzi tych samych mówców w obu izbach",
    ylabel: str = "Etykiety na 1000 znaków",
):
    """Slope chart: one line per speaker, Sejm density on the left, Senat on the right.

    A grouped bar chart would show the two means and hide what matters here,
    which is that almost every individual line goes down. Lines that go up are
    drawn in a contrasting colour so the exceptions stay countable.
    """
    if df.empty or not {"density_sejm", "density_senat"}.issubset(df.columns):
        return

    fig, ax = plt.subplots(figsize=(7, 8))
    for _, row in df.iterrows():
        rising = row["density_senat"] >= row["density_sejm"]
        ax.plot(
            [0, 1],
            [row["density_sejm"], row["density_senat"]],
            color="#c44e52" if rising else "#4c72b0",
            alpha=0.75 if rising else 0.4,
            linewidth=1.6 if rising else 1.0,
            marker="o",
            markersize=3,
        )
    ax.plot(
        [0, 1],
        [df["density_sejm"].mean(), df["density_senat"].mean()],
        color="black",
        linewidth=3,
        marker="o",
        label="średnia",
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Sejm", "Senat"], fontsize=12)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{title}\n(n = {len(df)} mówców)", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_speaker_density_by_party(
    df: pd.DataFrame,
    output_path: str | Path,
    metric: str = "labels_per_1000_chars",
    min_speakers: int = 5,
    title: str = "Rozrzut nasycenia wypowiedzi wewnątrz klubów",
    ylabel: str = "Etykiety na 1000 znaków",
):
    """Box plot with individual speakers overlaid, one box per party.

    The point of the chart is the width of each box relative to the distance
    between boxes, so the individual points are drawn on top: a reader should
    be able to see that party means differ while the ranges overlap almost
    completely.
    """
    if df.empty or metric not in df.columns:
        return

    keep = df.groupby("party")["speaker"].transform("size") >= min_speakers
    plot_df = df[keep]
    if plot_df.empty:
        return
    order = plot_df.groupby("party")[metric].median().sort_values(ascending=False).index

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=plot_df, x="party", y=metric, order=order, ax=ax,
                color="#dce4ef", fliersize=0, width=0.6)
    sns.stripplot(data=plot_df, x="party", y=metric, order=order, ax=ax,
                  color="#2f4b7c", alpha=0.55, size=4, jitter=0.22)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ranking_comparison(
    df: pd.DataFrame,
    output_path: str | Path,
    n_annotate: int = 8,
    title: str = "Dwa sposoby liczenia wskaźnika mówcy",
):
    """Scatter of labels-per-speech against labels-per-1000-characters.

    Speakers far off the diagonal are the ones whose position in a ranking
    depends entirely on which denominator was chosen; those are annotated by
    name, since they are the argument for not quoting a single ranking.
    """
    needed = {"labels_per_speech", "labels_per_1000_chars", "median_chars", "speaker"}
    if df.empty or not needed.issubset(df.columns):
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    points = ax.scatter(
        df["labels_per_speech"],
        df["labels_per_1000_chars"],
        c=df["median_chars"],
        cmap="viridis_r",
        norm=mpl.colors.LogNorm(),
        s=26,
        alpha=0.8,
    )
    fig.colorbar(points, ax=ax, label="Mediana długości wypowiedzi (znaki)")

    ranks = df["labels_per_speech"].rank() - df["labels_per_1000_chars"].rank()
    x_max = df["labels_per_speech"].max()
    for idx in ranks.abs().nlargest(n_annotate).index:
        row = df.loc[idx]
        # Names near the right edge would run under the colour bar; flip them left.
        flip = row["labels_per_speech"] > 0.75 * x_max
        ax.annotate(
            row["speaker"],
            (row["labels_per_speech"], row["labels_per_1000_chars"]),
            textcoords="offset points", xytext=(-6 if flip else 6, 4), fontsize=8,
            horizontalalignment="right" if flip else "left",
        )
    ax.margins(x=0.06)

    ax.set_xlabel("Etykiety na wypowiedź", fontsize=12)
    ax.set_ylabel("Etykiety na 1000 znaków", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
