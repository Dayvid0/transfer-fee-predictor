"""Matplotlib figures for the real-data pages.

The synthetic pages use Plotly because they are interactive scatter and bar
charts. These figures are static and card-shaped, so matplotlib suits them, and
it is the only place the project draws anything that has to look like printed
player art rather than a chart widget.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display on this box; required before pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#1B2430"
MUTED = "#6B7785"
BLUE = "#2E6F9E"
ORANGE = "#D9822B"
GREY = "#B0B7C3"

# Six axes of the classic scouting radar, built from the source's own ratings.
RADAR_GROUPS: dict[str, list[str]] = {
    "Pace": ["Sprint speed", "Acceleration"],
    "Shooting": ["Finishing", "Long shots", "Shot power", "Penalties"],
    "Passing": ["Short passing", "Long passing", "Vision", "Crossing"],
    "Dribbling": ["Dribbling", "Agility", "Balance", "Ball control"],
    "Defending": ["Standing tackle", "Sliding tackle", "Defensive awareness", "Interceptions"],
    "Physical": ["Strength", "Stamina", "Jumping", "Aggression"],
}


def group_scores(row: pd.Series) -> dict[str, float]:
    """Mean rating per radar axis, 0-99, skipping goalkeeper-only attributes."""
    scores = {}
    for axis, columns in RADAR_GROUPS.items():
        present = [c for c in columns if c in row.index and pd.notna(row[c])]
        scores[axis] = float(np.mean([row[c] for c in present])) if present else 0.0
    return scores


def radar(row: pd.Series, colour: str = BLUE, size: float = 4.6):
    """A player's attribute profile as a filled spider chart."""
    scores = group_scores(row)
    labels = list(scores)
    values = [scores[k] for k in labels]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.append(angles, angles[0])
    closed_values = np.append(values, values[0])

    fig, ax = plt.subplots(figsize=(size, size), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(closed_angles, closed_values, color=colour, linewidth=2.2, zorder=3)
    ax.fill(closed_angles, closed_values, color=colour, alpha=0.16, zorder=2)
    ax.scatter(angles, values, color=colour, s=34, zorder=4)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "99"], fontsize=7, color=MUTED)
    ax.grid(color="#DFE4EA", linewidth=0.8)
    ax.spines["polar"].set_color("#DFE4EA")
    fig.tight_layout()
    return fig


def model_bars(metrics: pd.DataFrame, split: str = "clubs"):
    """Horizontal bars of test MAE in euros for one split."""
    rows = metrics.loc[metrics["split"] == split].sort_values("mae_eur")
    fig, ax = plt.subplots(figsize=(7.4, 2.5))
    colours = [GREY if m.startswith("A.") else BLUE for m in rows["model"]]
    bars = ax.barh(rows["model"], rows["mae_eur"] / 1e6, color=colours, height=0.55)
    ax.bar_label(bars, labels=[f"{v / 1e6:.2f}M" for v in rows["mae_eur"]], padding=4,
                 fontsize=9, color=INK)
    ax.set_xlabel("Mean absolute error on held-out players (EUR millions)", color=INK)
    ax.set_title(
        "Test error by model - leave clubs out" if split == "clubs"
        else "Test error by model - random 80/20",
        color=INK, fontsize=11, loc="left",
    )
    ax.set_xlim(0, rows["mae_eur"].max() / 1e6 * 1.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="#EEF1F4")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def value_scatter(players: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray,
                  colour: str = ORANGE):
    """Actual against predicted value on log axes, with the perfect-prediction line."""
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(actual / 1e6, predicted / 1e6, s=11, alpha=0.45, color=colour,
               edgecolors="none")
    lo = float(min(actual.min(), predicted.min())) / 1e6
    hi = float(max(actual.max(), predicted.max())) / 1e6
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Real market value (EUR millions, log)", color=INK)
    ax.set_ylabel("Predicted market value (EUR millions, log)", color=INK)
    ax.set_title(f"Predicted vs real value, {len(actual):,} real players", color=INK,
                 fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#EEF1F4", which="both")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def value_distribution(players: pd.DataFrame, colour: str = BLUE):
    """How the snapshot's valuations are spread, in log counts per band."""
    bands = [0, 1e6, 5e6, 10e6, 20e6, 40e6, 80e6, 160e6]
    labels = ["<1M", "1-5M", "5-10M", "10-20M", "20-40M", "40-80M", "80M+"]
    counts = pd.cut(players["value_eur"], bins=bands, labels=labels, right=False)
    series = counts.value_counts().reindex(labels).fillna(0)

    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    bars = ax.bar(labels, series.to_numpy(), color=colour, width=0.62)
    ax.bar_label(bars, labels=[f"{int(v):,}" for v in series], padding=3, fontsize=8,
                 color=INK)
    ax.set_ylabel("Players", color=INK)
    ax.set_xlabel("Market value band (EUR)", color=INK)
    ax.set_title("Where the values sit", color=INK, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#EEF1F4")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig
