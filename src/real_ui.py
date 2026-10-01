"""Cached access and formatting for the real-data dashboard.

Mirrors ``ui.py`` for the synthetic pipeline. The real data is a single 2024
snapshot, so there is no price index to deflate and no chronological split; the
defences that matter here are the two split definitions in ``real_model``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from src import config as C
from src import football_data as FD
from src import real_model as RM
from src.real_data import fold, load

COMPETITION_LABELS = dict(FD.COMPETITIONS)

# The two splits the real pages offer, in the order they are reported.
SPLITS = {"random": RM.random_split, "clubs": RM.clubs_split}


@dataclass
class RealData:
    """Everything the real-data pages read, built once per process."""

    players: pd.DataFrame
    metrics: pd.DataFrame
    fitted_random: RM.Fitted
    fitted_clubs: RM.Fitted
    labels: dict = field(default_factory=dict)

    def find(self, query: str) -> pd.DataFrame:
        """Accent-insensitive name search, best matches first."""
        key = fold(query).strip()
        if not key:
            return self.players
        exact = self.players.loc[self.players["search_key"] == key]
        rest = self.players.loc[
            self.players["search_key"].str.contains(key, regex=False)
            & ~self.players["search_key"].eq(key)
        ]
        # Richer players first: a name typed into the box is usually a star.
        return pd.concat([exact, rest]).sort_values(
            "value_eur", ascending=False, kind="stable"
        )


@st.cache_resource(show_spinner=False)
def load_real() -> RealData:
    frame = RM.usable(load())
    # Fit each split once and reuse the fitted object for both the metrics table
    # and the single-player predictions; refitting per page would quadruple the
    # wait on first load for no extra information.
    splits = {name: RM.fit_split(frame, builder(frame)) for name, builder in SPLITS.items()}
    return RealData(
        players=frame,
        metrics=pd.concat([f.metrics for f in splits.values()], ignore_index=True),
        fitted_random=splits["random"],
        fitted_clubs=splits["clubs"],
        labels=COMPETITION_LABELS,
    )


@st.cache_resource(show_spinner=False)
def load_cabinet(code: str) -> pd.DataFrame:
    return FD.trophy_cabinet(code)


@st.cache_resource(show_spinner=False)
def load_table(code: str) -> pd.DataFrame:
    return FD.league_table(code)


def api_ready() -> bool:
    return FD.available()


def player_label(row: pd.Series) -> str:
    return f"{row['name']} · {row['position']} · {row['age']:.0f} · {row['club']} · {eur(row['value_eur'])}"


def options(data: RealData) -> list[str]:
    """Unique player names, ordered by best value so the box opens on the biggest.

    Names repeat when the snapshot lists a player at two clubs. Repeating them
    here would just show the same label twice, so the row is picked by index
    instead and the label carries the club.
    """
    ranked = data.players.sort_values("value_eur", ascending=False)
    return ranked["name"].drop_duplicates().tolist()


def eur(value: float) -> str:
    return f"EUR {value:,.0f}"


def eur_m(value: float) -> str:
    if value >= 1e6:
        return f"EUR {value / 1e6:,.1f}M"
    if value >= 1e3:
        return f"EUR {value / 1e3:,.0f}K"
    return f"EUR {value:,.0f}"


def metrics_frame(data: RealData) -> pd.DataFrame:
    """Metrics table with the honest headline: the gap between the two splits."""
    out = data.metrics.copy()
    out["mae_eur"] = out["mae_eur"].map(lambda v: eur(v))
    out["mape_pct"] = out["mape_pct"].map(lambda v: f"{v:.1f}%")
    out["r2"] = out["r2"].map(lambda v: f"{v:.3f}")
    out["mae_log"] = out["mae_log"].map(lambda v: f"{v:.4f}")
    out["split"] = out["split"].map(
        lambda v: "Random 80/20" if v == "random" else "Leave clubs out"
    )
    return out.rename(
        columns={
            "model": "Model",
            "split": "Split",
            "mae_eur": "MAE (EUR)",
            "mae_log": "MAE (log)",
            "mape_pct": "MAPE",
            "r2": "R2",
            "n_test": "Test players",
        }
    )
