"""Dashboard helpers: cached pipeline access, EUR formatting, profile scoring.

The scoring path reuses ``features.build_features`` and ``features.deflate``
rather than hand-assembling a design row, so the one-hot baselines and derived
columns can never drift from what the models were trained on. Nothing here
refits or alters a model. ``PALETTE`` and ``CSS`` are the single source of truth
for the theme; the SVG and gauge builders live in ``src.visuals``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src import evaluate as E
from src import features as F
from src.train import fit_models

# The one palette. Charts, cards and SVGs all read from this dict.
PALETTE = {
    "accent": "#22C55E", "accent_soft": "#16A34A", "secondary": "#0EA5E9",
    "warn": "#F59E0B", "danger": "#EF4444", "muted": "#64748B",
    "text": "#E6EDF5", "on_accent": "#04180D",
    "pitch_a": "#0F2A1D", "pitch_b": "#081C13",
    "position": {
        "GK": "#38BDF8", "CB": "#818CF8", "LB": "#A78BFA", "RB": "#A78BFA",
        "DM": "#2DD4BF", "CM": "#34D399", "AM": "#22C55E",
        "LW": "#A3E635", "RW": "#A3E635", "ST": "#FACC15",
    },
}

# Ordered series colours for categorical plotly traces.
SERIES = [PALETTE["accent"], PALETTE["secondary"], PALETTE["warn"],
          PALETTE["muted"], PALETTE["danger"], "#C084FC"]

CSS = """
<style>
.stApp { background: radial-gradient(1200px 600px at 20% -10%, #10241a 0%, #0B1220 60%); }
.hero { border: 1px solid rgba(34,197,94,.35); border-radius: 16px; overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,.45); margin-bottom: 1rem; }
.pcard-row { display: flex; gap: .75rem; flex-wrap: wrap; }
.pcard { flex: 1 1 190px; background: linear-gradient(160deg,#16233a,#101a2c);
         border: 1px solid rgba(34,197,94,.30); border-radius: 14px; padding: .85rem;
         box-shadow: 0 6px 18px rgba(0,0,0,.40); }
.pcard-top { display: flex; align-items: center; gap: .65rem; }
.pcard-name { font-weight: 700; font-size: 1.02rem; color: #E6EDF5; line-height: 1.15; }
.pcard-sub { font-size: .78rem; color: #9FB3C8; margin-top: .15rem; }
.pcard-fee { margin-top: .7rem; font-size: 1.5rem; font-weight: 800; color: #22C55E;
            font-variant-numeric: tabular-nums; }
div[data-testid="stMetricValue"] { color: #22C55E; font-weight: 700; }
h1, h2, h3 { letter-spacing: -.01em; } .stCaption { color: #9FB3C8; }
</style>
"""

DEFAULT_PROFILE = {
    "age": 25, "position": "ST", "league": "Premier League", "goals_p90": 0.25,
    "xg_p90": 0.24, "minutes": 2400, "contract_years_left": 3.0,
    "market_value_eur": 10_000_000, "season": 2025,
}


def inject_css() -> None:
    """Apply the theme CSS. Called once per script run."""
    st.markdown(CSS, unsafe_allow_html=True)


@dataclass
class Pipeline:
    """Everything the dashboard reads, computed once per process."""
    raw: pd.DataFrame
    paid: pd.DataFrame
    frame: pd.DataFrame
    index: float
    models: object
    preds: pd.DataFrame
    metrics: pd.DataFrame
    importance: pd.DataFrame
    worst: pd.DataFrame
    assist_ratio: dict = field(default_factory=dict)

    @property
    def test(self) -> pd.DataFrame:
        return self.models.split.test

    @property
    def cols(self) -> list[str]:
        return self.models.split.cols

    def predictions_eur(self, column: str) -> np.ndarray:
        """Nominal-EUR predictions for one model on the test block."""
        return np.expm1(E.to_nominal(self.models, self.preds[column].to_numpy()))

    def test_actual_eur(self) -> np.ndarray:
        return self.test["fee_eur"].to_numpy()


def assist_ratios(raw: pd.DataFrame, last_season: int = C.TRAIN_LAST_SEASON) -> dict:
    """Median assists-per-goal by position, from training seasons only.

    The UI has no assists control, so it is inferred from the chosen position
    and scoring volume rather than invented.
    """
    paid = raw.loc[(raw["fee_eur"] > 0) & (raw["season"] <= last_season)]
    ratio = (paid["assists_p90"] / paid["goals_p90"].clip(lower=0.02)).groupby(
        paid["position"]
    ).median()
    return {str(k): float(v) for k, v in ratio.items()}


def build_pipeline(path=None, last_season: int = C.TRAIN_LAST_SEASON) -> Pipeline:
    """Run the existing pipeline end to end (no caching, no Streamlit state)."""
    raw = F.load_raw(path)
    frame, index = F.load_features(path, last_season=last_season)
    models = fit_models(frame, index)
    preds = E.predictions(models)
    return Pipeline(
        raw=raw,
        paid=raw.loc[raw["fee_eur"] > 0].reset_index(drop=True),
        frame=frame,
        index=index,
        models=models,
        preds=preds,
        metrics=E.metrics_table(models, preds),
        importance=E.feature_importance(models),
        worst=E.worst_errors(models, preds, "C_lightgbm", n=10),
        assist_ratio=assist_ratios(raw, last_season),
    )


@st.cache_resource(show_spinner=False)
def load_pipeline() -> Pipeline:
    """Session-wide cached pipeline; the models are fit exactly once."""
    return build_pipeline()


def profile_frame(
    profiles: pd.DataFrame, index: float, assist_ratio: dict
) -> pd.DataFrame:
    """Turn user-supplied profiles into rows of the real design matrix.

    Accepts one or many rows. A placeholder fee keeps the row past the
    zero-fee filter; it only feeds the dropped target column.
    """
    d = profiles.copy()
    d["season_index"] = d["season"] - C.BASE_SEASON
    d["assists_p90"] = (
        d["goals_p90"] * d["position"].map(assist_ratio).fillna(0.45)
    ).clip(lower=0.0)
    d["origin_league"] = d["league"]
    d["player_id"] = 0
    d["fee_eur"] = 1
    d["log_fee_signal"] = np.nan
    d["is_free_or_loan"] = False
    return F.deflate(F.build_features(d), index)


def predict_profiles(pipeline: Pipeline, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Nominal-EUR fee predictions for one or many profiles."""
    built = profile_frame(rows, pipeline.index, pipeline.assist_ratio)
    log_fee_index = built["log_fee_index"].to_numpy()
    market_value_real = built["market_value_real"].to_numpy()
    # A short profile only produces the dummies for the categories it happens
    # to use; missing ones are the zero baseline, so fill rather than NaN.
    matrix = built.reindex(columns=pipeline.cols, fill_value=0.0)
    out = {
        "linear": np.expm1(
            F.inflate_log(pipeline.models.linear.predict(matrix), log_fee_index)
        ),
        "lightgbm": np.expm1(
            F.inflate_log(pipeline.models.booster.predict(matrix), log_fee_index)
        ),
    }
    out["market_value"] = market_value_real * np.exp(log_fee_index)
    return out


def predict_fee(pipeline: Pipeline, profile: dict | None = None) -> dict[str, float]:
    """Predicted transfer fee in EUR for a single player profile."""
    profile = {**DEFAULT_PROFILE, **(profile or {})}
    out = predict_profiles(pipeline, pd.DataFrame([profile]))
    return {k: float(np.atleast_1d(v)[0]) for k, v in out.items()}


def age_curve(
    pipeline: Pipeline, profile: dict, ages: np.ndarray
) -> dict[str, np.ndarray]:
    """Predicted fee across an age range, holding the rest of the profile."""
    rows = pd.DataFrame([{**profile, "age": int(a)} for a in ages])
    return predict_profiles(pipeline, rows)


def eur(value: float) -> str:
    return f"EUR {value:,.0f}"


def eur_m(value: float) -> str:
    return f"EUR {value / 1e6:,.2f}M"