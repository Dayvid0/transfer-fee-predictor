"""Turn the raw transfer table into a modelling frame.

Every column kept here is knowable *before* the transfer window opens: the
club's own valuation, the player's last-season production, contract length and
the calendar season. ``fee_eur`` becomes the target and never a feature, and
zero-fee loans/free transfers are dropped because they are not a paid market.

Money is also expressed in **base-year euros**. Fees inflate ~10% a year, so a
model fitted on nominal fees has to extrapolate a trend into seasons it has
never seen. Fitting the real-terms target instead makes the relationship
stationary; ``inflate_log`` puts predictions back into nominal euros for
scoring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C

KEY_COLUMNS = (
    "player_id",
    "season",
    "position",
    "league",
    "fee_eur",
    "market_value_eur",
    "log_fee_signal",
)
RESERVED_COLUMNS = set(KEY_COLUMNS) | {
    "target",
    "target_nominal",
    "log_fee_index",
    "is_free_or_loan",
    "origin_league",
}


def load_raw(path=None) -> pd.DataFrame:
    return pd.read_csv(C.RAW_CSV if path is None else path)


def drop_unpaid(df: pd.DataFrame) -> pd.DataFrame:
    """Loans and free transfers carry no market price signal."""
    return df.loc[df["fee_eur"] > 0].reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Nominal log-fee target plus pre-transfer design columns."""
    d = drop_unpaid(df).copy()

    d["target_nominal"] = np.log1p(d["fee_eur"])
    d["target"] = d["target_nominal"]

    # Age enters quadratically: value peaks in the mid-twenties.
    d["age_sq"] = d["age"] ** 2

    # Aggregated output rates, all per 90 from the season before the move.
    d["goal_contrib_p90"] = d["goals_p90"] + d["assists_p90"]
    d["xg_over_shots"] = d["xg_p90"] - d["goals_p90"]

    # Exposure and deal context.
    d["log_minutes"] = np.log1p(d["minutes"])
    d["output_x_minutes"] = d["goal_contrib_p90"] * np.log1p(d["minutes"]) / 100.0
    d["contract_x_age"] = d["contract_years_left"] * (30.0 - d["age"])

    # Linear time trend, so the model can see the calendar at all.
    d["season_index"] = d["season"] - C.BASE_SEASON

    categorical = pd.get_dummies(
        d[["position", "league"]].astype(str),
        prefix=["pos", "lg"],
        drop_first=True,
        dtype=float,
    )

    numeric = [
        "age",
        "age_sq",
        "goals_p90",
        "assists_p90",
        "xg_p90",
        "goal_contrib_p90",
        "xg_over_shots",
        "minutes",
        "log_minutes",
        "output_x_minutes",
        "contract_years_left",
        "contract_x_age",
        "season_index",
    ]
    features = pd.concat([d[numeric].astype(float), categorical], axis=1)

    bad = [c for c in C.NON_FEATURE_COLUMNS if c in features.columns]
    if bad:
        raise ValueError(f"post-transfer columns leaked into features: {bad}")

    keys = d[list(KEY_COLUMNS)].reset_index(drop=True)
    targets = d[["target", "target_nominal"]].reset_index(drop=True)
    return (
        pd.concat([keys, targets, features], axis=1).reset_index(drop=True)
    )


def estimate_price_index(frame: pd.DataFrame, last_season: int) -> float:
    """Annual fee inflation estimated from the training seasons only.

    Uses the median log fee per season and its slope, which is robust to the
    fee distribution's long right tail. The true generator uses 10%/yr; this
    recovers it from data rather than assuming it.
    """
    paid = frame.loc[frame["season"] <= last_season]
    season_medians = paid.groupby("season")["target_nominal"].median()
    slope = np.polyfit(
        season_medians.index.to_numpy(dtype=float),
        season_medians.to_numpy(dtype=float),
        1,
    )[0]
    return float(np.exp(slope))


def deflate(frame: pd.DataFrame, index: float) -> pd.DataFrame:
    """Restate money in base-year euros.

    ``target`` becomes the real-terms log fee that the models fit, and the
    valuation columns are expressed in base-year euros too so that no feature
    carries a non-stationary price level.
    """
    d = frame.copy()
    log_index = np.log(index)
    d["log_fee_index"] = d["season_index"] * log_index

    d["target"] = d["target_nominal"] - d["log_fee_index"]
    d["market_value_real"] = d["market_value_eur"] / np.exp(d["log_fee_index"])
    d["log_market_value_real"] = np.log1p(d["market_value_real"])
    return d


def inflate_log(log_real: np.ndarray, log_fee_index: np.ndarray) -> np.ndarray:
    """Real-terms log prediction back to the nominal log-fee scale."""
    return np.asarray(log_real) + np.asarray(log_fee_index)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Everything usable as a design-matrix column."""
    return [c for c in frame.columns if c not in RESERVED_COLUMNS]


def load_features(
    path=None, last_season: int = C.TRAIN_LAST_SEASON
) -> tuple[pd.DataFrame, float]:
    """Return ``(deflated_frame_with_keys, price_index)``."""
    frame = build_features(load_raw(path))
    index = estimate_price_index(frame, last_season)
    return deflate(frame, index), index