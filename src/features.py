"""Turn the raw transfer table into a modelling frame.

Every column kept here is knowable *before* the transfer window opens: the
club's own valuation, the player's last-season production, contract length and
the calendar season. ``fee_eur`` becomes the target and never a feature, and
zero-fee loans/free transfers are dropped because they are not a paid market.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C


def load_raw(path=None) -> pd.DataFrame:
    return pd.read_csv(C.RAW_CSV if path is None else path)


def drop_unpaid(df: pd.DataFrame) -> pd.DataFrame:
    """Loans and free transfers carry no market price signal."""
    return df.loc[df["fee_eur"] > 0].reset_index(drop=True)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric + dummy design matrix with the log-fee target attached."""
    d = drop_unpaid(df).copy()

    d["target"] = np.log1p(d["fee_eur"])

    # Age enters quadratically: value peaks in the mid-twenties.
    d["age_sq"] = d["age"] ** 2

    # Aggregated output rates, all per 90 from the season before the move.
    d["goal_contrib_p90"] = d["goals_p90"] + d["assists_p90"]
    d["xg_over_shots"] = d["xg_p90"] - d["goals_p90"]

    # Exposure and deal context.
    d["log_minutes"] = np.log1p(d["minutes"])
    d["output_x_minutes"] = d["goal_contrib_p90"] * np.log1p(d["minutes"]) / 100.0
    d["contract_x_age"] = d["contract_years_left"] * (30.0 - d["age"])

    # Log market value: the strongest single prior, on the same scale as target.
    d["log_market_value"] = np.log1p(d["market_value_eur"])

    # Linear time trend, so the model can absorb fee inflation.
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
        "market_value_eur",
        "log_market_value",
        "season_index",
    ]
    features = pd.concat([d[numeric].astype(float), categorical], axis=1)

    bad = [c for c in C.NON_FEATURE_COLUMNS if c in features.columns]
    if bad:
        raise ValueError(f"post-transfer columns leaked into features: {bad}")

    features.insert(0, "target", d["target"].to_numpy())
    return features.reset_index(drop=True)


KEY_COLUMNS = ("player_id", "season", "position", "league", "fee_eur")


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Everything usable as a design-matrix column."""
    reserved = set(KEY_COLUMNS) | {"target"}
    return [c for c in frame.columns if c not in reserved]


def attach_keys(frame: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Re-join identifying columns for splitting and error inspection.

    These are never design-matrix columns; ``feature_columns`` filters them
    back out.
    """
    keys = drop_unpaid(raw)[list(KEY_COLUMNS)].reset_index(drop=True)
    return pd.concat([keys, frame.reset_index(drop=True)], axis=1)


def load_features(path=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(frame_with_keys, frame_without_keys)``."""
    raw = load_raw(path)
    model = build_features(raw)
    keyed = attach_keys(model, raw)
    return keyed, model