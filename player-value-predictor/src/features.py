"""Stateless features shared by training and serving.

Targets, identifiers, dates and snapshot columns never enter the feature frame.
Missing height is preserved; any learned imputation belongs inside the model's
training-only pipeline in Phase 2b.
"""
import numpy as np
import pandas as pd

import config

RATE_COLUMNS = {
    "goals_365": "goals_per90",
    "assists_365": "assists_per90",
    "yellow_365": "yellow_per90",
    "red_365": "red_per90",
}
NUMERIC_FEATURES = [
    "age_years", "height_cm", "minutes_365", "appearances_365",
    *RATE_COLUMNS.values(),
]
CATEGORICAL_FEATURES = ["league", "position_group"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
INPUT_COLUMNS = [
    "age_years", "height_cm", "minutes_365", "appearances_365",
    *RATE_COLUMNS, *CATEGORICAL_FEATURES,
]


def make_features(rows):
    """Return a new frame in FEATURE_COLUMNS order, preserving the row index.

    Each rate is count * 90 / minutes. Minutes must be positive: a rate at zero
    minutes is undefined. The training-table eligibility rule (>=450 minutes)
    remains the responsibility of the caller, separate from this transform.
    """
    missing = sorted(set(INPUT_COLUMNS) - set(rows.columns))
    if missing:
        raise ValueError(f"Missing feature inputs: {missing}")
    result = rows[INPUT_COLUMNS].copy()
    for column in INPUT_COLUMNS[:-2]:
        values = pd.to_numeric(result[column], errors="raise").astype(float)
        present = values.dropna() if column == "height_cm" else values
        if not np.isfinite(present).all() or (present < 0).any():
            raise ValueError(f"{column} must contain finite, nonnegative values")
        if column in ("age_years", "height_cm", "minutes_365") and (present <= 0).any():
            raise ValueError(f"{column} must be positive when present")
        result[column] = values
    if not result["league"].isin(config.TOP5).all():
        raise ValueError("league must be one of the configured top-five leagues")
    if not result["position_group"].isin(config.POSITION_GROUPS.values()).all():
        raise ValueError("position_group must be GK, DEF, MID or ATT")
    for source, target in RATE_COLUMNS.items():
        result[target] = result[source] * 90.0 / result["minutes_365"]
    return result[FEATURE_COLUMNS]
