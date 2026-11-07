"""Chronological development split and regression metrics in original euros."""
import numpy as np
import pandas as pd

import config

METRIC_COLUMNS = [
    "mae_eur", "rmse_eur", "rmsle", "median_ape_pct", "mean_signed_error_eur",
]


def split_train_validation(table):
    """Return independent train/validation frames; never return the test period.

    Dates are daily: train includes TRAIN_END; validation begins the next day
    and includes VALID_END. Rows outside the development window are excluded.
    """
    dates = pd.to_datetime(table["valuation_date"], errors="raise")
    if dates.isna().any():
        raise ValueError("valuation_date must not be missing")
    if not dates.eq(dates.dt.normalize()).all():
        raise ValueError("valuation_date must be a date without a time component")
    train = table.loc[dates.between(config.FIRST_VALUATION_DATE, config.TRAIN_END)].copy()
    valid = table.loc[(dates > config.TRAIN_END) & (dates <= config.VALID_END)].copy()
    for frame in (train, valid):
        frame["valuation_date"] = pd.to_datetime(frame["valuation_date"])
    return train, valid


def regression_metrics(y_true, y_pred):
    """Score aligned 1D arrays of euro values, weighting each valuation equally.

    Targets must be positive and predictions nonnegative, with no NaN/inf.
    RMSLE uses natural log1p on euro values. Median APE is a percentage.
    Signed error is prediction minus listing: negative means underestimation.
    Invalid predictions are rejected rather than silently clipped.
    """
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1 or actual.shape != predicted.shape:
        raise ValueError("Targets and predictions must be equally sized 1D arrays")
    if not len(actual):
        raise ValueError("Cannot score an empty sample")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Targets and predictions must be finite")
    if (actual <= 0).any() or (predicted < 0).any():
        raise ValueError("Targets must be positive and predictions nonnegative")
    error = predicted - actual
    return {
        "n": len(actual),
        "mae_eur": float(np.mean(np.abs(error))),
        "rmse_eur": float(np.sqrt(np.mean(error ** 2))),
        "rmsle": float(np.sqrt(np.mean((np.log1p(predicted) - np.log1p(actual)) ** 2))),
        "median_ape_pct": float(100 * np.median(np.abs(error) / actual)),
        "mean_signed_error_eur": float(np.mean(error)),
    }


def summarize_predictions(validation, predictions, training_player_ids):
    """Score overall, calendar quarters, and players known/new to training.

    The index must align exactly; inputs outside validation are rejected.
    Cohort membership is fixed at the training cutoff, even if a new player
    appears several times during validation. Empty cohorts have n=0 and NaN
    metrics, not a misleading zero error.
    """
    if not predictions.index.equals(validation.index):
        raise ValueError("Prediction index must match validation rows in order")
    dates = pd.to_datetime(validation["valuation_date"], errors="raise")
    if not ((dates > config.TRAIN_END) & (dates <= config.VALID_END)).all():
        raise ValueError("Only validation dates may be scored by this report")
    if validation.empty:
        raise ValueError("Validation must not be empty")
    quarters = dates.dt.to_period("Q").astype(str)
    known = validation["player_id"].isin(set(training_player_ids)).to_numpy()
    slices = [("overall", "all", np.ones(len(validation), dtype=bool))]
    slices.extend(("period", quarter, quarters.eq(quarter).to_numpy())
                  for quarter in sorted(quarters.unique()))
    slices.extend([("player_cohort", "known", known), ("player_cohort", "new", ~known)])
    scores = []
    for baseline in predictions.columns:
        actual = validation["value_eur"].to_numpy()
        predicted = predictions[baseline].to_numpy()
        for scope, segment, mask in slices:
            metrics = regression_metrics(actual[mask], predicted[mask]) if mask.any() else {
                "n": 0, **dict.fromkeys(METRIC_COLUMNS, float("nan")),
            }
            scores.append({"baseline": baseline, "scope": scope, "segment": segment, **metrics})
    return pd.DataFrame(scores)


def interval_metrics(actual, lower, upper):
    """Empirical coverage and width for inclusive euro-valued prediction bounds."""
    actual, lower, upper = (np.asarray(values, dtype=float) for values in (actual, lower, upper))
    regression_metrics(actual, lower)
    regression_metrics(actual, upper)
    if (lower > upper).any():
        raise ValueError("Lower bounds must not exceed upper bounds")
    return {
        "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "mean_width_eur": float(np.mean(upper - lower)),
        "median_width_eur": float(np.median(upper - lower)),
    }


def score_by_slice(rows, predictions, training_player_ids, ranges=None):
    """Explicit caller-supplied holdout: overall, quarters, cohorts, league/position.

    Does not load or choose a period. Phase 2b's fit and locked test entry points
    control data access. Known/new is fixed against the final model's fit rows.
    All prediction/range arrays follow the supplied row order.
    """
    if rows.empty or not predictions.index.equals(rows.index):
        raise ValueError("Nonempty rows and aligned prediction index required")
    dates = pd.to_datetime(rows.valuation_date, errors="raise")
    if dates.isna().any():
        raise ValueError("Valuation dates must not be missing")
    known = rows.player_id.isin(set(training_player_ids)).to_numpy()
    slices = [("overall", "all", np.ones(len(rows), dtype=bool)),
              ("player_cohort", "known", known), ("player_cohort", "new", ~known)]
    for scope, labels in (("period", dates.dt.to_period("Q").astype(str)),
                          ("league", rows.league), ("position", rows.position_group)):
        slices.extend((scope, label, labels.eq(label).to_numpy()) for label in sorted(labels.unique()))
    result = []
    actual = rows.value_eur.to_numpy()
    ranges = ranges or {}
    for name in predictions:
        predicted = predictions[name].to_numpy()
        regression_metrics(actual, predicted)
        if name in ranges:
            lower, upper = (np.asarray(bound, dtype=float) for bound in ranges[name])
            interval_metrics(actual, lower, upper)
        for scope, segment, mask in slices:
            metrics = regression_metrics(actual[mask], predicted[mask]) if mask.any() else {
                "n": 0, **dict.fromkeys(METRIC_COLUMNS, float("nan")),
            }
            if name in ranges:
                metrics.update(interval_metrics(actual[mask], lower[mask], upper[mask]) if mask.any() else
                               dict.fromkeys(["coverage", "mean_width_eur", "median_width_eur"], float("nan")))
            result.append({"model": name, "scope": scope, "segment": segment, **metrics})
    return pd.DataFrame(result)
