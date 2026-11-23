"""Three historical benchmarks with strictly earlier labels only.

The report supplies training history only: no validation labels update these
baselines. In addition, each lookup is as-of the query date with exact matches
disabled. Even if a caller supplies later history, it cannot change an earlier
prediction. Same-day labels are unavailable, independent of input row order.
"""
import numpy as np
import pandas as pd

BASELINE_COLUMNS = ["global_median", "league_position_median", "previous_value"]
QUERY_COLUMNS = ["player_id", "valuation_date", "league", "position_group"]


def _prepare(rows, columns):
    missing = sorted(set(columns) - set(rows.columns))
    if missing:
        raise ValueError(f"Missing baseline inputs: {missing}")
    result = rows[columns].copy().reset_index(drop=True)
    result["valuation_date"] = pd.to_datetime(result["valuation_date"], errors="raise")
    if result.isna().any().any():
        raise ValueError("Baseline inputs must not be missing")
    if not result["valuation_date"].eq(result["valuation_date"].dt.normalize()).all():
        raise ValueError("valuation_date must be a date without a time component")
    return result.sort_values("valuation_date", kind="stable").reset_index(drop=True)


def _asof(queries, history, by=None):
    """Return the latest strictly earlier reference, in the query's row order."""
    by = by or []
    left = queries[QUERY_COLUMNS].copy()
    left["_row"] = np.arange(len(left))
    matched = pd.merge_asof(
        left.sort_values("valuation_date", kind="stable"),
        history[["valuation_date", *by, "_prediction"]].sort_values("valuation_date", kind="stable"),
        on="valuation_date", by=by or None,
        direction="backward", allow_exact_matches=False,
    )
    return matched.sort_values("_row")["_prediction"].to_numpy()


def predict_baselines(history, rows):
    """Return euro predictions in rows' original index/order, ignoring targets.

    Medians weight valuation rows equally. The group median falls back to the
    global median for unseen league/position pairs. Previous value uses the
    player's latest earlier eligible historical row (including a prior league),
    falling back to group then global median. History is the filtered training
    table, not all raw Transfermarkt valuations. Queries with no earlier history
    fail explicitly; there is no future-derived or hard-coded fallback.
    """
    reference = _prepare(history, QUERY_COLUMNS + ["value_eur"])
    if reference.empty:
        raise ValueError("Baseline history must not be empty")
    reference["value_eur"] = pd.to_numeric(reference["value_eur"], errors="raise").astype(float)
    if not np.isfinite(reference["value_eur"]).all() or (reference["value_eur"] <= 0).any():
        raise ValueError("Historical values must be finite and positive")
    if reference.duplicated(["player_id", "valuation_date"]).any():
        raise ValueError("History must have unique (player_id, valuation_date) rows")
    # Validate without losing the caller's order or index (which need not be unique).
    _prepare(rows, QUERY_COLUMNS)
    queries = rows[QUERY_COLUMNS].copy().reset_index(drop=True)
    queries["valuation_date"] = pd.to_datetime(queries["valuation_date"])
    if queries.empty:
        return pd.DataFrame(index=rows.index, columns=BASELINE_COLUMNS, dtype=float)

    global_history = reference[["valuation_date", "value_eur"]].copy()
    global_history["_prediction"] = global_history["value_eur"].expanding().median()
    global_history = global_history.drop_duplicates("valuation_date", keep="last")
    global_prediction = _asof(queries, global_history)
    if np.isnan(global_prediction).any():
        raise ValueError("Every query needs at least one strictly earlier historical valuation")

    group_columns = ["league", "position_group"]
    group_history = reference.copy()
    group_history["_prediction"] = group_history.groupby(group_columns, sort=False)["value_eur"].transform(
        lambda values: values.expanding().median())
    group_history = group_history.drop_duplicates([*group_columns, "valuation_date"], keep="last")
    group_prediction = _asof(queries, group_history, by=group_columns)
    group_prediction = np.where(np.isnan(group_prediction), global_prediction, group_prediction)

    previous_history = reference.rename(columns={"value_eur": "_prediction"})
    previous_prediction = _asof(queries, previous_history, by=["player_id"])
    previous_prediction = np.where(np.isnan(previous_prediction), group_prediction, previous_prediction)
    return pd.DataFrame({
        "global_median": global_prediction,
        "league_position_median": group_prediction,
        "previous_value": previous_prediction,
    }, index=rows.index)[BASELINE_COLUMNS]
