import math

import numpy as np
import pandas as pd
import pytest

from evaluate import regression_metrics, split_train_validation, summarize_predictions
from phase2_fixtures import valuation


def test_split_boundaries_and_no_test_returned():
    days = ["2013-06-30", "2013-07-01", "2024-06-30", "2024-07-01",
            "2025-06-30", "2025-07-01", "2026-06-12", "2026-06-13"]
    table = pd.DataFrame([valuation(day=day) for day in days]).iloc[::-1]
    before = table.copy(deep=True)
    train, valid = split_train_validation(table)
    assert set(train.valuation_date.dt.strftime("%Y-%m-%d")) == {"2013-07-01", "2024-06-30"}
    assert set(valid.valuation_date.dt.strftime("%Y-%m-%d")) == {"2024-07-01", "2025-06-30"}
    train.loc[:, "value_eur"] = 999
    pd.testing.assert_frame_equal(table, before)


@pytest.mark.parametrize("day", [None, "invalid", "2024-06-30 12:00"])
def test_invalid_split_dates_fail(day):
    with pytest.raises(ValueError):
        split_train_validation(pd.DataFrame({"valuation_date": [day]}))


def test_metrics_on_hand_computed_arrays():
    result = regression_metrics([1, 3], [3, 1])
    assert result == pytest.approx({
        "n": 2, "mae_eur": 2, "rmse_eur": 2, "rmsle": math.log(2),
        "median_ape_pct": 400 / 3, "mean_signed_error_eur": 0,
    })
    result = regression_metrics([100, 200], [80, 260])
    assert result["mae_eur"] == 40
    assert result["rmse_eur"] == pytest.approx(math.sqrt(2000))
    assert result["median_ape_pct"] == 25
    assert result["mean_signed_error_eur"] == 20


def test_signed_error_zero_predictions_and_perfect_predictions():
    result = regression_metrics([1, 3], [0, 0])
    assert result["mean_signed_error_eur"] == -2
    assert result["median_ape_pct"] == 100
    assert result["rmsle"] == pytest.approx(math.sqrt((math.log(2)**2 + math.log(4)**2) / 2))
    perfect = regression_metrics([1, 100], [1, 100])
    assert all(value == 0 for key, value in perfect.items() if key != "n")


@pytest.mark.parametrize("actual,predicted", [
    ([], []), ([1, 2], [1]), ([[1]], [[1]]), ([1], [np.nan]), ([np.inf], [1]),
    ([0], [1]), ([-1], [1]), ([1], [-1]),
])
def test_invalid_metric_inputs_fail(actual, predicted):
    with pytest.raises(ValueError):
        regression_metrics(actual, predicted)


def test_quarter_and_fixed_new_known_cohort_scores():
    valid = pd.DataFrame([
        valuation(1, "2024-07-01", 100), valuation(2, "2024-09-30", 200),
        valuation(2, "2024-10-01", 400), valuation(1, "2025-06-30", 200),
    ], index=[8, 2, 9, 4])
    predictions = pd.DataFrame({"example": [80, 220, 300, 100]}, index=valid.index)
    scores = summarize_predictions(valid, predictions, {1}).set_index(["scope", "segment"])
    assert scores.loc[("overall", "all"), "n"] == 4
    assert scores.loc[("period", "2024Q3"), "mean_signed_error_eur"] == 0
    assert scores.loc[("period", "2024Q4"), "mean_signed_error_eur"] == -100
    assert scores.loc[("period", "2025Q2"), "n"] == 1
    assert scores.loc[("player_cohort", "known"), "mean_signed_error_eur"] == -60
    assert scores.loc[("player_cohort", "new"), "mean_signed_error_eur"] == -40
    assert scores.loc[("player_cohort", "new"), "n"] == 2


def test_empty_cohort_has_count_zero_and_no_score():
    valid = pd.DataFrame([valuation(1, "2024-07-01")])
    scores = summarize_predictions(valid, pd.DataFrame({"example": [100]}), {1})
    empty = scores.loc[scores.segment == "new"].iloc[0]
    assert empty["n"] == 0
    assert pd.isna(empty["mae_eur"])


def test_report_scoring_rejects_test_rows_and_misaligned_predictions():
    rows = pd.DataFrame([valuation(day="2025-07-01")])
    with pytest.raises(ValueError, match="Only validation"):
        summarize_predictions(rows, pd.DataFrame({"example": [100]}), {1})
    rows["valuation_date"] = pd.Timestamp("2024-07-01")
    with pytest.raises(ValueError, match="index"):
        summarize_predictions(rows, pd.DataFrame({"example": [100]}, index=[2]), {1})
