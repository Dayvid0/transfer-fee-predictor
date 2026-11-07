import numpy as np
import pandas as pd
import pytest

from features import make_features
from phase2_fixtures import valuation


def test_per90_formulas_exact_column_order_and_index():
    rows = pd.DataFrame([
        valuation(),
        valuation(minutes_365=450, goals_365=1, assists_365=0, yellow_365=3, red_365=0),
    ], index=[7, 2])
    before = rows.copy(deep=True)
    features = make_features(rows[rows.columns[::-1]])
    assert features.columns.tolist() == [
        "age_years", "height_cm", "minutes_365", "appearances_365",
        "goals_per90", "assists_per90", "yellow_per90", "red_per90", "league", "position_group",
    ]
    assert features.index.tolist() == [7, 2]
    assert features.loc[7, ["goals_per90", "assists_per90", "yellow_per90", "red_per90"]].tolist() == [1, 0.5, 0.2, 0.1]
    assert features.loc[2, ["goals_per90", "assists_per90", "yellow_per90", "red_per90"]].tolist() == [0.2, 0, 0.6, 0]
    assert features.loc[7, "minutes_365"] == 900
    assert features.loc[7, "appearances_365"] == 12
    pd.testing.assert_frame_equal(rows, before)


def test_missing_height_remains_missing_and_later_rows_cannot_impute_it():
    rows = pd.DataFrame([valuation(height_cm=np.nan)])
    before = make_features(rows)
    assert pd.isna(before.loc[0, "height_cm"])
    later = pd.DataFrame([valuation(day="2026-01-01", height_cm=9999, goals_365=99999)])
    after = make_features(pd.concat([rows, later], ignore_index=True))
    pd.testing.assert_frame_equal(before, after.iloc[:1])


def test_features_work_without_target_id_or_date_and_ignore_poison_columns():
    rows = pd.DataFrame([valuation()])
    expected = make_features(rows)
    serving = rows.drop(columns=["player_id", "valuation_date", "value_eur"])
    pd.testing.assert_frame_equal(make_features(serving), expected)
    rows["value_eur"] = 999_999_999
    rows["player_id"] = 999
    rows["valuation_date"] = pd.Timestamp("2099-01-01")
    rows["market_value_in_eur"] = 999_999_999
    rows["current_club_id"] = 999
    pd.testing.assert_frame_equal(make_features(rows), expected)


@pytest.mark.parametrize("column,value", [
    ("minutes_365", 0), ("minutes_365", -1), ("goals_365", -1),
    ("assists_365", np.nan), ("height_cm", np.inf), ("age_years", np.nan),
    ("league", "ZZ9"), ("position_group", None),
])
def test_invalid_inputs_fail_explicitly(column, value):
    with pytest.raises(ValueError):
        make_features(pd.DataFrame([valuation(**{column: value})]))


def test_missing_required_input_fails_clearly():
    with pytest.raises(ValueError, match="Missing feature inputs.*goals_365"):
        make_features(pd.DataFrame([valuation()]).drop(columns="goals_365"))
