import pandas as pd
import pytest

from baselines import predict_baselines
from phase2_fixtures import valuation


def history():
    return pd.DataFrame([
        valuation(1, "2024-01-01", 100), valuation(1, "2024-02-01", 200),
        valuation(2, "2024-03-01", 300, "ES1", "MID"),
    ])


def test_hand_computed_baselines_and_fallbacks_preserve_order_and_index():
    queries = pd.DataFrame([
        valuation(2, "2024-07-01", league="L1", position="DEF"),
        valuation(1, "2024-05-01"), valuation(3, "2024-06-01"),
        valuation(4, "2024-04-01", league="FR1", position="GK"),
    ], index=[6, 1, 1, 2])
    actual = predict_baselines(history(), queries)
    expected = pd.DataFrame({
        "global_median": [200., 200., 200., 200.],
        "league_position_median": [200., 150., 150., 200.],
        "previous_value": [300., 200., 150., 200.],
    }, index=queries.index)
    pd.testing.assert_frame_equal(actual, expected)


def test_same_day_and_later_history_cannot_change_earlier_predictions():
    # Every baseline is tested, including unseen-player and unseen-group fallbacks.
    queries = pd.DataFrame([
        valuation(1, "2024-02-01"), valuation(3, "2024-02-01"),
        valuation(4, "2024-02-01", league="FR1", position="GK"),
    ])
    initial = history().iloc[:1].copy()
    expected = predict_baselines(initial, queries)
    poisoned = pd.concat([initial, pd.DataFrame([
        valuation(1, "2024-02-01", 9_000_000),
        valuation(3, "2024-02-01", 8_000_000),
        valuation(4, "2024-02-01", 7_000_000, "FR1", "GK"),
        valuation(1, "2024-02-02", 6_000_000),
        valuation(3, "2024-02-02", 5_000_000),
        valuation(4, "2024-02-02", 4_000_000, "FR1", "GK"),
    ])], ignore_index=True).sample(frac=1, random_state=3)
    pd.testing.assert_frame_equal(predict_baselines(poisoned, queries), expected)


def test_all_prior_same_day_rows_contribute_to_medians_regardless_of_order():
    same_day = pd.DataFrame([valuation(1, value=100), valuation(2, value=300)])
    queries = pd.DataFrame([valuation(3, "2024-01-02")])
    expected = pd.DataFrame({
        "global_median": [200.], "league_position_median": [200.], "previous_value": [200.],
    })
    pd.testing.assert_frame_equal(predict_baselines(same_day, queries), expected)
    pd.testing.assert_frame_equal(predict_baselines(same_day.iloc[::-1], queries), expected)


def test_validation_targets_are_ignored_and_history_is_frozen():
    queries = pd.DataFrame([valuation(1, "2024-07-01", 500), valuation(1, "2025-01-01", 900)])
    reference = history()
    before = reference.copy(deep=True)
    expected = predict_baselines(reference, queries)
    assert expected.previous_value.tolist() == [200., 200.]
    queries["value_eur"] = [99_000_000, -999]
    pd.testing.assert_frame_equal(predict_baselines(reference, queries), expected)
    pd.testing.assert_frame_equal(predict_baselines(reference, queries.drop(columns="value_eur")), expected)
    pd.testing.assert_frame_equal(reference, before)


def test_latest_strictly_prior_value_is_selected_per_query_date():
    queries = pd.DataFrame([valuation(1, "2024-02-02"), valuation(1, "2024-02-01")])
    actual = predict_baselines(history(), queries)
    assert actual.previous_value.tolist() == [200., 100.]
    assert actual.global_median.tolist() == [150., 100.]


def test_no_prior_history_does_not_fall_back_to_future():
    with pytest.raises(ValueError, match="strictly earlier"):
        predict_baselines(history(), pd.DataFrame([valuation(day="2024-01-01")]))


def test_duplicate_player_date_is_rejected():
    reference = pd.concat([history(), history().iloc[:1]])
    with pytest.raises(ValueError, match="unique"):
        predict_baselines(reference, pd.DataFrame([valuation(day="2024-07-01")]))


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_history_values_are_rejected(value):
    reference = history()
    reference["value_eur"] = value
    with pytest.raises(ValueError):
        predict_baselines(reference, pd.DataFrame([valuation(day="2024-07-01")]))
