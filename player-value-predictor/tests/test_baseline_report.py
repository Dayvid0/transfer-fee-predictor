import duckdb
import pandas as pd
import pytest

from baseline_report import build_report, load_development_table, render_report
from phase2_fixtures import valuation


def development():
    return pd.DataFrame([
        valuation(1, "2024-06-30", 100),
        valuation(1, "2024-07-01", 200),
        valuation(2, "2025-06-30", 300),
    ])


def test_real_parquet_loader_excludes_test_rows_before_feature_validation(tmp_path):
    # A synthetic test row would fail features/metrics if it reached either.
    table = pd.concat([development(), pd.DataFrame([
        valuation(99, "2025-07-01", -999, minutes_365=0, league="ZZ9"),
        valuation(98, "2013-06-30", -999, minutes_365=0, league="ZZ9"),
    ])], ignore_index=True)
    path = tmp_path / "training_table.parquet"
    with duckdb.connect(":memory:") as con:
        con.register("fixture", table)
        con.execute("COPY fixture TO ? (FORMAT PARQUET)", [str(path)])
    loaded = load_development_table(path)
    pd.testing.assert_frame_equal(loaded, development(), check_dtype=False)
    report = build_report(loaded)
    overall = report["scores"].loc[report["scores"].scope == "overall"]
    assert overall["n"].tolist() == [2, 2, 2]
    assert overall.mae_eur.tolist() == [150., 150., 150.]
    assert overall.mean_signed_error_eur.tolist() == [-150., -150., -150.]
    text = render_report(report)
    assert "Test period: closed" in text
    assert "2025Q3" not in text


def test_poisoned_test_rows_cannot_change_report():
    expected = build_report(development())
    table = pd.concat([development(), pd.DataFrame([
        valuation(1, "2025-07-01", float("inf"), goals_365=-999),
        valuation(99, "2026-06-12", 999_999_999, minutes_365=0),
    ])], ignore_index=True)
    actual = build_report(table)
    pd.testing.assert_frame_equal(actual.pop("scores"), expected.pop("scores"))
    assert actual == expected


def test_validation_labels_do_not_update_later_baselines():
    table = development()
    expected = build_report(table)["scores"]
    table.loc[1, "value_eur"] = 900_000_000
    actual = build_report(table)["scores"]
    # July's own score changes; June's predictions and scores must not.
    pd.testing.assert_frame_equal(
        actual.loc[actual.segment == "2025Q2"], expected.loc[expected.segment == "2025Q2"],
    )


@pytest.mark.parametrize("selection", [[0], [1, 2]])
def test_report_requires_both_development_periods(selection):
    with pytest.raises(ValueError, match="Both training and validation"):
        build_report(development().iloc[selection])
