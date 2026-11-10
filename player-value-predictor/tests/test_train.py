"""Phase 2b contracts tested on synthetic data; real test labels stay closed."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

import train
from evaluate import interval_metrics, score_by_slice
from features import FEATURE_COLUMNS, make_features
from phase2_fixtures import valuation

RIDGE = {"family": "ridge", "history": "all", "alpha": 10.0}
HIST = {"family": "hist_gradient_boosting", "history": "all", "max_leaf_nodes": 15, "max_iter": 10}


def sample_table():
    rows = []
    for year in range(2019, 2026):
        for month in (6, 12):
            for pid in range(1, 13):
                rows.append(valuation(pid, f"{year}-{month:02}-15", 100_000 * (pid + year - 2018),
                                      age_years=18 + pid, height_cm=np.nan if pid == 1 else 170 + pid,
                                      goals_365=pid, assists_365=pid // 2,
                                      league="ES1" if pid % 2 else "GB1", position="MID" if pid % 3 else "ATT"))
    return pd.DataFrame(rows)


def parquet(tmp_path, table):
    path = tmp_path / "table.parquet"
    train.write_parquet(table, path)
    return path


def test_selection_and_calibration_boundaries():
    days = ["2013-06-30", "2013-07-01", "2024-06-30", "2024-07-01", "2024-12-31",
            "2025-01-01", "2025-06-30", "2025-07-01", "2026-06-12"]
    table = pd.DataFrame([valuation(i, day) for i, day in enumerate(days)])
    fitting, selection, calibration = train.development_periods(table)
    assert fitting.valuation_date.dt.strftime("%Y-%m-%d").tolist() == days[1:3]
    assert selection.valuation_date.dt.strftime("%Y-%m-%d").tolist() == days[3:5]
    assert calibration.valuation_date.dt.strftime("%Y-%m-%d").tolist() == days[5:7]
    recent = train.eligible_history(table, {**RIDGE, "history": "recent"}, "2024-12-31")
    assert recent.valuation_date.min() >= pd.Timestamp("2020-07-01")
    assert recent.valuation_date.max() == pd.Timestamp("2024-12-31")


@pytest.mark.parametrize("spec", [RIDGE, HIST])
def test_fitted_preprocessing_is_training_only_and_batch_independent(spec):
    fitting = sample_table().iloc[:60].copy()
    fitting["league"] = "GB1"
    fitting["height_cm"] = 175.0
    fitting.loc[fitting.index[0], "height_cm"] = np.nan
    model = train.fit_model(fitting, spec)
    numeric = model.named_steps['preprocess'].named_transformers_['numeric']
    assert numeric.named_steps['impute'].statistics_[1] == 175
    categories = model.named_steps['preprocess'].named_transformers_['category'].categories_
    assert categories[0].tolist() == ['GB1']
    query = pd.DataFrame([valuation(999, "2025-02-01", league="ES1", height_cm=np.nan)])
    first = train.predict_values(model, query)
    later = pd.DataFrame([valuation(1000, "2026-01-01", height_cm=210, goals_365=40)])
    together = train.predict_values(model, pd.concat([query, later], ignore_index=True))
    np.testing.assert_allclose(first, together[:1])
    assert np.isfinite(first).all() and (first >= 0).all()
    assert numeric.named_steps['impute'].statistics_[1] == 175
    assert model.feature_names_in_.tolist() == FEATURE_COLUMNS
    if spec['family'] == 'hist_gradient_boosting':
        assert model.named_steps['regressor'].early_stopping is False


def test_conformal_order_statistic_and_range_math():
    # Five exactly known log residuals, finite-sample rank ceil(6*.8)=5.
    actual = np.expm1(np.array([1., 2., 3., 4., 5.]))
    radius = train.calibrate_radius(actual, np.zeros(5), 0.8)
    assert radius == pytest.approx(5)
    lo, hi = train.prediction_range([99., 0.], np.log(2))
    np.testing.assert_allclose(lo, [49., 0.])
    np.testing.assert_allclose(hi, [199., 1.])
    assert train.calibrate_radius([1, 2, 3, 4], [1, 2, 3, 4], .8) == 0
    with pytest.raises(ValueError, match="Too few"):
        train.calibrate_radius([1], [1], .8)
    with pytest.raises(ValueError):
        train.prediction_range([100], -1)


def test_interval_metrics_and_cohorts_are_hand_computed():
    assert interval_metrics([10, 20, 30, 40], [10, 10, 10, 10], [10, 30, 20, 40]) == {
        'coverage': .75, 'mean_width_eur': 15., 'median_width_eur': 15.}
    with pytest.raises(ValueError):
        interval_metrics([10], [20], [10])
    rows = pd.DataFrame([valuation(1, '2025-07-01', 10), valuation(2, '2025-10-01', 20)])
    scores = score_by_slice(rows, pd.DataFrame({'model': [10, 10]}), [1], {'model': ([5, 5], [15, 15])})
    assert scores.loc[scores.segment == 'known', 'coverage'].iloc[0] == 1
    assert scores.loc[scores.segment == 'new', 'coverage'].iloc[0] == 0
    assert scores.loc[scores.segment == '2025Q4', 'mean_signed_error_eur'].iloc[0] == -10


def test_calibration_and_test_labels_cannot_change_selection_or_fit():
    table = sample_table()
    baseline = train.build_bundle(table, [RIDGE, HIST])
    poisoned = table.copy()
    dates = poisoned.valuation_date
    poisoned.loc[dates > '2024-12-31', 'value_eur'] = 999_999_999
    poisoned.loc[dates > '2025-06-30', 'minutes_365'] = 0
    poisoned.loc[dates > '2025-06-30', 'value_eur'] = -1
    changed = train.build_bundle(poisoned, [RIDGE, HIST])
    assert baseline[0]['spec'] == changed[0]['spec']
    assert baseline[2]['selection'] == changed[2]['selection']
    np.testing.assert_allclose(train.predict_values(baseline[0]['model'], table.iloc[:10]),
                               train.predict_values(changed[0]['model'], table.iloc[:10]))
    assert baseline[0]['radius'] != changed[0]['radius']  # calibration labels affect only the range


def test_selection_rejects_later_periods():
    fitting, selection, calibration = train.development_periods(sample_table())
    with pytest.raises(ValueError, match='early validation'):
        train.choose_model(fitting, calibration, [RIDGE])
    with pytest.raises(ValueError, match='cannot use validation'):
        train.choose_model(pd.concat([fitting, selection]), selection, [RIDGE])


def test_frozen_artifact_round_trip_and_one_final_test(tmp_path):
    table = sample_table()
    path = parquet(tmp_path, table)
    output = tmp_path / 'artifacts'
    metadata = train.freeze_artifacts(path, output, specs=[RIDGE], progress=None)
    assert not (output / 'test_metrics.json').exists()
    assert metadata['calibration_end'] == '2025-06-30'
    assert metadata['fit_end'] == '2024-12-31'
    bundle, _ = train.load_bundle(output)
    before = train.predict_values(bundle['model'], table.iloc[:5])
    reloaded, _ = train.load_bundle(output)
    np.testing.assert_array_equal(before, train.predict_values(reloaded['model'], table.iloc[:5]))
    with duckdb.connect(':memory:') as con:
        lookup = con.execute('SELECT * FROM read_parquet(?)', [str(output / 'players.parquet')]).df()
    assert not lookup.player_id.duplicated().any()
    assert lookup.valuation_date.max() <= pd.Timestamp('2025-06-30')
    assert 'date_of_birth' not in lookup and 'current_club_id' not in lookup
    hashes = {p.name: train.digest(p) for p in output.iterdir()}
    report = train.evaluate_test(path, output)
    assert report['rows'] == 12
    assert report['stage'] == 'final_test'
    assert 'coverage' in next(row for row in report['scores'] if row['model'] == 'selected_model')
    for name, expected in hashes.items():
        assert train.digest(output / name) == expected
    with pytest.raises(FileExistsError, match='already exists'):
        train.evaluate_test(path, output)
    with pytest.raises(FileExistsError):
        train.freeze_artifacts(path, output, specs=[RIDGE])


def test_test_requires_frozen_intact_bundle_before_data_access(tmp_path, monkeypatch):
    path = parquet(tmp_path, sample_table())
    output = tmp_path / 'artifacts'
    train.freeze_artifacts(path, output, specs=[RIDGE], progress=None)
    (output / 'model.joblib').write_bytes(b'changed')
    monkeypatch.setattr(train, 'load_development_table', lambda path: pytest.fail('must verify first'))
    with pytest.raises(ValueError, match='artifact changed'):
        train.evaluate_test(path, output)


def test_fit_does_not_materialize_invalid_test_rows(tmp_path):
    table = sample_table()
    table.loc[table.valuation_date > '2025-06-30', ['value_eur', 'minutes_365']] = -999
    output = tmp_path / 'artifacts'
    train.freeze_artifacts(parquet(tmp_path, table), output, specs=[RIDGE], progress=None)
    assert json.loads((output / 'development_metrics.json').read_text())['test_status'] == 'closed'
