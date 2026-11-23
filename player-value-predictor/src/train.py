"""Phase 2b: select on early validation, calibrate later, then freeze.

fit never loads test rows. evaluate-test requires an intact frozen bundle and
refuses to overwrite a completed test evaluation. No test result changes fit.
"""
import argparse
import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

import config
from baseline_report import load_development_table
from baselines import predict_baselines
from evaluate import regression_metrics, score_by_slice, interval_metrics
from features import make_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_COLUMNS, INPUT_COLUMNS

SELECTION_END = "2024-12-31"
RECENT_START = "2020-07-01"
COVERAGE = 0.80
SEED = 42
ARTIFACT_DIR = config.ROOT / "artifacts" / "phase2b"


def candidates():
    """Small, declared search; no calibration/test-driven expansion."""
    result = []
    for history in ("all", "recent"):
        for alpha in (1.0, 10.0, 100.0):
            result.append({"family": "ridge", "history": history, "alpha": alpha})
        for leaves, iterations in ((15, 200), (31, 300)):
            result.append({"family": "hist_gradient_boosting", "history": history,
                           "max_leaf_nodes": leaves, "max_iter": iterations})
    return result


def candidate_name(spec):
    params = {k: v for k, v in spec.items() if k not in ("family", "history")}
    return f"{spec['family']}_{spec['history']}_" + "_".join(f"{k}={v}" for k, v in sorted(params.items()))


def make_pipeline(spec):
    numeric = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if spec["family"] == "ridge":
        numeric.append(("scale", StandardScaler()))
        estimator = Ridge(alpha=spec["alpha"])
    elif spec["family"] == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            max_leaf_nodes=spec["max_leaf_nodes"], max_iter=spec["max_iter"],
            learning_rate=0.06, l2_regularization=10.0, min_samples_leaf=30,
            early_stopping=False, random_state=SEED,
        )
    else:
        raise ValueError(f"Unknown model family: {spec['family']}")
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline(numeric), NUMERIC_FEATURES),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocessing), ("regressor", estimator)])


def eligible_history(table, spec, cutoff):
    start = RECENT_START if spec["history"] == "recent" else config.FIRST_VALUATION_DATE
    if spec["history"] not in ("all", "recent"):
        raise ValueError("Unknown history policy")
    dates = pd.to_datetime(table.valuation_date)
    return table.loc[dates.between(start, cutoff)].copy()


def validate_rows(rows):
    if rows.empty:
        raise ValueError("Expected nonempty valuation rows")
    dates = pd.to_datetime(rows.valuation_date, errors="raise")
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():
        raise ValueError("Valuation dates must be present and have no time component")
    if rows.player_id.isna().any() or rows.duplicated(["player_id", "valuation_date"]).any():
        raise ValueError("Valuation rows require unique, present (player_id, date) keys")
    values = rows.value_eur.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Listed values must be finite and positive")
    make_features(rows)


def fit_model(rows, spec):
    validate_rows(rows)
    model = make_pipeline(spec)
    with threadpool_limits(limits=2):
        model.fit(make_features(rows), np.log1p(rows.value_eur.to_numpy(dtype=float)))
    return model


def predict_values(model, rows):
    """Shared training/serving route; accept raw feature inputs, return EUR."""
    with threadpool_limits(limits=2):
        log_predictions = np.asarray(model.predict(make_features(rows)), dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        values = np.maximum(np.expm1(log_predictions), 0.0)
    if not np.isfinite(values).all():
        raise ValueError("Model returned nonfinite predictions")
    return values


def calibrate_radius(actual, predicted, coverage=COVERAGE):
    """Finite-sample split-conformal rank in log1p(EUR) residuals.

    Temporal drift and repeated players break exchangeability; the requested
    coverage is a nominal target, not a guaranteed coverage claim.
    """
    regression_metrics(actual, predicted)  # validates values and alignment
    if not 0 < coverage < 1:
        raise ValueError("Coverage must be between zero and one")
    residuals = np.abs(np.log1p(np.asarray(actual, dtype=float)) - np.log1p(predicted))
    rank = math.ceil((len(residuals) + 1) * coverage)
    if rank > len(residuals):
        raise ValueError("Too few calibration rows for finite bounds at this coverage")
    return float(np.sort(residuals)[rank - 1])


def prediction_range(predicted, radius):
    predicted = np.asarray(predicted, dtype=float)
    if predicted.ndim != 1 or not np.isfinite(predicted).all() or (predicted < 0).any():
        raise ValueError("Predictions must be finite nonnegative 1D euro values")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Radius must be finite and nonnegative")
    lower = np.maximum(np.expm1(np.log1p(predicted) - radius), 0.0)
    upper = np.expm1(np.log1p(predicted) + radius)
    if not np.isfinite(upper).all():
        raise ValueError("Range upper bound is not finite")
    return lower, upper


def development_periods(table):
    dates = pd.to_datetime(table.valuation_date, errors="raise")
    # Discard everything outside development before feature/target inspection.
    development = table.loc[dates.between(config.FIRST_VALUATION_DATE, config.VALID_END)].copy()
    validate_rows(development)
    dates = pd.to_datetime(development.valuation_date)
    train = development.loc[dates <= config.TRAIN_END].copy()
    selection = development.loc[(dates > config.TRAIN_END) & (dates <= SELECTION_END)].copy()
    calibration = development.loc[(dates > SELECTION_END) & (dates <= config.VALID_END)].copy()
    if any(frame.empty for frame in (train, selection, calibration)):
        raise ValueError("Training, selection and calibration periods must all be nonempty")
    return train, selection, calibration


def choose_model(train, selection, specs=None, progress=None):
    """Only these two periods are available to model selection."""
    if pd.to_datetime(train.valuation_date).max() > pd.Timestamp(config.TRAIN_END):
        raise ValueError("Candidate fitting cannot use validation rows")
    dates = pd.to_datetime(selection.valuation_date)
    if not ((dates > config.TRAIN_END) & (dates <= SELECTION_END)).all():
        raise ValueError("Selection must use early validation only")
    validate_rows(train)
    validate_rows(selection)
    rows = []
    predictions = predict_baselines(train, selection)
    specs = candidates() if specs is None else specs
    if not specs:
        raise ValueError("At least one candidate is required")
    for spec in specs:
        history = eligible_history(train, spec, config.TRAIN_END)
        model = fit_model(history, spec)
        name = candidate_name(spec)
        predicted = predict_values(model, selection)
        predictions[name] = predicted
        scores = regression_metrics(selection.value_eur, predicted)
        rows.append({"candidate": name, "spec": spec, "fit_rows": len(history), **scores})
        if progress:
            progress(f"{name}: selection MAE EUR {scores['mae_eur']:,.0f}; RMSLE {scores['rmsle']:.4f}")
    # Predeclared selection rule; no calibration/test metrics enter this choice.
    ranked = sorted(rows, key=lambda row: (row["mae_eur"], row["rmsle"], row["candidate"]))
    return ranked[0]["spec"], rows, score_by_slice(selection, predictions, train.player_id)


def build_bundle(table, specs=None, progress=None):
    train, selection, calibration = development_periods(table)
    spec, search, selection_scores = choose_model(train, selection, specs, progress)
    fit_pool = pd.concat([train, selection], ignore_index=True)
    fitted_rows = eligible_history(fit_pool, spec, SELECTION_END)
    model = fit_model(fitted_rows, spec)
    predicted = predict_values(model, calibration)
    radius = calibrate_radius(calibration.value_eur.to_numpy(), predicted)
    lower, upper = prediction_range(predicted, radius)
    predictions = predict_baselines(fit_pool, calibration)
    predictions["selected_model"] = predicted
    calibration_scores = score_by_slice(calibration, predictions, fitted_rows.player_id, {"selected_model": (lower, upper)})
    report = {
        "selection_rule": "Lowest early-validation MAE; then RMSLE; then candidate name.",
        "selection": search,
        "selection_scores": records(selection_scores),
        "calibration_scores": records(calibration_scores),
        "calibration_intervals": interval_metrics(calibration.value_eur, lower, upper),
        "calibration_note": "Calibration coverage is in-sample for range construction, not independent validation.",
        "test_status": "closed",
    }
    bundle = {"model": model, "radius": radius, "nominal_coverage": COVERAGE,
              "spec": spec, "fit_end": SELECTION_END,
              "fit_player_ids": sorted(int(x) for x in fitted_rows.player_id.unique()),
              "feature_columns": FEATURE_COLUMNS, "format_version": 1}
    metadata = {
        "chosen_model": spec, "model_name": candidate_name(spec),
        "fit_start": str(fitted_rows.valuation_date.min().date()), "fit_end": SELECTION_END,
        "fit_rows": len(fitted_rows), "selection_start_exclusive": config.TRAIN_END,
        "selection_end": SELECTION_END, "calibration_start_exclusive": SELECTION_END,
        "calibration_end": config.VALID_END, "calibration_rows": len(calibration),
        "test_start_exclusive": config.VALID_END, "test_end": config.TEST_END,
        "nominal_coverage": COVERAGE, "log_radius": radius,
        "target": "Transfermarkt listed market value in EUR", "target_transform": "log1p",
        "selection_rule": report["selection_rule"],
        "range_method": "abs(log1p(actual)-log1p(predicted)); ceil((n+1)*coverage) order statistic",
        "range_limit": "Temporal drift and repeated players mean nominal coverage is not guaranteed.",
    }
    return bundle, metadata, report, fit_pool, pd.concat([train, selection, calibration], ignore_index=True)


def records(frame):
    # pandas handles numpy scalars and writes absent cohort metrics as JSON null.
    return json.loads(frame.to_json(orient="records", double_precision=12))


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def table_digest(frame):
    columns = ["player_id", "valuation_date", "value_eur", *INPUT_COLUMNS]
    data = frame[columns].sort_values(["valuation_date", "player_id"]).reset_index(drop=True)
    return hashlib.sha256(pd.util.hash_pandas_object(data, index=False).values.tobytes()).hexdigest()


def write_parquet(frame, path):
    with duckdb.connect(":memory:") as con:
        con.register("rows", frame)
        con.execute("COPY rows TO ? (FORMAT PARQUET)", [str(path)])


def player_lookup(development, db_path=None):
    """One historical demo row per eligible player; no DOB or snapshot features."""
    lookup = development.sort_values(["valuation_date", "player_id"]).drop_duplicates("player_id", keep="last").copy()
    lookup = lookup[["player_id", "valuation_date", "value_eur", *INPUT_COLUMNS]].reset_index(drop=True)
    lookup.insert(1, "player_name", pd.Series([None] * len(lookup), dtype="str"))
    if db_path is not None:
        with duckdb.connect(str(db_path), read_only=True) as con:
            ids = lookup[["player_id"]]
            con.register("selected_ids", ids)
            names = con.execute("SELECT p.player_id, p.name FROM players p JOIN selected_ids s USING (player_id)").df()
        if names.player_id.duplicated().any():
            raise ValueError("Player names must have unique IDs")
        lookup["player_name"] = lookup.player_id.map(names.set_index("player_id")["name"])
    return lookup


def freeze_artifacts(table_path, output, db_path=None, specs=None, progress=print):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Artifact directory is not empty; refusing to overwrite a prior run")
    table = load_development_table(table_path)  # the only data loaded by fit
    bundle, metadata, report, history, development = build_bundle(table, specs, progress)
    lookup = player_lookup(development, db_path)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / "model.joblib", compress=3)
    write_parquet(lookup, output / "players.parquet")
    write_parquet(history[["player_id", "valuation_date", "value_eur", "league", "position_group"]], output / "baseline_history.parquet")
    schema = {
        "format_version": 1, "input_columns": INPUT_COLUMNS, "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES,
        "allowed_leagues": config.TOP5, "allowed_positions": list(config.POSITION_GROUPS.values()),
        "missing_policy": "Only height may be missing; training-fitted median plus missing indicator.",
        "eligibility": f"Top-five league; positive listing; at least {config.MIN_MINUTES} prior-year minutes.",
        "lookup_as_of": config.VALID_END, "lookup_note": "Historical eligible valuations, not live player profiles.",
        "range": {"nominal_coverage": COVERAGE, "log_radius": bundle["radius"]},
    }
    write_json(output / "schema.json", schema)
    write_json(output / "development_metrics.json", report)
    metadata.update({
        "status": "frozen_before_test", "development_digest": table_digest(development),
        "versions": {name: version(name) for name in ("scikit-learn", "numpy", "pandas", "scipy", "joblib", "duckdb")},
        "python": platform.python_version(),
        "source_hashes": {name: digest(Path(__file__).parent / name) for name in
                          ("train.py", "features.py", "evaluate.py", "baselines.py", "config.py")},
        "artifact_hashes": {path.name: digest(path) for path in sorted(output.iterdir()) if path.is_file()},
    })
    # This is written last, so a partial fit cannot unlock test evaluation.
    write_json(output / "selection.json", metadata)
    if progress:
        progress(f"Frozen {metadata['model_name']}; {len(lookup):,} player lookup rows; test period still closed.")
    return metadata


def load_bundle(output):
    """Load only locally trusted joblib artifacts; hashes are integrity checks."""
    output = Path(output)
    metadata = json.loads((output / "selection.json").read_text())
    if metadata["status"] != "frozen_before_test":
        raise ValueError("Final model must be frozen before test evaluation")
    for name, expected in metadata["artifact_hashes"].items():
        if digest(output / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    for name, expected in metadata["versions"].items():
        if version(name) != expected:
            raise ValueError(f"Version mismatch for {name}; use the pinned training environment")
    for name, expected in metadata["source_hashes"].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError(f"Frozen inference/evaluation source changed: {name}")
    bundle = joblib.load(output / "model.joblib")
    if bundle["feature_columns"] != FEATURE_COLUMNS or bundle["spec"] != metadata["chosen_model"]:
        raise ValueError("Frozen model/schema mismatch")
    return bundle, metadata


def evaluate_test(table_path, output):
    output = Path(output)
    target = output / "test_metrics.json"
    if target.exists():
        raise FileExistsError("Test evaluation already exists; read it instead of reopening the test period")
    bundle, metadata = load_bundle(output)
    development = load_development_table(table_path)
    if table_digest(development) != metadata["development_digest"]:
        raise ValueError("Development data changed since final model selection")
    # Only after the frozen model, source, versions and data have been verified.
    with duckdb.connect(":memory:") as con:
        test = con.execute(
            "SELECT * FROM read_parquet(?) WHERE valuation_date > CAST(? AS DATE) "
            "AND valuation_date <= CAST(? AS DATE) ORDER BY valuation_date, player_id",
            [str(table_path), metadata["test_start_exclusive"], metadata["test_end"]],
        ).df()
        history = con.execute("SELECT * FROM read_parquet(?)", [str(output / "baseline_history.parquet")]).df()
    validate_rows(test)
    predictions = predict_baselines(history, test)
    predicted = predict_values(bundle["model"], test)
    predictions["selected_model"] = predicted
    lower, upper = prediction_range(predicted, bundle["radius"])
    report = {
        "stage": "final_test", "model_name": metadata["model_name"],
        "test_start_exclusive": metadata["test_start_exclusive"], "test_end": metadata["test_end"],
        "rows": len(test), "players": int(test.player_id.nunique()),
        "test_digest": table_digest(test), "selection_digest": digest(output / "selection.json"),
        "scores": records(score_by_slice(test, predictions, bundle["fit_player_ids"], {"selected_model": (lower, upper)})),
        "nominal_coverage": bundle["nominal_coverage"],
        "note": "Final test only; no model fitting, tuning, bias correction or recalibration performed.",
    }
    with target.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["fit", "evaluate-test"])
    parser.add_argument("--table", type=Path, default=config.TRAINING_TABLE)
    parser.add_argument("--output", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--db", type=Path, default=None, help="Optional raw DB for player names only")
    args = parser.parse_args()
    if args.action == "fit":
        freeze_artifacts(args.table, args.output, args.db)
    else:
        report = evaluate_test(args.table, args.output)
        scores = pd.DataFrame(report["scores"])
        print(scores.loc[scores.scope == "overall"].to_string(index=False))
        print(f"Final evaluation saved to {args.output / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
