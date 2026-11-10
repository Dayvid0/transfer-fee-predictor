"""Phase 2a: validation-only baseline report; no models or saved artifacts.

Run from the project root: .venv/bin/python src/baseline_report.py
The optional --table argument selects an existing Phase 1 parquet table.
There is deliberately no option to evaluate the test period.
"""
import argparse

import duckdb

import config
from baselines import predict_baselines
from evaluate import split_train_validation, summarize_predictions
from features import make_features


def load_development_table(path):
    """Select train/validation rows in DuckDB before materializing any rows.

    The parquet file also contains test rows. Those rows are never returned to
    feature creation, history, predictions, metrics or report summaries.
    """
    with duckdb.connect(":memory:") as con:
        return con.execute(
            "SELECT * FROM read_parquet(?) "
            "WHERE valuation_date >= CAST(? AS DATE) AND valuation_date <= CAST(? AS DATE) "
            "ORDER BY valuation_date, player_id",
            [str(path), config.FIRST_VALUATION_DATE, config.VALID_END],
        ).df()


def build_report(table):
    """Return development metadata and validation scores; ignore later rows."""
    train, validation = split_train_validation(table)
    if train.empty or validation.empty:
        raise ValueError("Both training and validation must contain rows")
    # Check that the same feature contract works for both periods, without
    # learning imputation, encodings or any other full-table statistics.
    make_features(train)
    make_features(validation)
    predictions = predict_baselines(train, validation)
    return {
        "training_rows": len(train),
        "training_players": train.player_id.nunique(),
        "validation_rows": len(validation),
        "validation_players": validation.player_id.nunique(),
        "scores": summarize_predictions(validation, predictions, train.player_id),
    }


def render_report(report):
    """Readable stdout report; every displayed metric is validation-only."""
    lines = [
        "Phase 2a: validation-only baselines",
        f"Training: {config.FIRST_VALUATION_DATE} through {config.TRAIN_END}; "
        f"{report['training_rows']:,} rows / {report['training_players']:,} players",
        f"Validation: after {config.TRAIN_END} through {config.VALID_END}; "
        f"{report['validation_rows']:,} rows / {report['validation_players']:,} players",
        "Test period: closed (not loaded or scored).",
        "History: training labels only; frozen at the training cutoff.",
        "Previous value: latest strictly earlier eligible training valuation; group/global fallback.",
        "Known/new: membership in training, fixed for the entire validation period.",
        "Errors are per valuation row. Signed error = prediction - listing (negative = underestimate).",
        "MAE, RMSE and signed error: EUR; RMSLE: natural log1p(EUR); median APE: percent.",
        "",
    ]
    for scope, title in [("overall", "Overall validation"), ("period", "By calendar quarter"),
                         ("player_cohort", "By player cohort")]:
        scores = report["scores"].loc[report["scores"].scope == scope].drop(columns="scope")
        lines.extend([title, scores.to_string(index=False, float_format=lambda n: f"{n:,.4f}"), ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=config.TRAINING_TABLE, help="Phase 1 training-table parquet")
    args = parser.parse_args()
    print(render_report(build_report(load_development_table(args.table))))


if __name__ == "__main__":
    main()
