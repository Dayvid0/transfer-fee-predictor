"""Predict real player market value from a player's own statistics.

Target is ``log1p(value_eur)`` from a June-2024 snapshot of 2,570 real players.
Unlike the synthetic pipeline there is **no time axis**, so a chronological
split is impossible. Two splits are reported side by side: ``random`` (a plain
80/20 shuffle) and ``clubs`` (whole clubs held out, so test players sit at clubs
the model has never seen; a player listed at two clubs is dropped from the test
side so nobody straddles the boundary).

The gap between them is the point: the random split is flattered by club
identity leaking across, and the leave-clubs-out number is what the model is
worth on a genuinely unseen club.

Deliberately **not** features: ``value_eur`` (the target), ``wage_eur`` and
``release_clause_eur`` (the same vendor derives them from the value, so they
would smuggle the label back in), and ``club`` (unknowable by construction under
the leave-clubs-out split).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C
from src.real_data import load

TARGET = "value_eur"
CATEGORICAL = ["position", "foot"]
TEST_FRACTION = 0.2

# Never features: the label, vendor figures derived from it, and identity.
EXCLUDED_NUMERIC = {"value_eur", "wage_eur", "release_clause_eur"}
EXCLUDED_ANY = {
    "name", "search_key", "club", "player_id", "contract_joined", "contract_expiry",
    "on_loan", "Height", "Weight", "Value", "Wage", "Release clause",
    "Team & Contract", "Age", "ID", "Best position",
} | set(CATEGORICAL)


@dataclass
class Split:
    train: pd.DataFrame
    test: pd.DataFrame
    name: str
    dropped_players: int = 0


@dataclass
class Fitted:
    """Fitted estimators plus the exact feature layout they were trained on."""

    split: Split
    columns: list
    baseline: Pipeline
    linear: Pipeline
    booster: HistGradientBoostingRegressor
    metrics: pd.DataFrame = field(default_factory=pd.DataFrame)


def feature_columns(frame: pd.DataFrame) -> tuple[list, list]:
    """Numeric and categorical feature names, derived from the frame itself."""
    banned = EXCLUDED_ANY | EXCLUDED_NUMERIC
    numeric = [
        c for c in frame.columns
        if c not in banned and pd.api.types.is_numeric_dtype(frame[c])
    ]
    return numeric, [c for c in CATEGORICAL if c in frame.columns]


def make_matrix(frame: pd.DataFrame, numeric: list, categorical: list):
    """Design matrix for the estimator, imputing the unknown-contract rows."""
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]),
                numeric,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )


def random_split(frame: pd.DataFrame, frac: float = TEST_FRACTION, seed: int = C.SEED) -> Split:
    shuffled = frame.sample(frac=1 - frac, random_state=seed)
    test = frame.drop(index=shuffled.index)
    return Split(train=shuffled, test=test, name="random")


def clubs_split(frame: pd.DataFrame, frac: float = TEST_FRACTION, seed: int = C.SEED) -> Split:
    """Hold out whole clubs, then evict any player also present in training."""
    rng = np.random.default_rng(seed)
    counts = frame["club"].value_counts()
    chosen, rows = [], 0
    for club in rng.permutation(counts.index.to_numpy()):
        chosen.append(club)
        rows += int(counts[club])
        if rows >= frac * len(frame):
            break

    test = frame.loc[frame["club"].isin(chosen)]
    train = frame.loc[~frame["club"].isin(chosen)]
    straddling = test["player_id"].isin(train["player_id"])
    return Split(
        train=train,
        test=test.loc[~straddling],
        name="clubs",
        dropped_players=int(straddling.sum()),
    )


def _metrics(name: str, split: str, pred_log: np.ndarray, test: pd.DataFrame) -> dict:
    actual = np.expm1(np.log1p(test[TARGET].to_numpy()))
    pred = np.expm1(pred_log)
    err = pred - actual
    # 59 rows in the snapshot are club-less (a national team stands in for the
    # club) and carry a value of 0, so a percentage error is undefined there.
    priced = actual > 0
    return {
        "model": name,
        "split": split,
        "mae_eur": float(np.abs(err).mean()),
        "mae_log": float(np.abs(pred_log - np.log1p(actual)).mean()),
        "mape_pct": float((np.abs(err[priced]) / actual[priced]).mean() * 100),
        "r2": float(1 - (err**2).sum() / ((actual - actual.mean()) ** 2).sum()),
        "n_test": len(test),
    }


def fit_split(frame: pd.DataFrame, split: Split) -> Fitted:
    """Fit the baseline, the linear model and the booster on one split."""
    numeric, categorical = feature_columns(frame)
    columns = make_matrix(frame, numeric, categorical)

    x_tr = columns.fit_transform(split.train)
    x_te = columns.transform(split.test)
    y_tr = np.log1p(split.train[TARGET].to_numpy())
    y_te_log = np.log1p(split.test[TARGET].to_numpy())

    # A: a scout's single number. One feature, taken from the same layout.
    at = [numeric.index("overall")]
    baseline = LinearRegression().fit(x_tr[:, at], y_tr)
    linear = LinearRegression().fit(x_tr, y_tr)
    booster = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, random_state=C.SEED
    ).fit(x_tr, y_tr)

    rows = [
        _metrics("A. Overall rating (scout)", split.name, baseline.predict(x_te[:, at]), split.test),
        _metrics("B. Linear regression", split.name, linear.predict(x_te), split.test),
        _metrics("C. Gradient boosting", split.name, booster.predict(x_te), split.test),
    ]
    return Fitted(split, columns, baseline, linear, booster, pd.DataFrame(rows))


def usable(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows worth modelling: priced, and each player counted once.

    Club-less players (a national team stands in for the club) carry a value of
    0, which makes a percentage error undefined. And the source repeats a player
    once per competition they appear in, so 791 of the 3,001 rows are exact
    copies: only 2,570 players exist, and keeping the copies would weight them.
    """
    priced = frame.loc[frame[TARGET] > 0]
    keep = [c for c in priced.columns if c != "player_id"]
    return priced.drop_duplicates(subset=keep).reset_index(drop=True)


def build(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The player table plus both splits' metrics in one frame."""
    frame = usable(load(force=force))
    metrics = pd.concat(
        [fit_split(frame, s).metrics for s in (random_split(frame), clubs_split(frame))],
        ignore_index=True,
    )
    return frame, metrics


def predict(fitted: Fitted, rows: pd.DataFrame) -> dict:
    """Predicted market value in EUR for one or more player rows."""
    x = fitted.columns.transform(rows)
    names = list(fitted.columns.named_transformers_["num"].feature_names_in_)
    at = [names.index("overall")]
    return {
        "scout": np.expm1(fitted.baseline.predict(x[:, at])),
        "linear": np.expm1(fitted.linear.predict(x)),
        "booster": np.expm1(fitted.booster.predict(x)),
    }


if __name__ == "__main__":
    frame, table = build()
    print(f"{len(frame):,} real players\n")
    print(table.to_string(index=False))
