"""Scoring: one metrics table, LightGBM importances, worst errors."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import features as F
from src.train import TrainedModels

EUR_FMT = "{:,.0f}"


def markdown_table(frame: pd.DataFrame) -> str:
    """Minimal GitHub-flavoured markdown table (avoids a tabulate dep)."""
    headers = [str(c) for c in frame.columns]
    rows = [[str(v) for v in row] for row in frame.to_numpy()]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    rule = "| " + " | ".join("-" * w for w in widths) + " |"
    body = [
        "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"
        for row in rows
    ]
    return "\n".join([line, rule, *body])


def predictions(models: TrainedModels) -> pd.DataFrame:
    """Real-terms log predictions for all three models on the test block."""
    x_te = models.split.test[models.split.cols]
    return pd.DataFrame(
        {
            "A_market_value": np.asarray(models.baseline_log_real),
            "B_linear": models.linear.predict(x_te),
            "C_lightgbm": models.booster.predict(x_te),
        },
        index=models.split.test.index,
    )


def to_nominal(models: TrainedModels, log_real: np.ndarray) -> np.ndarray:
    """Real-terms log values back to the nominal log-fee scale."""
    return F.inflate_log(log_real, models.split.test["log_fee_index"].to_numpy())


def actual_log(models: TrainedModels) -> np.ndarray:
    """Nominal log1p fee actually paid, on the test block."""
    return models.split.test["target_nominal"].to_numpy()


def noise_floor(models: TrainedModels) -> dict[str, float]:
    """Oracle metrics: predicting the generator's noise-free log fee exactly.

    The negotiation shock is independent noise, so no model can score below
    this. Any model that does is reading something it should not have.
    """
    signal = models.split.test["log_fee_signal"].to_numpy()
    actual = actual_log(models)
    return {
        "mae_log": float(np.abs(signal - actual).mean()),
        "mae_eur": float(np.abs(np.exp(signal) - models.split.test["fee_eur"]).mean()),
    }


def _row(name: str, p_log: np.ndarray, actual_log_: np.ndarray, actual_eur: np.ndarray) -> dict:
    p_eur = np.expm1(p_log)
    err_eur = p_eur - actual_eur
    return {
        "model": name,
        "mae_log": float(np.abs(p_log - actual_log_).mean()),
        "mae_eur": float(np.abs(err_eur).mean()),
        "rmse_eur": float(np.sqrt((err_eur**2).mean())),
        "mape_pct": float((np.abs(err_eur) / actual_eur).mean() * 100),
        "bias_eur": float(err_eur.mean()),
        "r2_log": float(
            1
            - ((p_log - actual_log_) ** 2).sum()
            / ((actual_log_ - actual_log_.mean()) ** 2).sum()
        ),
    }


def metrics_table(models: TrainedModels, preds: pd.DataFrame) -> pd.DataFrame:
    """MAE on the log scale and in EUR for every model, plus the oracle floor.

    Predictions and targets are both re-inflated to nominal euros first, so the
    numbers are directly comparable to the prices actually paid.
    """
    actual_log_ = actual_log(models)
    actual_eur = np.expm1(actual_log_)

    rows = [
        _row(name, to_nominal(models, preds[name].to_numpy()), actual_log_, actual_eur)
        for name in preds.columns
    ]

    floor = noise_floor(models)
    rows.append(
        {
            "model": "oracle_noise_floor",
            "mae_log": floor["mae_log"],
            "mae_eur": floor["mae_eur"],
            "rmse_eur": float("nan"),
            "mape_pct": float("nan"),
            "bias_eur": float("nan"),
            "r2_log": float("nan"),
        }
    )
    return pd.DataFrame(rows)


def feature_importance(models: TrainedModels, top: int = 15) -> pd.DataFrame:
    """Gain-based LightGBM importance, scaled to a percentage of total gain."""
    gain = models.booster.booster_.feature_importance(importance_type="gain")
    frame = pd.DataFrame(
        {"feature": models.split.cols, "gain": gain.astype(float)}
    )
    frame["share_pct"] = 100 * frame["gain"] / frame["gain"].sum()
    return frame.sort_values("gain", ascending=False).head(top).reset_index(drop=True)


def worst_errors(
    models: TrainedModels, preds: pd.DataFrame, model_col: str, n: int = 10
) -> pd.DataFrame:
    """The n largest signed errors in EUR for the chosen model."""
    test = models.split.test
    actual_eur = test["fee_eur"].to_numpy()
    pred_eur = np.expm1(to_nominal(models, preds[model_col].to_numpy()))
    err = pred_eur - actual_eur

    table = test[["player_id", "season", "position", "league", "age"]].copy()
    table["actual_fee_eur"] = actual_eur
    table["predicted_fee_eur"] = pred_eur
    table["error_eur"] = err
    table["abs_error_eur"] = np.abs(err)
    table["error_pct"] = 100 * err / actual_eur
    return (
        table.reindex(columns=["player_id", "season", "age", "position", "league",
                               "actual_fee_eur", "predicted_fee_eur", "error_eur",
                               "abs_error_eur", "error_pct"])
        .iloc[np.argsort(-np.abs(err))[:n]]
        .reset_index(drop=True)
    )


def metrics_markdown(table: pd.DataFrame) -> str:
    out = table.copy()
    out["mae_log"] = out["mae_log"].map(lambda v: f"{v:.4f}")
    out["mae_eur"] = out["mae_eur"].map(lambda v: EUR_FMT.format(v))
    out["rmse_eur"] = out["rmse_eur"].map(lambda v: EUR_FMT.format(v))
    out["mape_pct"] = out["mape_pct"].map(lambda v: f"{v:.1f}")
    out["bias_eur"] = out["bias_eur"].map(lambda v: EUR_FMT.format(v))
    out["r2_log"] = out["r2_log"].map(lambda v: f"{v:.3f}")
    return markdown_table(out)


def importance_markdown(frame: pd.DataFrame) -> str:
    out = frame.copy()
    out["gain"] = out["gain"].map(lambda v: f"{v:,.0f}")
    out["share_pct"] = out["share_pct"].map(lambda v: f"{v:.1f}")
    return markdown_table(out)


def errors_markdown(frame: pd.DataFrame) -> str:
    out = frame.copy()
    for col in ("actual_fee_eur", "predicted_fee_eur", "error_eur", "abs_error_eur"):
        out[col] = out[col].map(lambda v: EUR_FMT.format(v))
    out["error_pct"] = out["error_pct"].map(lambda v: f"{v:+.1f}%")
    return markdown_table(out)