"""Scoring: one metrics table, LightGBM importances, worst errors."""
from __future__ import annotations

import numpy as np
import pandas as pd

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
    """Log-scale predictions for all three models on the test block."""
    x_te = models.split.test[models.split.cols]
    return pd.DataFrame(
        {
            "A_market_value": np.asarray(models.baseline_log),
            "B_linear": models.linear.predict(x_te),
            "C_lightgbm": models.booster.predict(x_te),
        },
        index=models.split.test.index,
    )


def metrics_table(models: TrainedModels, preds: pd.DataFrame) -> pd.DataFrame:
    """MAE on the log scale and in EUR for every model."""
    test = models.split.test
    actual_log = test["target"].to_numpy()
    actual_eur = np.expm1(actual_log)

    rows = []
    for name in preds.columns:
        p_log = preds[name].to_numpy()
        p_eur = np.expm1(p_log)
        err_eur = p_eur - actual_eur
        rows.append(
            {
                "model": name,
                "mae_log": np.abs(p_log - actual_log).mean(),
                "mae_eur": np.abs(err_eur).mean(),
                "rmse_eur": float(np.sqrt((err_eur**2).mean())),
                "mape_pct": float((np.abs(err_eur) / actual_eur).mean() * 100),
                "bias_eur": float(err_eur.mean()),
                "r2_log": float(
                    1 - ((p_log - actual_log) ** 2).sum()
                    / ((actual_log - actual_log.mean()) ** 2).sum()
                ),
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
    actual_eur = np.expm1(test["target"].to_numpy())
    pred_eur = np.expm1(preds[model_col].to_numpy())
    err = pred_eur - actual_eur

    table = test[["player_id", "season", "position", "league"]].copy()
    table.insert(0, "age", test["age"].to_numpy() if "age" in test else np.nan)
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