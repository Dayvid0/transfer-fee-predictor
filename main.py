"""End-to-end run: generate data, build features, train, score, report.

    .venv/bin/python main.py
"""
from __future__ import annotations

import time
from datetime import datetime

import pandas as pd

from src import config as C
from src import evaluate as E
from src import features as F
from src import make_data as MD
from src import report as R
from src.train import fit_models


def run(regenerate: bool = False) -> str:
    started = time.time()
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if regenerate or not C.RAW_CSV.exists():
        MD.write_transfers()
        data_note = "regenerated"
    else:
        data_note = "reused existing CSV"
    raw = pd.read_csv(C.RAW_CSV)

    keyed, index = F.load_features()
    models = fit_models(keyed, index)
    preds = E.predictions(models)

    text = R.build_report(
        raw=raw,
        models=models,
        preds=preds,
        table=E.metrics_table(models, preds),
        importance=E.feature_importance(models),
        worst=E.worst_errors(models, preds, R.BEST_MODEL, n=R.N_WORST),
        index=index,
        data_note=data_note,
        runtime_s=time.time() - started,
        generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    out_path = C.REPORTS_DIR / "results.md"
    out_path.write_text(text)
    print(text)
    print(f"\nwrote {out_path}")
    return str(out_path)


if __name__ == "__main__":
    run()