"""Shared fixtures. Generation + fit is fast (~2s), so do it once per session."""
from __future__ import annotations

import pandas as pd
import pytest

from src import evaluate as E
from src import features as F
from src import make_data as MD
from src.train import fit_models


@pytest.fixture(scope="session")
def raw() -> pd.DataFrame:
    return MD.make_transfers()


@pytest.fixture(scope="session")
def csv_path(tmp_path_factory, raw: pd.DataFrame) -> str:
    path = tmp_path_factory.mktemp("data") / "transfers.csv"
    raw.to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="session")
def keyed(csv_path: str) -> pd.DataFrame:
    frame, _ = F.load_features(csv_path)
    return frame


@pytest.fixture(scope="session")
def models(keyed: pd.DataFrame):
    return fit_models(keyed)


@pytest.fixture(scope="session")
def preds(models) -> pd.DataFrame:
    return E.predictions(models)