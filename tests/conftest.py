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
def loaded(csv_path: str) -> tuple[pd.DataFrame, float]:
    """Deflated feature frame plus the price index estimated from it."""
    return F.load_features(csv_path)


@pytest.fixture(scope="session")
def keyed(loaded) -> pd.DataFrame:
    return loaded[0]


@pytest.fixture(scope="session")
def price_index(loaded) -> float:
    return loaded[1]


@pytest.fixture(scope="session")
def models(keyed, price_index):
    return fit_models(keyed, price_index)


@pytest.fixture(scope="session")
def preds(models) -> pd.DataFrame:
    return E.predictions(models)