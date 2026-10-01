"""Real player market values: download, cache and clean.

Source is a public Kaggle dataset (``football-players-transfer-value-prediction``)
that needs no login: the public download endpoint serves the zip anonymously.
It is a **single June-2024 snapshot** of 3,060 rows, so there is no time
axis to split on. That is the single most important fact about this data and it
is why ``real_model`` offers a leave-clubs-out split next to the random one.

The snapshot carries each player's estimated **market value** (a club's asking
price), not a fee any club actually paid. Target columns are held apart from
feature columns by ``real_model.TARGET_COLUMNS``.
"""
from __future__ import annotations

import io
import re
import unicodedata
import zipfile

import pandas as pd
import requests

from src import config as C

DATASET = "dhruvpjariwala/football-players-transfer-value-prediction"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
RAW_CACHE = C.DATA_DIR / "real_players_raw.zip"
CLEAN_CACHE = C.DATA_DIR / "real_players.csv"

# The CSV was published in June 2024; every contract and age figure is as of then.
SNAPSHOT_YEAR = 2024
TIMEOUT = 60

EUR_MULTIPLIER = {"K": 1e3, "M": 1e6, "B": 1e9}
CONTRACT_RE = re.compile(r"((?:19|20)\d{2})\s*~\s*((?:19|20)\d{2})")
NUMBER_RE = re.compile(r"([\d.]+)")
# Ratings use EA-FC in-form notation: '76+2' is a base of 76 lifted two points,
# '85-1' a base of 85 marked down one. Left as text, sklearn cannot use them.
RATING_RE = re.compile(r"^(\d+)([+-]\d+)?$")

# Columns that stay textual: identity, and the fields parsed by dedicated helpers.
TEXT_COLUMNS = {
    "name", "search_key", "club", "foot", "position", "Best position", "foot",
    "Team & Contract", "Height", "Weight", "Value", "Wage", "Release clause",
}

# Every row in the source appends the player's position codes to the name, as in
# "Mikel Merino CM CAM". Left in place, name search matches nothing useful.
POSITION_CODES = {
    "AM", "CAM", "CB", "CDM", "CF", "CM", "DM", "GK", "LB", "LM", "LW", "LWB",
    "RB", "RM", "RW", "RWB", "ST", "WB",
}


def clean_name(text: object) -> str:
    """Strip trailing position codes: 'B. Sesko ST' -> 'B. Sesko'."""
    parts = str(text).split()
    while len(parts) > 1 and parts[-1].upper() in POSITION_CODES:
        parts.pop()
    return " ".join(parts)


def fold(text: object) -> str:
    """Accent-stripped lowercase, so 'Mbappe' finds 'K. Mbappe'.

    The source spells names with diacritics, but nobody types them, so the
    search box has to match the unaccented version.
    """
    decomposed = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def download(timeout: int = TIMEOUT) -> bytes:
    """Fetch the dataset zip. Raises on any non-200 so a stale cache is visible."""
    response = requests.get(DOWNLOAD_URL, timeout=timeout)
    response.raise_for_status()
    return response.content


def _zip_to_csv(payload: bytes) -> pd.DataFrame:
    """The archive holds exactly one CSV; read the first one found."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError(f"no CSV in archive: {archive.namelist()}")
        with archive.open(names[0]) as handle:
            return pd.read_csv(handle)


def parse_eur(text: object) -> float:
    """'€28.5M' -> 28_500_000. Returns NaN for blanks and unrecognised text."""
    match = re.match(r"\s*€\s*([\d.]+)\s*([KMB]?)\s*$", str(text))
    if not match:
        return float("nan")
    return float(match.group(1)) * EUR_MULTIPLIER.get(match.group(2), 1.0)


def parse_cm(text: object) -> float:
    """'170cm / 5\'7"' -> 170.0"""
    match = NUMBER_RE.search(str(text))
    return float(match.group(1)) if match else float("nan")


def parse_kg(text: object) -> float:
    """'71kg / 157lbs' -> 71.0"""
    match = NUMBER_RE.search(str(text))
    return float(match.group(1)) if match else float("nan")


def parse_rating(text: object) -> float:
    """EA-FC in-form notation to an effective rating: '76+2' -> 78.0."""
    match = RATING_RE.match(str(text).strip())
    if not match:
        return float("nan")
    return float(match.group(1)) + float(match.group(2) or 0)


def _contract(text: object) -> dict:
    """'Al Ettifaq\\n2022 ~ 2024' -> joined, expiry, on_loan.

    Loans carry an end date instead of a range ('Jun 30, 2024 On loan') and so
    have no known length; those get NaN and are flagged rather than guessed.
    """
    head, _, tail = str(text).partition("\n")
    match = CONTRACT_RE.search(tail)
    return {
        "club": head.strip(),
        "contract_joined": float(match.group(1)) if match else float("nan"),
        "contract_expiry": float(match.group(2)) if match else float("nan"),
        "on_loan": "on loan" in str(text).lower(),
    }


def build(raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Kaggle frame into a modelling-ready table."""
    d = raw.loc[:, [c for c in raw.columns if c != "Unnamed: 64"]].copy()

    # Coerce the EA-FC rating columns first, while `d` still holds only the
    # source columns. Doing it later would also hit the derived columns below
    # and wipe them to NaN.
    for column in [c for c in d.columns if c not in TEXT_COLUMNS]:
        d[column] = d[column].map(parse_rating)

    contracts = pd.DataFrame([_contract(v) for v in d["Team & Contract"]], index=d.index)
    d = pd.concat([d, contracts], axis=1)
    d["contract_years_left"] = d["contract_expiry"] - SNAPSHOT_YEAR

    d["value_eur"] = d["Value"].map(parse_eur)
    d["wage_eur"] = d["Wage"].map(parse_eur)
    d["release_clause_eur"] = d["Release clause"].map(parse_eur)
    d["height_cm"] = d["Height"].map(parse_cm)
    d["weight_kg"] = d["Weight"].map(parse_kg)
    d["age"] = d["Age"].astype(float)
    d["player_id"] = d["ID"].astype("int64")
    d["name"] = d["name"].map(clean_name)
    d["search_key"] = d["name"].map(fold)
    d["position"] = d["Best position"].astype(str).str.strip()
    d["club"] = d["club"].astype(str).str.strip()
    d["foot"] = d["foot"].astype(str).str.strip()

    d = d.rename(
        columns={
            "Overall rating": "overall",
            "Potential": "potential",
            "Best overall": "best_overall",
            "Growth": "growth",
            "International reputation": "reputation",
        }
    )
    return d.reset_index(drop=True)


def load(force: bool = False) -> pd.DataFrame:
    """The clean frame, downloading once and caching the parsed result on disk."""
    if CLEAN_CACHE.exists() and not force:
        return pd.read_csv(CLEAN_CACHE)

    if RAW_CACHE.exists() and not force:
        payload = RAW_CACHE.read_bytes()
    else:
        C.DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = download()
        RAW_CACHE.write_bytes(payload)

    frame = build(_zip_to_csv(payload))
    frame.to_csv(CLEAN_CACHE, index=False)
    return frame


if __name__ == "__main__":
    data = load(force=True)
    print(f"{len(data):,} real players -> {CLEAN_CACHE}")
    print(data[["name", "age", "position", "club", "value_eur"]].head(10).to_string(index=False))
