"""Shared constants and paths for the transfer-fee pipeline."""
import os
from pathlib import Path

SEED = 20240517

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
RAW_CSV = DATA_DIR / "transfers.csv"
ENV_FILE = ROOT / ".env"

# football-data.org v4 base URL. The token is never committed; it lives in .env.
API_BASE = "https://api.football-data.org/v4"
API_KEY_ENV = "FOOTBALL_DATA_KEY"
API_TIMEOUT = 30
API_RETRIES = 3


def api_key() -> str | None:
    """The football-data.org token from the environment, or from .env.

    .env is gitignored, so the token never reaches the repository. Returns None
    rather than raising so the dashboard can degrade to offline mode instead of
    crashing when the key is absent.
    """
    token = os.environ.get(API_KEY_ENV)
    if token:
        return token.strip()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            name, sep, value = line.partition("=")
            if sep and name.strip() == API_KEY_ENV:
                return value.strip().strip("'\"")
    return None

N_TRANSFERS = 5000
N_PLAYERS = 2600
FIRST_SEASON = 2015
LAST_SEASON = 2025
BASE_SEASON = FIRST_SEASON

TRAIN_LAST_SEASON = 2022
TEST_FIRST_SEASON = 2023

YEARLY_INFLATION = 1.10
FEE_NOISE_SIGMA = 0.33
MARKET_VALUE_NOISE_SIGMA = 0.34

LOG_BASE_FEE = 12.15  # ~190k EUR before any modifiers
FREE_OR_LOAN_SHARE = 0.04

LEAGUES = (
    "Premier League",
    "La Liga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
    "Eredivisie",
    "Primeira Liga",
)

LEAGUE_STRENGTH = {
    "Premier League": 1.00,
    "La Liga": 0.92,
    "Bundesliga": 0.86,
    "Serie A": 0.85,
    "Ligue 1": 0.74,
    "Eredivisie": 0.62,
    "Primeira Liga": 0.55,
}

POSITIONS = ("ST", "LW", "RW", "AM", "CM", "DM", "LB", "RB", "CB", "GK")

POSITION_PREMIUM = {
    "ST": 1.00,
    "LW": 0.75,
    "RW": 0.75,
    "AM": 0.68,
    "CM": 0.30,
    "DM": 0.12,
    "LB": 0.05,
    "RB": 0.05,
    "CB": -0.05,
    "GK": -0.10,
}

# Columns that describe what happened *after* the transfer decision, are pure
# identifiers, or are the generator's ground-truth answer key. They must never
# become model features.
NON_FEATURE_COLUMNS = (
    "player_id",
    "fee_eur",
    "origin_league",
    "is_free_or_loan",
    "log_fee_signal",
)