"""Project settings in one place. Change a number here, not inside the code."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DB_PATH = os.environ.get("TM_DB", "/mnt/data/datasets/transfermarkt/transfermarkt-datasets.duckdb")
TRAINING_TABLE = ROOT / "data" / "training_table.parquet"

TOP5 = ["GB1", "ES1", "IT1", "L1", "FR1"]    # Premier League, LaLiga, Serie A, Bundesliga, Ligue 1
WINDOW_DAYS = 365                             # stats come from this many days BEFORE the valuation
MIN_MINUTES = 450                             # about five full matches
FIRST_VALUATION_DATE = "2013-07-01"           # appearances start in 2012, so a full window exists from here

# Split by valuation date (train / validation / test)
TRAIN_END = "2024-06-30"
VALID_END = "2025-06-30"
TEST_END = "2026-06-12"

# players.position -> our four groups (anything else is dropped and counted)
POSITION_GROUPS = {"goalkeeper": "GK", "defender": "DEF", "midfield": "MID", "attack": "ATT"}
