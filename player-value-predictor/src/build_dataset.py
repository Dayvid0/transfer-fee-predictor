"""Phase 1: build the training table.

One row per valuation of a top-five-league player. Every stat comes from the
365 days BEFORE the valuation date, so the table never contains information
from the future.
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd

import config

COLUMNS = [
    "player_id", "valuation_date", "value_eur", "league", "position_group", "height_cm",
    "age_years", "minutes_365", "appearances_365", "goals_365", "assists_365",
    "yellow_365", "red_365",
]

BASE_SQL = """
WITH v AS (
    SELECT DISTINCT
           player_id,
           date AS valuation_date,
           market_value_in_eur AS value_eur,
           player_club_domestic_competition_id AS league
    FROM player_valuations
    WHERE date >= DATE '{since}'
      AND player_club_domestic_competition_id IN ({top5})
      AND market_value_in_eur > 0
),
agg AS (
    SELECT v.player_id, v.valuation_date,
           CAST(COALESCE(SUM(a.minutes_played), 0) AS INTEGER) AS minutes_365,
           COUNT(a.appearance_id)                              AS appearances_365,
           CAST(COALESCE(SUM(a.goals), 0) AS INTEGER)          AS goals_365,
           CAST(COALESCE(SUM(a.assists), 0) AS INTEGER)        AS assists_365,
           CAST(COALESCE(SUM(a.yellow_cards), 0) AS INTEGER)   AS yellow_365,
           CAST(COALESCE(SUM(a.red_cards), 0) AS INTEGER)      AS red_365
    FROM v
    LEFT JOIN appearances a
      ON a.player_id = v.player_id
     AND a.date >= v.valuation_date - INTERVAL {window} DAY   -- the window starts 365 days back ...
     AND a.date <  v.valuation_date                           -- ... and stops BEFORE the valuation day
    GROUP BY v.player_id, v.valuation_date
)
SELECT v.player_id, v.valuation_date, v.value_eur, v.league,
       p.position,
       p.height_in_cm AS height_cm,
       date_diff('day', CAST(p.date_of_birth AS DATE), v.valuation_date) / 365.25 AS age_years,
       agg.minutes_365, agg.appearances_365, agg.goals_365, agg.assists_365,
       agg.yellow_365, agg.red_365
FROM v
JOIN agg ON agg.player_id = v.player_id AND agg.valuation_date = v.valuation_date
JOIN players p ON p.player_id = v.player_id
"""


def build_training_table(con, min_minutes=config.MIN_MINUTES):
    """Return (table, steps). `steps` is the filter waterfall: (rule, rows left)."""
    sql = BASE_SQL.format(
        since=config.FIRST_VALUATION_DATE,
        top5=", ".join(f"'{c}'" for c in config.TOP5),
        window=config.WINDOW_DAYS,
    )
    df = con.execute(sql).df()
    steps = [(f"top-5 league, positive value, from {config.FIRST_VALUATION_DATE}", len(df))]

    df = df[df["minutes_365"] >= min_minutes]
    steps.append((f"at least {min_minutes} minutes in the prior {config.WINDOW_DAYS} days", len(df)))

    df = df[df["age_years"].notna()]
    steps.append(("known birth date", len(df)))

    df = df.copy()
    df["position_group"] = df["position"].str.lower().map(config.POSITION_GROUPS)
    df = df[df["position_group"].notna()]
    steps.append(("known position group", len(df)))

    df = df[COLUMNS].sort_values(["valuation_date", "player_id"]).reset_index(drop=True)
    return df, steps


def write_parquet(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mem = duckdb.connect(":memory:")     # a scratch database, only used to write the file
    mem.register("t", df)
    mem.execute(f"COPY t TO '{path}' (FORMAT PARQUET)")


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else config.DB_PATH
    if not Path(db_path).exists():
        sys.exit(f"Database file not found: {db_path}")
    con = duckdb.connect(str(db_path), read_only=True)

    print("position values in players (check that our four groups cover the real ones):")
    print(con.execute("SELECT position, COUNT(*) AS n FROM players GROUP BY 1 ORDER BY n DESC").df().to_string(index=False))

    df, steps = build_training_table(con)
    print("\nfilter waterfall:")
    for rule, n in steps:
        print(f"  {n:>9,}  {rule}")

    print(f"\nrows: {len(df):,}   players: {df.player_id.nunique():,}")
    print(f"age: {df.age_years.min():.1f} to {df.age_years.max():.1f}   missing height: {100 * df.height_cm.isna().mean():.1f}%")
    print("by league:  ", df.league.value_counts().to_dict())
    print("by position:", df.position_group.value_counts().to_dict())

    d = pd.to_datetime(df.valuation_date)
    train = df[d <= config.TRAIN_END]
    valid = df[(d > config.TRAIN_END) & (d <= config.VALID_END)]
    test = df[(d > config.VALID_END) & (d <= config.TEST_END)]
    print(f"\nsplit rows: train {len(train):,}   validation {len(valid):,}   test {len(test):,}")
    known = test.player_id.isin(set(train.player_id)).mean() if len(test) else float("nan")
    print(f"share of test rows whose player also appears in train: {100 * known:.1f}%")

    write_parquet(df, config.TRAINING_TABLE)
    print(f"\nwritten: {config.TRAINING_TABLE}")


if __name__ == "__main__":
    main()
