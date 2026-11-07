"""Phase 0c: follow-up checks raised by the Phase 0b output."""
import os
import sys

import duckdb
import pandas as pd

from verify_data import DEFAULT, window_table


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TM_DB", DEFAULT)
    if not os.path.exists(db_path):
        sys.exit(f"Database file not found: {db_path}")
    con = duckdb.connect(db_path, read_only=True)

    print("\n=== A. do national-team matches appear in appearances? (season 2024/25, all clubs) ===")
    a = con.execute("""
        SELECT COALESCE(comp.type, '(competition not in table)') AS type,
               COUNT(*) AS rows, SUM(a.minutes_played) AS minutes
        FROM appearances a
        LEFT JOIN competitions comp ON comp.competition_id = a.competition_id
        WHERE a.date >= DATE '2024-07-01' AND a.date < DATE '2025-07-01'
        GROUP BY 1 ORDER BY minutes DESC
    """).df()
    print(a.to_string(index=False))

    df = window_table(con)
    df["year"] = pd.to_datetime(df["vdate"]).dt.year
    df["played"] = df["minutes"] >= 450

    print("\n=== B. per valuation year (provisional top-5): does the 2024 drop survive a minutes filter? ===")
    base = df.groupby("year").agg(
        rows=("value", "size"),
        pct_zero_minutes=("minutes", lambda m: round(100 * float((m == 0).mean()), 1)),
        rows_450=("played", "sum"),
    )
    played = df[df["played"]].groupby("year").agg(
        players_450=("player_id", "nunique"),
        median_value_m=("value", lambda v: round(float(v.median()) / 1e6, 2)),
        p90_value_m=("value", lambda v: round(float(v.quantile(0.9)) / 1e6, 2)),
    )
    print(base.join(played).to_string())

    print("\n=== C. the target: listed value for rows with >= 450 minutes (millions of EUR) ===")
    v = df.loc[df["played"], "value"] / 1e6
    print(v.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).round(2).to_string())
    print(f"max: {v.max():.1f}   share above 50m: {100 * float((v > 50).mean()):.2f}%")


if __name__ == "__main__":
    main()
