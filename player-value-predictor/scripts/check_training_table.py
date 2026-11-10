"""Phase 1 acceptance check on the REAL table.

The unit tests prove the logic on hand-made cases. This script proves the real
table agrees with an independent recomputation and passes basic sanity rules.
Read-only: it changes nothing.
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config  # noqa: E402

SAMPLE = 300


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else config.DB_PATH
    con = duckdb.connect(str(db_path), read_only=True)
    df = duckdb.sql(f"SELECT * FROM '{config.TRAINING_TABLE}'").df()
    df["valuation_date"] = pd.to_datetime(df["valuation_date"])
    problems = []

    def check(name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
        if not ok:
            problems.append(name)

    print(f"table: {len(df):,} rows, {df.player_id.nunique():,} players\n")

    # ---- 1. basic rules -----------------------------------------------------------------
    check("no duplicate (player, date)", not df.duplicated(["player_id", "valuation_date"]).any())
    required = [c for c in df.columns if c != "height_cm"]
    check("no missing values except height", not df[required].isna().any().any())
    check("every row has at least MIN_MINUTES", bool((df.minutes_365 >= config.MIN_MINUTES).all()))
    check("only top-5 leagues", set(df.league) <= set(config.TOP5))
    last_valuation = pd.Timestamp(con.execute("SELECT MAX(date) FROM player_valuations").fetchone()[0])
    check("dates inside the allowed range",
          df.valuation_date.min() >= pd.Timestamp(config.FIRST_VALUATION_DATE) and df.valuation_date.max() <= last_valuation,
          f"({df.valuation_date.min().date()} to {df.valuation_date.max().date()})")
    check("values are positive", bool((df.value_eur > 0).all()))
    check("ages between 15 and 46", bool(df.age_years.between(15, 46).all()),
          f"(min {df.age_years.min():.1f}, max {df.age_years.max():.1f})")
    check("at most 100 games and 9,000 minutes in a year (more would hint at double counting)",
          bool((df.appearances_365 <= 100).all() and (df.minutes_365 <= 9000).all()),
          f"(max {df.appearances_365.max()} games, {df.minutes_365.max()} minutes)")

    # ---- 2. duplicated appearance rows in the raw data would inflate minutes -----------------
    dup_games = con.execute(
        "SELECT COUNT(*) FROM (SELECT player_id, game_id FROM appearances GROUP BY 1, 2 HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    check("no player appears twice in the same game (raw data)", dup_games == 0, f"({dup_games} duplicates)")

    # ---- 3. independent recomputation with pandas, on a random sample --------------------------
    sample = df.sample(min(SAMPLE, len(df)), random_state=0)
    ids = [int(x) for x in sample.player_id.unique()]
    apps = con.execute(
        "SELECT player_id, date, minutes_played, goals, assists, yellow_cards, red_cards "
        "FROM appearances WHERE player_id IN (SELECT UNNEST(?))", [ids]).df()
    apps["date"] = pd.to_datetime(apps["date"])
    dobs = con.execute("SELECT player_id, date_of_birth FROM players WHERE player_id IN (SELECT UNNEST(?))", [ids]).df()
    dob = dict(zip(dobs.player_id, pd.to_datetime(dobs.date_of_birth)))
    by_player = {pid: g for pid, g in apps.groupby("player_id")}
    mismatches = []
    for r in sample.itertuples():
        g = by_player.get(r.player_id)
        w = g[(g.date >= r.valuation_date - pd.Timedelta(days=config.WINDOW_DAYS)) & (g.date < r.valuation_date)] if g is not None else apps.iloc[0:0]
        mine = (int(w.minutes_played.sum()), len(w), int(w.goals.sum()), int(w.assists.sum()),
                int(w.yellow_cards.sum()), int(w.red_cards.sum()))
        theirs = (r.minutes_365, r.appearances_365, r.goals_365, r.assists_365, r.yellow_365, r.red_365)
        age = (r.valuation_date - dob[r.player_id]).days / 365.25
        if mine != theirs or abs(age - r.age_years) > 0.01:
            mismatches.append((r.player_id, str(r.valuation_date.date()), mine, theirs))
    check(f"independent recomputation matches on {len(sample)} random rows", not mismatches,
          f"({len(mismatches)} mismatches)")
    for m in mismatches[:5]:
        print("      mismatch:", m)

    # ---- 4. what the three periods look like -------------------------------------------------
    d = df.valuation_date
    df["split"] = pd.cut(d, bins=[pd.Timestamp("1900-01-01"), pd.Timestamp(config.TRAIN_END),
                                  pd.Timestamp(config.VALID_END), pd.Timestamp(config.TEST_END)],
                         labels=["train", "validation", "test"])
    summary = df.groupby("split", observed=True).agg(
        rows=("value_eur", "size"),
        median_value_m=("value_eur", lambda v: round(v.median() / 1e6, 2)),
        p90_value_m=("value_eur", lambda v: round(v.quantile(0.9) / 1e6, 2)),
        median_age=("age_years", lambda v: round(v.median(), 1)),
        median_minutes=("minutes_365", "median"),
    )
    print("\nthe three periods:")
    print(summary.to_string())

    print("\nRESULT:", "all checks passed" if not problems else f"{len(problems)} check(s) failed: {problems}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
