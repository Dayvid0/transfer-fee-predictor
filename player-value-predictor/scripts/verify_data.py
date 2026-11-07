"""Phase 0b: read-only checks that answer the blueprint's open data questions."""
import os
import sys

import duckdb

DEFAULT = "/mnt/data/datasets/transfermarkt/transfermarkt-datasets.duckdb"
TOP5 = ["GB1", "ES1", "IT1", "L1", "FR1"]
TOP5_SQL = ", ".join(f"'{c}'" for c in TOP5)
SINCE = "2013-07-01"   # appearances start in 2012, so a full 365-day window exists from mid-2013

WINDOW_SQL = """
WITH v AS (
    SELECT DISTINCT player_id, date AS vdate, market_value_in_eur AS value
    FROM player_valuations
    WHERE date >= DATE '{since}'
      AND player_club_domestic_competition_id IN ({top5})
      AND market_value_in_eur > 0
)
SELECT v.player_id, v.vdate, v.value,
       COALESCE(SUM(a.minutes_played), 0) AS minutes,
       COUNT(a.appearance_id) AS apps
FROM v
LEFT JOIN appearances a
  ON a.player_id = v.player_id
 AND a.date >= v.vdate - INTERVAL 365 DAY
 AND a.date <  v.vdate
GROUP BY v.player_id, v.vdate, v.value
"""


def window_table(con, since=SINCE, top5_sql=TOP5_SQL):
    """One row per valuation: minutes and appearances in the 365 days BEFORE it."""
    return con.execute(WINDOW_SQL.format(since=since, top5=top5_sql)).df()


def show(con, title, sql):
    print(f"\n=== {title} ===")
    df = con.execute(sql).df()
    print(df.to_string(index=False) if len(df) else "(no rows)")
    return df


def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TM_DB", DEFAULT)
    if not os.path.exists(db_path):
        sys.exit(f"Database file not found: {db_path}")
    con = duckdb.connect(db_path, read_only=True)

    show(con, "1. version table", "SELECT * FROM version")

    show(con, "2. date ranges",
         "SELECT 'player_valuations' AS tbl, MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS n FROM player_valuations "
         "UNION ALL SELECT 'appearances', MIN(date), MAX(date), COUNT(*) FROM appearances")

    found = show(con, "3a. the five league codes we expect",
                 f"SELECT competition_id, name, country_name, type, sub_type FROM competitions "
                 f"WHERE competition_id IN ({TOP5_SQL}) ORDER BY competition_id")
    missing = sorted(set(TOP5) - set(found["competition_id"]))
    print("MISSING CODES:", missing if missing else "none")
    show(con, "3b. competition types in the dataset",
         "SELECT type, sub_type, COUNT(*) AS n FROM competitions GROUP BY 1, 2 ORDER BY n DESC")

    print("\n=== 4. is player_club_domestic_competition_id time-stamped, or a snapshot? ===")
    show(con, "4a. players who changed club: does the competition column change too?",
         "WITH p AS (SELECT player_id, COUNT(DISTINCT current_club_id) AS n_clubs, "
         "COUNT(DISTINCT player_club_domestic_competition_id) AS n_comps FROM player_valuations GROUP BY 1) "
         "SELECT n_clubs > 1 AS changed_club, COUNT(*) AS players, "
         "SUM(CASE WHEN n_comps > 1 THEN 1 ELSE 0 END) AS comp_also_changes FROM p GROUP BY 1 ORDER BY 1")
    show(con, "4b. agreement with the clubs table",
         "SELECT COUNT(*) AS rows_with_known_club, "
         "SUM(CASE WHEN c.domestic_competition_id = v.player_club_domestic_competition_id THEN 1 ELSE 0 END) AS agree "
         "FROM player_valuations v JOIN clubs c ON TRY_CAST(c.club_id AS INTEGER) = v.current_club_id")
    show(con, "4c. one well-travelled player: rows where the club changes",
         "WITH m AS (SELECT player_id FROM player_valuations GROUP BY 1 "
         "HAVING COUNT(DISTINCT current_club_id) >= 3 ORDER BY COUNT(*) DESC LIMIT 1), "
         "x AS (SELECT v.*, LAG(v.current_club_id) OVER (PARTITION BY v.player_id ORDER BY v.date) AS prev_club "
         "FROM player_valuations v JOIN m USING (player_id)) "
         "SELECT p.name, x.date, x.current_club_name, x.current_club_id, x.player_club_domestic_competition_id AS comp "
         "FROM x JOIN players p USING (player_id) WHERE x.prev_club IS NULL OR x.prev_club <> x.current_club_id "
         "ORDER BY x.date LIMIT 12")

    show(con, "5. valuation rows per year (top-5 by the valuation column; provisional until 4 is settled)",
         f"SELECT year(date) AS yr, COUNT(*) AS valuations, COUNT(DISTINCT player_id) AS players "
         f"FROM player_valuations WHERE player_club_domestic_competition_id IN ({TOP5_SQL}) AND date >= DATE '{SINCE}' "
         f"GROUP BY 1 ORDER BY 1")

    print(f"\n=== 6. does a 365-day window have data? (valuations since {SINCE}, provisional top-5) ===")
    df = window_table(con)
    total = len(df)
    print(f"valuation rows: {total:,}   distinct players: {df.player_id.nunique():,}")
    for thr in [1, 450, 900, 1800]:
        n = int((df.minutes >= thr).sum())
        print(f"  minutes in prior 365 days >= {thr:>5}: {n:>8,} rows ({n / total:6.1%})")
    print("minutes quantiles (10/25/50/75/90%):",
          [int(x) for x in df.minutes.quantile([.1, .25, .5, .75, .9])])

    show(con, "7. where do those minutes come from? (clubs in top-5 leagues, season 2024/25)",
         f"SELECT comp.type, comp.sub_type, SUM(a.minutes_played) AS minutes, "
         f"ROUND(100.0 * SUM(a.minutes_played) / SUM(SUM(a.minutes_played)) OVER (), 1) AS pct "
         f"FROM appearances a JOIN clubs c ON TRY_CAST(c.club_id AS INTEGER) = a.player_club_id "
         f"JOIN competitions comp ON comp.competition_id = a.competition_id "
         f"WHERE c.domestic_competition_id IN ({TOP5_SQL}) AND a.date >= DATE '2024-07-01' AND a.date < DATE '2025-07-01' "
         f"GROUP BY 1, 2 ORDER BY minutes DESC")

    show(con, "8. data quality",
         "SELECT "
         "(SELECT COUNT(*) FROM player_valuations WHERE market_value_in_eur IS NULL OR market_value_in_eur <= 0) AS bad_values, "
         "(SELECT COUNT(*) FROM (SELECT player_id, date FROM player_valuations GROUP BY 1, 2 HAVING COUNT(*) > 1)) AS dup_player_dates, "
         "(SELECT COUNT(*) FROM appearances WHERE minutes_played IS NULL) AS null_minutes, "
         "(SELECT COUNT(*) FROM appearances WHERE minutes_played < 0 OR minutes_played > 150) AS odd_minutes, "
         "(SELECT COUNT(*) FROM players WHERE date_of_birth IS NULL) AS no_birthdate")


if __name__ == "__main__":
    main()
