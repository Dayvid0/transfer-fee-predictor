"""A tiny database with the REAL table definitions, filled with hand-made cases.

Every row exists to test one rule of build_dataset.py. If you add a rule there,
add a case here. The columns match the real dataset exactly: an earlier test
database with fewer columns let a real bug slip through.
"""
from datetime import date, timedelta

PLAYERS_DDL = """CREATE TABLE players (
    player_id INTEGER, first_name VARCHAR, last_name VARCHAR, name VARCHAR, last_season VARCHAR,
    current_club_id VARCHAR, player_code VARCHAR, country_of_birth VARCHAR, city_of_birth VARCHAR,
    country_of_citizenship VARCHAR, date_of_birth TIMESTAMP, sub_position VARCHAR, position VARCHAR,
    foot VARCHAR, height_in_cm INTEGER, contract_expiration_date TIMESTAMP, agent_name VARCHAR,
    image_url VARCHAR, international_caps INTEGER, international_goals INTEGER,
    current_national_team_id VARCHAR, url VARCHAR, current_club_domestic_competition_id VARCHAR,
    current_club_name VARCHAR, market_value_in_eur INTEGER, highest_market_value_in_eur INTEGER)"""

VALUATIONS_DDL = """CREATE TABLE player_valuations (
    player_id INTEGER, date DATE, market_value_in_eur INTEGER, current_club_name VARCHAR,
    current_club_id INTEGER, player_club_domestic_competition_id VARCHAR)"""

APPEARANCES_DDL = """CREATE TABLE appearances (
    appearance_id VARCHAR, game_id INTEGER, player_id INTEGER, player_club_id INTEGER,
    player_current_club_id VARCHAR, date DATE, player_name VARCHAR, competition_id VARCHAR,
    yellow_cards INTEGER, red_cards INTEGER, goals INTEGER, assists INTEGER, minutes_played INTEGER)"""

# Columns of `players` that describe the player TODAY. They must never reach the features.
SNAPSHOT_UPDATE = """UPDATE players SET
    market_value_in_eur = 999999999, highest_market_value_in_eur = 999999999,
    contract_expiration_date = TIMESTAMP '2035-01-01', current_club_id = '777',
    current_club_name = 'Poison FC', current_club_domestic_competition_id = 'ZZ9',
    last_season = '2099', international_caps = 500, international_goals = 500,
    current_national_team_id = '1'"""


def make_fixture(con):
    con.execute(PLAYERS_DDL)
    con.execute(VALUATIONS_DDL)
    con.execute(APPEARANCES_DDL)

    counter = {"n": 0}

    def player(pid, dob, position, height):
        con.execute(
            "INSERT INTO players (player_id, name, date_of_birth, position, height_in_cm, "
            "market_value_in_eur, highest_market_value_in_eur, current_club_name) "
            "VALUES (?, ?, ?, ?, ?, 1000, 1000, 'Real FC')",
            [pid, f"Player {pid}", dob, position, height])

    def valuation(pid, day, value, league):
        con.execute("INSERT INTO player_valuations VALUES (?, ?, ?, 'Some Club', 1, ?)",
                    [pid, day, value, league])

    def appearance(pid, day, comp="GB1", minutes=90, goals=0, assists=0, yellow=0, red=0):
        counter["n"] += 1
        con.execute(
            "INSERT INTO appearances (appearance_id, game_id, player_id, player_club_id, date, "
            "competition_id, yellow_cards, red_cards, goals, assists, minutes_played) "
            "VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)",
            [f"a{counter['n']}", counter["n"], pid, day, comp, yellow, red, goals, assists, minutes])

    def five_games(pid, start, comp):
        """Five 90-minute games, one a month: exactly 450 minutes."""
        for i in range(5):
            appearance(pid, date.fromisoformat(start) + timedelta(days=31 * i), comp)

    # ---- Player 1: the window-boundary player -------------------------------------------
    player(1, "1995-06-15", "Attack", 180)
    valuation(1, "2015-01-01", 1_000_000, "GB1")    # KEPT: exactly 450 minutes in its window
    valuation(1, "2016-01-01", 3_000_000, "GB1")    # DROPPED: only 180 minutes in its window
    # Inside the window of the 2015-01-01 valuation: [2014-01-01, 2015-01-01)
    appearance(1, "2014-01-01", "GB1", goals=1, yellow=1)    # exactly 365 days before: INCLUDED
    appearance(1, "2014-02-01", "GB1", assists=1)
    appearance(1, "2014-03-01", "CL", goals=2)               # a different competition still counts
    appearance(1, "2014-04-01", "GB1")
    appearance(1, "2014-05-01", "GBFA", goals=1)
    # "Poison" games: huge numbers, so any leak would be obvious
    appearance(1, "2013-12-31", "GB1", goals=10, assists=10, yellow=5, red=1)   # 366 days before: EXCLUDED
    appearance(1, "2015-01-01", "GB1", goals=10, assists=10, yellow=5, red=1)   # same day: EXCLUDED
    appearance(1, "2015-01-02", "GB1", goals=10, assists=10, yellow=5, red=1)   # the future: EXCLUDED

    # ---- Player 2: changes league; height unknown ---------------------------------------
    player(2, "1999-03-01", "Midfield", None)
    valuation(2, "2016-01-01", 2_000_000, "ES1")
    five_games(2, "2015-02-01", "ES1")
    valuation(2, "2018-01-01", 4_000_000, "IT1")
    five_games(2, "2017-02-01", "IT1")

    # ---- Players that must be removed, one reason each ----------------------------------
    player(3, None, "Attack", 175)                  # no birth date
    valuation(3, "2017-01-01", 1_000_000, "GB1")
    five_games(3, "2016-02-01", "GB1")

    player(4, "1996-01-01", "Missing", 175)         # position we do not map
    valuation(4, "2017-01-01", 1_000_000, "GB1")
    five_games(4, "2016-02-01", "GB1")

    player(6, "1996-01-01", "Defender", 185)        # only 360 minutes
    valuation(6, "2017-01-01", 1_000_000, "GB1")
    for i in range(4):
        appearance(6, date.fromisoformat("2016-02-01") + timedelta(days=31 * i))

    player(7, "1996-01-01", "Attack", 185)          # not a top-five league
    valuation(7, "2017-01-01", 1_000_000, "NL1")
    five_games(7, "2016-02-01", "NL1")

    player(8, "1996-01-01", "Attack", 185)          # value of zero
    valuation(8, "2017-01-01", 0, "GB1")
    five_games(8, "2016-02-01", "GB1")

    player(9, "1990-01-01", "Attack", 185)          # before the first allowed valuation date
    valuation(9, "2012-01-01", 1_000_000, "GB1")

    # ---- Player 5: a goalkeeper with plenty of minutes ----------------------------------
    player(5, "1990-01-01", "Goalkeeper", 190)
    valuation(5, "2020-07-01", 5_000_000, "L1")
    for i in range(10):
        appearance(5, date.fromisoformat("2019-08-01") + timedelta(days=30 * i), "L1")
