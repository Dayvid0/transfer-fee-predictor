"""Small valuation rows for hand-computed Phase 2 checks."""
import pandas as pd


def valuation(player_id=1, day="2024-01-01", value=100, league="GB1", position="ATT", **changes):
    row = {
        "player_id": player_id, "valuation_date": pd.Timestamp(day), "value_eur": value,
        "league": league, "position_group": position, "age_years": 24.0,
        "height_cm": 180.0, "minutes_365": 900, "appearances_365": 12,
        "goals_365": 10, "assists_365": 5, "yellow_365": 2, "red_365": 1,
    }
    row.update(changes)
    return row
