"""football-data.org v4 client: league trophy cabinets and live tables.

The token lives in .env (gitignored), read through ``config.api_key``. Responses
are cached to disk and paced off the server's own quota headers, because the
free plan allows 10 requests a minute and a dashboard reruns constantly.

What this source does **not** have: transfer fees and market values. It is a
fixtures, standings and winners API. Player valuations come from ``real_data``.
"""
from __future__ import annotations

import json
import re
import time

import pandas as pd
import requests

from src import config as C

CACHE_DIR = C.DATA_DIR / "api"
BACKOFF_SECONDS = 2.0

# Competitions this plan can actually read. Verified against the API: PP1 and P1
# are 404 (Primeira Liga is not served) and the second tiers return 403.
COMPETITIONS = {
    "PL": "Premier League",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "FL1": "Ligue 1",
    "PD": "La Liga",
    "DED": "Eredivisie",
    "CL": "Champions League",
    "ELC": "Championship",
}


class ApiUnavailable(RuntimeError):
    """No token, or the token is not entitled to the resource."""


# The free plan is 10 requests/minute, and the API reports its own budget on
# every response: `x-requests-available-minute` is what is left, and
# `X-RequestCounter-Reset` is how many seconds until the window reopens. Reading
# those beats sleeping blind, because a page can queue up several calls (a
# cabinet, a table, eight club lists) and would otherwise trip a 429 at the end.
_requests_left = 10
_resets_at = 0.0


def available() -> bool:
    return C.api_key() is not None


def _as_float(value: str | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _note_quota(response: requests.Response) -> None:
    """Track the server's view of the per-minute budget."""
    global _requests_left, _resets_at
    left = _as_float(response.headers.get("x-requests-available-minute"))
    if left is not None:
        _requests_left = int(left)
    reset = _as_float(response.headers.get("X-RequestCounter-Reset"))
    if reset is not None:
        _resets_at = time.monotonic() + reset


def _throttle() -> None:
    """Block until the quota window reopens when the budget is spent."""
    global _requests_left
    if _requests_left > 0:
        return
    wait = _resets_at - time.monotonic()
    if wait > 0:
        time.sleep(wait + 0.5)
    _requests_left = 1  # one slot is enough to get headers that correct us


def get(path: str, params: dict | None = None, retries: int = C.API_RETRIES) -> dict:
    """GET a v4 resource with a token, quota pacing, retries and a disk cache.

    Raises ``ApiUnavailable`` when no token is configured, so callers can fall
    back to offline mode instead of crashing the dashboard.
    """
    global _requests_left

    token = C.api_key()
    if not token:
        raise ApiUnavailable("no FOOTBALL_DATA_KEY in .env or the environment")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / (path.strip("/").replace("/", "_") + ".json")
    if cache.exists():
        return json.loads(cache.read_text())

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            _throttle()
            response = requests.get(
                f"{C.API_BASE}/{path.lstrip('/')}",
                headers={"X-Auth-Token": token},
                params=params,
                timeout=C.API_TIMEOUT,
            )
            _note_quota(response)
            if response.status_code == 403:
                raise ApiUnavailable(f"403 for /{path}: check your subscription")
            if response.status_code == 429:
                # Wait exactly as long as the server says, else plain backoff.
                reset = _as_float(response.headers.get("X-RequestCounter-Reset"))
                _requests_left = 0
                time.sleep(min(reset if reset is not None else BACKOFF_SECONDS, 60.0))
                continue
            response.raise_for_status()
            payload = response.json()
            cache.write_text(json.dumps(payload))
            return payload
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(BACKOFF_SECONDS * (attempt + 1))
    raise ApiUnavailable(f"/{path} failed after {retries} tries: {last_error}")


def competition(code: str) -> dict:
    return get(f"competitions/{code}")


def trophy_cabinet(code: str) -> pd.DataFrame:
    """Season winners with club crests, newest first.

    Only part of a competition's history carries a recorded winner, so rows
    without one are dropped rather than shown blank.
    """
    payload = competition(code)
    records = [
        {
            "season": int(str(season.get("startDate", ""))[:4]),
            "winner": season["winner"].get("name"),
            "crest": season["winner"].get("crest"),
            "competition": payload.get("name"),
            "emblem": payload.get("emblem"),
        }
        for season in (payload.get("seasons") or [])
        if season.get("winner")
    ]
    return pd.DataFrame.from_records(records).sort_values("season", ascending=False)


def league_table(code: str) -> pd.DataFrame:
    """Current table with club crests, wins, draws, losses and points."""
    payload = get(f"competitions/{code}/standings")
    frames = [
        pd.DataFrame.from_records(rows)
        for group in (payload.get("standings") or [])
        for rows in [group.get("table") or []]
        if rows
    ]
    if not frames:
        return pd.DataFrame()

    table = pd.concat(frames, ignore_index=True)
    # Flatten the nested team object first: reading `crest` after overwriting
    # `team` with the name would call .get on a plain string.
    table["team_name"] = table["team"].map(lambda t: t.get("name") if t else None)
    table["crest"] = table["team"].map(lambda t: t.get("crest") if t else None)
    table = table.drop(columns=["team"]).rename(columns={"team_name": "team"})
    keep = ["position", "team", "crest", "playedGames", "won", "draw", "lost",
            "goalDifference", "goalsFor", "goalsAgainst", "points"]
    return table.loc[:, [c for c in keep if c in table.columns]]


def club_crests() -> dict:
    """Club name -> crest URL across the covered competitions.

    Matching is exact, by design. The valuation dataset spells clubs its own way
    ("Manchester City" vs "Manchester City FC"), and a fuzzy match would happily
    attach the wrong badge, so an unmatched club simply shows no crest.
    """
    mapping: dict = {}
    for code in COMPETITIONS:
        try:
            payload = get(f"competitions/{code}/teams")
        except ApiUnavailable:
            continue
        for team in payload.get("teams") or []:
            name, crest = team.get("name"), team.get("crest")
            if name and crest:
                mapping.setdefault(name, crest)
                # Also index the bare form, so "Manchester City" resolves.
                mapping.setdefault(re.sub(r"\s+(FC|SC|AC|AS|CF|CD|RC|SS)$", "", name), crest)
    return mapping


if __name__ == "__main__":
    if not available():
        raise SystemExit("set FOOTBALL_DATA_KEY in .env first")
    cabinet = trophy_cabinet("PL")
    print(f"{len(cabinet)} PL seasons with a recorded winner")
    print(cabinet.head(5).to_string(index=False))
    print(league_table("PL").head(5).to_string(index=False))
