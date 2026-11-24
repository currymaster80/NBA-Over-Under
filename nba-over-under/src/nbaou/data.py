"""Data loading, cleaning and team-name harmonisation.

Two raw sources are used (both come from the original "The Bank is Open" repo):

* ``data/final_game_data/<season>_games_final.csv`` - one row per *team* per
  game (so every game appears twice), box-score statistics from Basketball
  Reference, written from the point of view of ``team``.
* ``data/odds_data_processed/<season>.csv`` - one row per game with the
  sportsbook Over/Under line (``OU``) and the final point total (``Points``).

The sportsbook line is used **only for evaluation**, never as a model input.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Team names -> short franchise codes
# --------------------------------------------------------------------------- #
# Basketball Reference full names (game data).
GAME_TEAM_CODES = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "New Jersey Nets": "BKN",
    "Charlotte Bobcats": "CHA",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Hornets": "NOP",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Seattle SuperSonics": "SEA",  # own code: different city from OKC
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

# Sportsbook Review Online short names (odds data).
ODDS_TEAM_CODES = {
    "Atlanta": "ATL",
    "Boston": "BOS",
    "Brooklyn": "BKN",
    "Charlotte": "CHA",
    "Chicago": "CHI",
    "Cleveland": "CLE",
    "Dallas": "DAL",
    "Denver": "DEN",
    "Detroit": "DET",
    "Golden State": "GSW",
    "Houston": "HOU",
    "Indiana": "IND",
    "LA Clippers": "LAC",
    "LA Lakers": "LAL",
    "Memphis": "MEM",
    "Miami": "MIA",
    "Milwaukee": "MIL",
    "Minnesota": "MIN",
    "New Orleans": "NOP",
    "New York": "NYK",
    "Okla City": "OKC",
    "Oklahoma City": "OKC",
    "Seattle": "SEA",
    "Orlando": "ORL",
    "Philadelphia": "PHI",
    "Phoenix": "PHX",
    "Portland": "POR",
    "Sacramento": "SAC",
    "San Antonio": "SAS",
    "Toronto": "TOR",
    "Utah": "UTA",
    "Washington": "WAS",
}

# Approximate arena coordinates (lat, lon) used for the travel-distance feature.
TEAM_COORDS = {
    "ATL": (33.757, -84.396), "BOS": (42.366, -71.062), "BKN": (40.683, -73.975),
    "CHA": (35.225, -80.839), "CHI": (41.881, -87.674), "CLE": (41.497, -81.688),
    "DAL": (32.790, -96.810), "DEN": (39.749, -105.008), "DET": (42.341, -83.055),
    "GSW": (37.750, -122.203), "HOU": (29.751, -95.362), "IND": (39.764, -86.156),
    "LAC": (34.043, -118.267), "LAL": (34.043, -118.267), "MEM": (35.138, -90.051),
    "MIA": (25.781, -80.188), "MIL": (43.045, -87.918), "MIN": (44.980, -93.276),
    "NOP": (29.949, -90.082), "NYK": (40.751, -73.994), "OKC": (35.463, -97.515),
    "ORL": (28.539, -81.384), "PHI": (39.901, -75.172), "PHX": (33.446, -112.071),
    "POR": (45.532, -122.667), "SAC": (38.580, -121.500), "SAS": (29.427, -98.438),
    "SEA": (47.622, -122.354), "TOR": (43.643, -79.379), "UTA": (40.768, -111.901),
    "WAS": (38.898, -77.021),
}

# Full-game lines outside this range are impossible values in the odds files
# (five rows such as 19.5 or 43.5), so they are treated as data-entry errors.
VALID_OU_RANGE = (150.0, 280.0)


def haversine_miles(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in miles between two (lat, lon) points."""
    lat1, lon1, lat2, lon2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 3958.8 * np.arcsin(np.sqrt(h)))


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class RawData:
    """Cleaned raw tables.

    team_rows : one row per team per game (team point of view), with ``stat_cols``.
    games     : one row per game (home/away/total points).
    odds      : cleaned sportsbook lines keyed by (date, home, away).
    stat_cols : names of the numeric box-score columns.
    """

    team_rows: pd.DataFrame
    games: pd.DataFrame
    odds: pd.DataFrame
    stat_cols: list[str]


def _season_of(path: str) -> str:
    m = re.search(r"(\d{4}-\d{4})", os.path.basename(path))
    if not m:
        raise ValueError(f"cannot infer season from {path}")
    return m.group(1)


def load_team_rows(data_dir: str) -> tuple[pd.DataFrame, list[str]]:
    """Load every ``*_games_final.csv`` file into one team-perspective table."""
    files = sorted(glob.glob(os.path.join(data_dir, "final_game_data", "*_games_final.csv")))
    if not files:
        raise FileNotFoundError(
            f"No game files found under {data_dir}/final_game_data. "
            "Run `python scripts/download_data.py` or see the README."
        )
    frames = []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        df["season"] = _season_of(f)
        frames.append(df)
    rows = pd.concat(frames, ignore_index=True)

    # Dates are stored as YYYYMMDD followed by one extra digit (e.g. 201210300).
    rows["date"] = pd.to_datetime(rows["date"].astype(str).str[:8], format="%Y%m%d")
    rows["team_code"] = rows["team"].map(GAME_TEAM_CODES)
    rows["opp_code"] = rows["opponent"].map(GAME_TEAM_CODES)
    unknown = rows.loc[rows["team_code"].isna() | rows["opp_code"].isna(), ["team", "opponent"]]
    if len(unknown):
        raise ValueError(f"Unmapped team names: {sorted(set(unknown.stack()))}")
    rows["home"] = rows["home"].astype(int)

    stat_cols = [c for c in rows.columns if c.startswith("team_") or c.startswith("opponent_")]
    stat_cols = [c for c in stat_cols if c not in ("team_code", "opp_code")]
    rows[stat_cols] = rows[stat_cols].apply(pd.to_numeric, errors="coerce")
    # Ratios such as AST/TOV can be +-inf when the denominator is 0.
    rows[stat_cols] = rows[stat_cols].replace([np.inf, -np.inf], np.nan)

    # A game id shared by the two team rows of the same game.
    pair = rows[["team_code", "opp_code"]].apply(lambda r: "_".join(sorted(r)), axis=1)
    rows["game_id"] = rows["date"].dt.strftime("%Y%m%d") + "_" + pair
    return rows, stat_cols


def build_games(team_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per game: home, away, final total (home pts + away pts)."""
    counts = team_rows.groupby("game_id").size()
    bad = counts[counts != 2]
    if len(bad):  # drop incomplete games rather than guessing
        team_rows.drop(team_rows.index[team_rows["game_id"].isin(bad.index)], inplace=True)

    home = team_rows[team_rows["home"] == 1]
    games = pd.DataFrame(
        {
            "game_id": home["game_id"].values,
            "season": home["season"].values,
            "date": home["date"].values,
            "home": home["team_code"].values,
            "away": home["opp_code"].values,
            "total": (home["team_PTS"] + home["opponent_PTS"]).values,
        }
    )
    return games.sort_values(["date", "home"]).reset_index(drop=True)


def load_odds(data_dir: str) -> pd.DataFrame:
    """Load the sportsbook lines and drop obviously invalid ones."""
    files = sorted(glob.glob(os.path.join(data_dir, "odds_data_processed", "*.csv")))
    if not files:
        raise FileNotFoundError(f"No odds files found under {data_dir}/odds_data_processed")
    frames = []
    for f in files:
        d = pd.read_csv(f, index_col=0)
        d["odds_season"] = _season_of(f)
        frames.append(d)
    odds = pd.concat(frames, ignore_index=True)
    odds["date"] = pd.to_datetime(odds["Date"])
    odds["home"] = odds["Home"].map(ODDS_TEAM_CODES)
    odds["away"] = odds["Away"].map(ODDS_TEAM_CODES)
    odds = odds.dropna(subset=["home", "away", "OU"])
    lo, hi = VALID_OU_RANGE
    odds = odds[(odds["OU"] >= lo) & (odds["OU"] <= hi)]
    odds = odds.rename(columns={"OU": "line", "Points": "odds_total"})
    odds = odds[["date", "home", "away", "line", "odds_total"]]
    return odds.drop_duplicates(["date", "home", "away"]).reset_index(drop=True)


def attach_odds(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Left-join odds onto games by (date, home, away).

    A game only keeps its line if the odds file's recorded point total agrees
    with the box-score total; this guards against wrong-date / wrong-team joins.
    """
    merged = games.merge(odds, on=["date", "home", "away"], how="left")
    mismatch = merged["line"].notna() & (merged["odds_total"] != merged["total"])
    merged.loc[mismatch, ["line", "odds_total"]] = np.nan
    return merged.drop(columns=["odds_total"])


def load_all(data_dir: str = "data") -> RawData:
    team_rows, stat_cols = load_team_rows(data_dir)
    games = build_games(team_rows)
    games = attach_odds(games, load_odds(data_dir))
    return RawData(team_rows=team_rows, games=games, odds=load_odds(data_dir), stat_cols=stat_cols)
