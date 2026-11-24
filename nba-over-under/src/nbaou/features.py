"""Leakage-free feature construction.

For a game between home team H and away team A on date d we use, for *each*
team, its previous ``n_history`` games of the same season (oldest -> newest).
Each past game contributes

* the 92 box-score statistics of that game (team's and its opponent's), and
* the **opponent's season-to-date averages** of the same statistics, computed
  using only games played *before* that past game (strength-of-schedule
  context, Figure 1 of the paper).

Static features per game: days of rest and miles travelled for both teams
(fatigue), plus one-hot team identities (home / away).

Nothing from the game being predicted (or any later game) is used.  Games where
either team has fewer than ``n_history`` earlier games in that season are
dropped (first ~3 games of each season).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import TEAM_COORDS, RawData, haversine_miles

MAX_REST_DAYS = 10  # cap so that the off-season / All-Star break don't dominate


@dataclass
class Dataset:
    """Model-ready arrays, row-aligned with ``meta``."""

    meta: pd.DataFrame          # game_id, season, date, home, away, total, line, naive_*
    seq: np.ndarray             # [N, n_history, F]  (F = 4 * n_stats) oldest -> newest
    static_cont: np.ndarray     # [N, 4] rest_home, rest_away, dist_home, dist_away
    onehot: np.ndarray          # [N, 2 * n_teams] home one-hot then away one-hot
    team_vocab: list[str]

    @property
    def y(self) -> np.ndarray:
        return self.meta["total"].to_numpy(dtype=np.float32)

    def subset(self, mask: np.ndarray) -> "Dataset":
        return Dataset(
            meta=self.meta.loc[mask].reset_index(drop=True),
            seq=self.seq[mask],
            static_cont=self.static_cont[mask],
            onehot=self.onehot[mask],
            team_vocab=self.team_vocab,
        )


def _prior_season_average(rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Expanding mean of ``cols`` over each team's *earlier* games in a season."""
    keys = [rows["team_code"], rows["season"]]
    vals = rows[cols]
    filled = vals.fillna(0.0)
    present = vals.notna().astype(float)
    prior_sum = filled.groupby(keys, sort=False).cumsum() - filled      # sum over earlier games
    prior_cnt = present.groupby(keys, sort=False).cumsum() - present    # count over earlier games
    return prior_sum / prior_cnt.replace(0.0, np.nan)


def build_dataset(raw: RawData, n_history: int = 3) -> Dataset:
    stat_cols = raw.stat_cols
    S = len(stat_cols)

    rows = raw.team_rows.sort_values(["team_code", "season", "date"]).reset_index(drop=True)
    rows["pos"] = rows.groupby(["team_code", "season"]).cumcount()

    # --- per-row vector: own game stats + opponent's prior season average ----
    prior_avg = _prior_season_average(rows, stat_cols)
    key = pd.MultiIndex.from_arrays([rows["game_id"], rows["team_code"]])
    prior_by_key = prior_avg.copy()
    prior_by_key.index = key
    opp_key = pd.MultiIndex.from_arrays([rows["game_id"], rows["opp_code"]])
    opp_prior = prior_by_key.reindex(opp_key).to_numpy()  # opponent's prior averages
    V = np.hstack([rows[stat_cols].to_numpy(dtype=np.float64), opp_prior])  # [R, 2S]

    pts_idx = stat_cols.index("team_PTS")
    opp_pts_idx = stat_cols.index("opponent_PTS")
    prior_total = prior_avg.iloc[:, pts_idx].to_numpy() + prior_avg.iloc[:, opp_pts_idx].to_numpy()
    game_total_row = (rows["team_PTS"] + rows["opponent_PTS"]).to_numpy()

    # --- locate the two team rows of every game --------------------------------
    pos_of = {(g, t): i for i, (g, t) in enumerate(zip(rows["game_id"], rows["team_code"]))}
    games = raw.games.copy()
    ih = np.array([pos_of[(g, h)] for g, h in zip(games["game_id"], games["home"])])
    ia = np.array([pos_of[(g, a)] for g, a in zip(games["game_id"], games["away"])])
    ok = (rows["pos"].to_numpy()[ih] >= n_history) & (rows["pos"].to_numpy()[ia] >= n_history)
    games, ih, ia = games[ok].reset_index(drop=True), ih[ok], ia[ok]
    N = len(games)

    # --- sequences ---------------------------------------------------------------
    seq = np.zeros((N, n_history, 4 * S), dtype=np.float32)
    last_totals = np.zeros((N, 2, n_history))
    for step in range(n_history):
        k = n_history - step  # games ago: oldest first
        seq[:, step, :] = np.hstack([V[ih - k], V[ia - k]])
        last_totals[:, 0, step] = game_total_row[ih - k]
        last_totals[:, 1, step] = game_total_row[ia - k]

    # --- naive reference predictors (no learning) --------------------------------
    games["naive_season_avg"] = (prior_total[ih] + prior_total[ia]) / 2.0
    games["naive_last_n"] = last_totals.mean(axis=(1, 2))

    # --- fatigue features ----------------------------------------------------------
    dates = rows["date"].to_numpy()
    home_flag = rows["home"].to_numpy()
    team_code = rows["team_code"].to_numpy()
    opp_code = rows["opp_code"].to_numpy()

    def rest_days(idx):
        delta = (dates[idx] - dates[idx - 1]).astype("timedelta64[D]").astype(float)
        return np.clip(delta, 0, MAX_REST_DAYS)

    def prev_site(idx):
        j = idx - 1
        return np.where(home_flag[j] == 1, team_code[j], opp_code[j])

    site = games["home"].to_numpy()  # game is played in the home team's city
    dist_h = np.array([haversine_miles(TEAM_COORDS[p], TEAM_COORDS[s]) for p, s in zip(prev_site(ih), site)])
    dist_a = np.array([haversine_miles(TEAM_COORDS[p], TEAM_COORDS[s]) for p, s in zip(prev_site(ia), site)])
    static_cont = np.stack([rest_days(ih), rest_days(ia), dist_h, dist_a], axis=1).astype(np.float32)

    # --- team identity one-hots ----------------------------------------------------
    vocab = sorted(set(games["home"]) | set(games["away"]))
    index = {t: i for i, t in enumerate(vocab)}
    onehot = np.zeros((N, 2 * len(vocab)), dtype=np.float32)
    onehot[np.arange(N), [index[t] for t in games["home"]]] = 1.0
    onehot[np.arange(N), [len(vocab) + index[t] for t in games["away"]]] = 1.0

    return Dataset(meta=games, seq=seq, static_cont=static_cont, onehot=onehot, team_vocab=vocab)


# --------------------------------------------------------------------------- #
# Splitting + scaling
# --------------------------------------------------------------------------- #
@dataclass
class Splits:
    train: Dataset
    val: Dataset
    test: Dataset


def split_by_season(ds: Dataset, train_seasons, val_season: str, test_season: str) -> Splits:
    s = ds.meta["season"].to_numpy()
    return Splits(
        train=ds.subset(np.isin(s, list(train_seasons))),
        val=ds.subset(s == val_season),
        test=ds.subset(s == test_season),
    )


class Scaler:
    """Z-scores features using **training** statistics only.

    Sequence features share one mean/std per statistic across all time steps, so
    a statistic has the same scale whichever game it came from.  One-hots are
    left untouched.
    """

    def fit(self, train: Dataset) -> "Scaler":
        flat = train.seq.reshape(-1, train.seq.shape[-1]).astype(np.float64)
        self.seq_mu = np.nanmean(flat, axis=0)
        sd = np.nanstd(flat, axis=0)
        self.seq_sd = np.where(sd < 1e-8, 1.0, sd)
        self.st_mu = train.static_cont.mean(axis=0)
        sd = train.static_cont.std(axis=0)
        self.st_sd = np.where(sd < 1e-8, 1.0, sd)
        y = train.y
        self.y_mu, self.y_sd = float(y.mean()), float(y.std())
        return self

    def seq(self, ds: Dataset) -> np.ndarray:
        z = (ds.seq - self.seq_mu) / self.seq_sd
        return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def static(self, ds: Dataset) -> np.ndarray:
        cont = (ds.static_cont - self.st_mu) / self.st_sd
        return np.hstack([cont, ds.onehot]).astype(np.float32)

    def flat(self, ds: Dataset) -> np.ndarray:
        """Input for non-recurrent models: flattened sequence + static features."""
        z = self.seq(ds)
        return np.hstack([z.reshape(len(z), -1), self.static(ds)]).astype(np.float32)
