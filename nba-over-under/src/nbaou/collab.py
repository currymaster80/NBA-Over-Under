"""Collaborative-filtering view of the problem (Section 4.1 of the paper).

Treat the league as a recommender system: rows = home teams, columns = away
teams, "ratings" = total points scored when that pair met.  For every game date
we fit a *biased matrix factorisation* on all league games from the preceding
``window`` days and read off the predicted rating for the upcoming matchup::

    y_hat(i, j) = mu + b_i + c_j + <u_i, v_j>

This is the same model family as the SVD of the Surprise library that the paper
used (SVD-style factorisation with user/item biases), implemented here in
NumPy so that the repository has no fragile extra dependency.  Fitting is
full-batch Adam on an L2-regularised squared loss.  Only games strictly
*before* the predicted date are used.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from .evaluate import mse


def fit_biased_mf(h, a, y, n_teams, k=3, reg_bias=3.0, reg_factor=3.0, steps=120, lr=0.15, seed=0):
    """Fit mu + b_i + c_j + u_i.v_j on ratings y[(h, a)]; returns a predict closure."""
    mu = float(np.mean(y))
    r = y - mu
    rng = np.random.default_rng(seed)
    params = {
        "b": np.zeros(n_teams),
        "c": np.zeros(n_teams),
        "U": 0.1 * rng.standard_normal((n_teams, k)),
        "V": 0.1 * rng.standard_normal((n_teams, k)),
    }
    m1 = {p: np.zeros_like(v) for p, v in params.items()}
    m2 = {p: np.zeros_like(v) for p, v in params.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    for t in range(1, steps + 1):
        b, c, U, V = params["b"], params["c"], params["U"], params["V"]
        err = b[h] + c[a] + np.einsum("ij,ij->i", U[h], V[a]) - r  # [m]
        grads = {
            "b": 2 * np.bincount(h, err, n_teams) + 2 * reg_bias * b,
            "c": 2 * np.bincount(a, err, n_teams) + 2 * reg_bias * c,
            "U": 2 * reg_factor * U,
            "V": 2 * reg_factor * V,
        }
        np.add.at(grads["U"], h, 2 * err[:, None] * V[a])
        np.add.at(grads["V"], a, 2 * err[:, None] * U[h])
        for p in params:
            g = grads[p]
            m1[p] = b1 * m1[p] + (1 - b1) * g
            m2[p] = b2 * m2[p] + (1 - b2) * g * g
            params[p] = params[p] - lr * (m1[p] / (1 - b1**t)) / (np.sqrt(m2[p] / (1 - b2**t)) + eps)

    def predict(qh, qa):
        p = params
        return mu + p["b"][qh] + p["c"][qa] + np.einsum("ij,ij->i", p["U"][qh], p["V"][qa])

    return predict


class RollingCF:
    """Predict each target game from league results of the previous ``window`` days."""

    def __init__(self, games: pd.DataFrame, fallback: float, min_ratings: int = 5):
        teams = sorted(set(games["home"]) | set(games["away"]))
        self.idx = {t: i for i, t in enumerate(teams)}
        self.n = len(teams)
        self.fallback = fallback
        self.min_ratings = min_ratings
        g = games.sort_values("date")
        self.by_season = {}
        for season, d in g.groupby("season"):
            self.by_season[season] = (
                d["date"].to_numpy("datetime64[D]").astype(np.int64),
                d["home"].map(self.idx).to_numpy(),
                d["away"].map(self.idx).to_numpy(),
                d["total"].to_numpy(dtype=float),
            )

    def predict(self, targets: pd.DataFrame, window: int, **mf_kwargs) -> np.ndarray:
        out = np.full(len(targets), self.fallback, dtype=float)
        tdate = targets["date"].to_numpy("datetime64[D]").astype(np.int64)
        for (season, day), pos in pd.Series(np.arange(len(targets))).groupby(
            [targets["season"].to_numpy(), tdate]
        ):
            dates, h, a, y = self.by_season[season]
            lo = np.searchsorted(dates, day - window, side="left")
            hi = np.searchsorted(dates, day, side="left")  # strictly before the game date
            if hi - lo < self.min_ratings:
                continue
            pred = fit_biased_mf(h[lo:hi], a[lo:hi], y[lo:hi], self.n, **mf_kwargs)
            q = pos.to_numpy()
            qh = targets["home"].iloc[q].map(self.idx).to_numpy()
            qa = targets["away"].iloc[q].map(self.idx).to_numpy()
            out[q] = pred(qh, qa)
        return out


def tune_and_predict(games, val_meta, test_meta, y_val, fallback, quick=False, verbose=True):
    """Grid-search (window, regularisation) on the validation season.

    Returns (val_pred, test_pred, best_params, sweep_table).
    """
    cf = RollingCF(games, fallback)
    if quick:
        windows, regs = [3, 5, 8], [(3.0, 3.0)]
    else:
        # Windows go well beyond the paper's 5 days: the optimum must not sit on the grid edge.
        windows = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 45, 60, 90]
        # (reg_bias, reg_factor); reg_factor=1000 effectively switches the latent factors off.
        regs = list(itertools.product([1.0, 3.0, 10.0, 30.0], [3.0, 1000.0]))

    rows, best = [], None
    t0 = time.time()
    for w, (rb, rf) in itertools.product(windows, regs):
        pred = cf.predict(val_meta, w, reg_bias=rb, reg_factor=rf)
        score = mse(y_val, pred)
        rows.append({"window_days": w, "reg_bias": rb, "reg_factor": rf, "val_mse": score})
        if best is None or score < best[0]:
            best = (score, dict(window=w, reg_bias=rb, reg_factor=rf), pred)
    if verbose:
        print(f"  CF grid ({len(rows)} configs) {time.time() - t0:.0f}s -> best {best[1]}  val MSE {best[0]:.1f}")
    params = best[1]
    test_pred = cf.predict(
        test_meta, params["window"], reg_bias=params["reg_bias"], reg_factor=params["reg_factor"]
    )
    return best[2], test_pred, params, pd.DataFrame(rows)
