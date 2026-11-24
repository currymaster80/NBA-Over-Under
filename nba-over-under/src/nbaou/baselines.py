"""Non-neural baselines: mean predictor, season-average heuristic, Random Forest."""
from __future__ import annotations

import itertools
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .evaluate import mse


def mean_predictor(y_train: np.ndarray, n: int) -> np.ndarray:
    """Always predict the training-set mean total."""
    return np.full(n, float(np.mean(y_train)))


def random_forest(Xtr, ytr, Xva, yva, Xte, quick: bool = False, seed: int = 0, verbose: bool = True):
    """Random Forest with a small validation-set grid search.

    Returns (val_pred, test_pred, best_params, grid_results).
    """
    if quick:
        grid = [dict(max_features=0.1, min_samples_leaf=10)]
        n_estimators = 100
    else:
        grid = [
            dict(max_features=mf, min_samples_leaf=leaf)
            for mf, leaf in itertools.product([0.05, 0.2, 0.5], [5, 20, 50])
        ]
        n_estimators = 300

    results, best = [], None
    for params in grid:
        t0 = time.time()
        rf = RandomForestRegressor(
            n_estimators=n_estimators, n_jobs=-1, random_state=seed, **params
        ).fit(Xtr, ytr)
        val_pred = rf.predict(Xva)
        score = mse(yva, val_pred)
        results.append({**params, "val_mse": score})
        if verbose:
            print(f"  RF {params}  val MSE {score:7.1f}  ({time.time() - t0:.0f}s)")
        if best is None or score < best[0]:
            best = (score, params, rf, val_pred)
    _, params, rf, val_pred = best
    return val_pred, rf.predict(Xte), params, results
