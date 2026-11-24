"""End-to-end experiment: data -> features -> models -> metrics -> figures."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from . import baselines, collab, nets
from .data import load_all
from .evaluate import BREAKEVEN_AT_MINUS_110, betting_simulation, mse
from .features import Scaler, build_dataset, split_by_season

PAPER_REPORTED = {  # Table 1 / Section 5 of the paper (test season 2017-18)
    "Random Forest": 910.33,
    "Collaborative Filtering (biased MF)": 459.85,
    "Neural Net (MLP)": 369.84,
    "LSTM": 426.56,
    "Sportsbook line": 320.696,
}
ALL_MODELS = ("rf", "cf", "mlp", "lstm")


@dataclass
class Config:
    data_dir: str = "data"
    out_dir: str = "results"
    train_seasons: tuple = ("2012-2013", "2013-2014", "2014-2015", "2015-2016")
    val_season: str = "2016-2017"
    test_season: str = "2017-2018"
    n_history: int = 3
    seeds: int = 5
    quick: bool = False
    models: tuple = ALL_MODELS
    max_epochs: int = 150
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
def _score(name, val_preds, test_preds, splits):
    """Metrics for one model. ``*_preds`` are lists (one array per seed)."""
    y_va, y_te = splits.val.y, splits.test.y
    line = splits.test.meta["line"].to_numpy(dtype=float)
    has = ~np.isnan(line)
    per_seed = []
    for pv, pt in zip(val_preds, test_preds):
        bet = betting_simulation(y_te, pt, line)
        per_seed.append(
            {
                "val_mse": mse(y_va, pv),
                "test_mse": mse(y_te, pt),
                "test_mse_line_games": mse(y_te[has], np.asarray(pt)[has]),
                "test_bias": float(np.mean(np.asarray(pt) - y_te)),
                "win_rate": bet.win_rate,
                "ci_low": bet.ci_low,
                "ci_high": bet.ci_high,
                "n_bets": bet.n_bets,
                "roi_minus_110": bet.roi_minus_110,
                "under_share": bet.under_share,
            }
        )
    df = pd.DataFrame(per_seed)
    out = {"model": name, "n_runs": len(df)}
    for c in df.columns:
        out[c] = float(df[c].mean())
        out[c + "_std"] = float(df[c].std(ddof=0)) if len(df) > 1 else 0.0
    return out


def run(cfg: Config) -> pd.DataFrame:
    t_start = time.time()
    os.makedirs(cfg.out_dir, exist_ok=True)
    quick = cfg.quick

    print("== Loading data")
    raw = load_all(cfg.data_dir)
    ds = build_dataset(raw, cfg.n_history)
    sp = split_by_season(ds, cfg.train_seasons, cfg.val_season, cfg.test_season)
    print(f"   games: train {len(sp.train.meta)} | val {len(sp.val.meta)} | test {len(sp.test.meta)}")
    scaler = Scaler().fit(sp.train)
    y_tr, y_va, y_te = sp.train.y, sp.val.y, sp.test.y
    line_te = sp.test.meta["line"].to_numpy(dtype=float)

    scored, test_preds_out = [], {}
    n_va, n_te = len(y_va), len(y_te)

    # ---- reference predictors (no learning from features) -------------------
    ref = {
        "Predict training mean": (
            [baselines.mean_predictor(y_tr, n_va)], [baselines.mean_predictor(y_tr, n_te)]),
        "Season-average heuristic": (
            [sp.val.meta["naive_season_avg"].to_numpy()], [sp.test.meta["naive_season_avg"].to_numpy()]),
    }
    for name, (pv, pt) in ref.items():
        scored.append(_score(name, pv, pt, sp))
        test_preds_out[name] = pt[0]

    models = set(cfg.models)
    extras = {}

    # ---- Random Forest --------------------------------------------------------
    if "rf" in models:
        print("== Random Forest")
        Xtr, Xva, Xte = scaler.flat(sp.train), scaler.flat(sp.val), scaler.flat(sp.test)
        pv, pt, params, grid = baselines.random_forest(Xtr, y_tr, Xva, y_va, Xte, quick=quick)
        scored.append(_score("Random Forest", [pv], [pt], sp))
        test_preds_out["Random Forest"] = pt
        extras["rf_best_params"] = params
        pd.DataFrame(grid).to_csv(os.path.join(cfg.out_dir, "rf_grid.csv"), index=False)

    # ---- Collaborative filtering (rolling biased MF) --------------------------
    if "cf" in models:
        print("== Collaborative filtering")
        pv, pt, params, sweep = collab.tune_and_predict(
            raw.games, sp.val.meta, sp.test.meta, y_va, fallback=float(y_tr.mean()), quick=quick)
        scored.append(_score("Collaborative Filtering (biased MF)", [pv], [pt], sp))
        test_preds_out["Collaborative Filtering (biased MF)"] = pt
        extras["cf_best_params"] = params
        sweep.to_csv(os.path.join(cfg.out_dir, "cf_grid.csv"), index=False)

    # ---- Neural network ---------------------------------------------------------
    histories = {}
    if "mlp" in models:
        print("== Neural network (MLP 500-100-20-1)")
        flat = [scaler.flat(sp.train), scaler.flat(sp.val), scaler.flat(sp.test)]
        in_dim = flat[0].shape[1]
        fits, best_h, grid = nets.select_and_train(
            "MLP",
            lambda h: (lambda: nets.MLP(in_dim, dropout=h.get("dropout", 0.0))),
            nets.mlp_grid(quick),
            inputs=((flat[0],), (flat[1],), (flat[2],)),
            ys=(y_tr, y_va),
            y_mu=scaler.y_mu, y_sd=scaler.y_sd, n_seeds=cfg.seeds, max_epochs=cfg.max_epochs,
        )
        scored.append(_score("Neural Net (MLP)", [f.val_pred for f in fits], [f.test_pred for f in fits], sp))
        test_preds_out["Neural Net (MLP)"] = np.mean([f.test_pred for f in fits], axis=0)
        histories["Neural Net (MLP)"] = fits[0].history
        extras["mlp_best_params"] = best_h
        pd.DataFrame(grid).to_csv(os.path.join(cfg.out_dir, "mlp_grid.csv"), index=False)

    # ---- LSTM ---------------------------------------------------------------------
    if "lstm" in models:
        print("== LSTM")
        seq = [scaler.seq(sp.train), scaler.seq(sp.val), scaler.seq(sp.test)]
        stat = [scaler.static(sp.train), scaler.static(sp.val), scaler.static(sp.test)]
        seq_dim, stat_dim = seq[0].shape[-1], stat[0].shape[-1]
        fits, best_h, grid = nets.select_and_train(
            "LSTM",
            lambda h: (lambda: nets.LSTMNet(seq_dim, stat_dim, hidden=h.get("hidden", 20),
                                            dropout=h.get("dropout", 0.2))),
            nets.lstm_grid(quick),
            inputs=tuple((s, t) for s, t in zip(seq, stat)),
            ys=(y_tr, y_va),
            y_mu=scaler.y_mu, y_sd=scaler.y_sd, n_seeds=cfg.seeds, max_epochs=cfg.max_epochs,
        )
        scored.append(_score("LSTM", [f.val_pred for f in fits], [f.test_pred for f in fits], sp))
        test_preds_out["LSTM"] = np.mean([f.test_pred for f in fits], axis=0)
        histories["LSTM"] = fits[0].history
        extras["lstm_best_params"] = best_h
        pd.DataFrame(grid).to_csv(os.path.join(cfg.out_dir, "lstm_grid.csv"), index=False)

    # ---- Sportsbook benchmark (only games that have a valid line) -------------------
    has = ~np.isnan(line_te)
    book_mse = mse(y_te[has], line_te[has])
    scored.append(
        {"model": "Sportsbook line", "n_runs": 1, "val_mse": np.nan, "test_mse": book_mse,
         "test_mse_line_games": book_mse,
         "test_bias": float(np.mean(line_te[has] - y_te[has])), "win_rate": np.nan, "n_bets": 0,
         "roi_minus_110": np.nan}
    )

    # ---- Constant-direction betting strategies: show how much of any "win rate"
    # is just a lean towards Over/Under rather than predictive skill. ------------------
    for name, shift in (("Always bet Under", -1.0), ("Always bet Over", +1.0)):
        bet = betting_simulation(y_te, line_te + shift, line_te)
        scored.append(
            {"model": name, "n_runs": 1, "val_mse": np.nan, "test_mse": np.nan,
             "test_mse_line_games": np.nan, "test_bias": np.nan, "win_rate": bet.win_rate,
             "ci_low": bet.ci_low, "ci_high": bet.ci_high, "n_bets": bet.n_bets,
             "roi_minus_110": bet.roi_minus_110, "under_share": bet.under_share}
        )

    table = pd.DataFrame(scored)
    table.to_csv(os.path.join(cfg.out_dir, "metrics.csv"), index=False)

    pred_df = sp.test.meta[["game_id", "date", "home", "away", "total", "line"]].copy()
    for k, v in test_preds_out.items():
        pred_df[k] = np.asarray(v)
    pred_df.to_csv(os.path.join(cfg.out_dir, "test_predictions.csv"), index=False)

    with open(os.path.join(cfg.out_dir, "run_info.json"), "w") as f:
        json.dump({"config": asdict(cfg), "selected": extras,
                   "split_sizes": [len(sp.train.meta), len(sp.val.meta), len(sp.test.meta)],
                   "test_games_with_line": int(has.sum()),
                   "test_mean_total": float(y_te.mean()), "train_mean_total": float(y_tr.mean()),
                   "runtime_sec": round(time.time() - t_start, 1)}, f, indent=2, default=str)

    from .report import make_figures, write_markdown  # local import: keeps matplotlib optional
    make_figures(cfg.out_dir, table, histories, os.path.join(cfg.out_dir, "cf_grid.csv"))
    write_markdown(cfg.out_dir, table, int(has.sum()), len(y_te))
    print(f"== Done in {time.time() - t_start:.0f}s. Results in {cfg.out_dir}/")
    return table
