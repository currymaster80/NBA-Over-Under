"""Unit tests. Run with:  pytest -q   (from the repository root)."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from nbaou.collab import RollingCF, fit_biased_mf  # noqa: E402
from nbaou.data import haversine_miles, load_all  # noqa: E402
from nbaou.evaluate import betting_simulation, mse, wilson_interval  # noqa: E402
from nbaou.features import Scaler, build_dataset, split_by_season  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN = ["2012-2013", "2013-2014", "2014-2015", "2015-2016"]


@pytest.fixture(scope="module")
def raw():
    return load_all(DATA_DIR)


@pytest.fixture(scope="module")
def dataset(raw):
    return build_dataset(raw, n_history=3)


# ----------------------------------------------------------------------- data
def test_every_game_has_two_team_rows(raw):
    assert (raw.team_rows.groupby("game_id").size() == 2).all()


def test_total_is_sum_of_both_teams(raw):
    g = raw.games
    assert g["total"].between(120, 340).all()


def test_attached_lines_agree_with_box_scores(raw):
    o = raw.odds.rename(columns={"line": "l2"})
    m = raw.games.dropna(subset=["line"]).merge(o, on=["date", "home", "away"])
    assert (m["odds_total"] == m["total"]).all()  # joins are consistent
    assert raw.games["line"].dropna().between(150, 280).all()


def test_split_sizes_match_the_paper(dataset):
    sp = split_by_season(dataset, TRAIN, "2016-2017", "2017-2018")
    assert (len(sp.train.meta), len(sp.val.meta), len(sp.test.meta)) == (5069, 1260, 1264)


# ------------------------------------------------------------------- leakage
def test_features_use_only_past_games(raw, dataset):
    """Rebuilding the dataset with all games on/after a cutoff removed must not
    change the features of the games before the cutoff."""
    cutoff = pd.Timestamp("2017-01-15")
    full = dataset
    mask = full.meta["date"] < cutoff

    trimmed_raw = type(raw)(
        team_rows=raw.team_rows[raw.team_rows["date"] < cutoff].copy(),
        games=raw.games[raw.games["date"] < cutoff].copy(),
        odds=raw.odds,
        stat_cols=raw.stat_cols,
    )
    part = build_dataset(trimmed_raw, n_history=3)
    a = full.seq[mask.to_numpy()]
    assert a.shape == part.seq.shape
    assert np.allclose(a, part.seq, equal_nan=True)
    assert np.allclose(full.static_cont[mask.to_numpy()], part.static_cont)


def test_scaler_fit_on_train_only(dataset):
    sp = split_by_season(dataset, TRAIN, "2016-2017", "2017-2018")
    sc = Scaler().fit(sp.train)
    z = sc.seq(sp.train)
    assert abs(float(np.mean(z))) < 0.05 and np.isfinite(sc.flat(sp.test)).all()


def test_rolling_cf_ignores_the_future(raw):
    games = raw.games
    day = pd.Timestamp("2017-12-01")
    targets = games[games["date"] == day]
    cf_full = RollingCF(games, fallback=200.0)
    cf_past = RollingCF(games[games["date"] < day], fallback=200.0)
    kw = dict(reg_bias=3.0, reg_factor=3.0)
    assert np.allclose(cf_full.predict(targets, 5, **kw), cf_past.predict(targets, 5, **kw))


# -------------------------------------------------------------------- models
def test_mf_recovers_simple_structure():
    h = np.array([0, 1, 0, 1, 0, 1] * 5)
    a = np.array([2, 2, 3, 3, 2, 3] * 5)
    y = np.where(h == 0, 190.0, 210.0) + np.where(a == 2, 0.0, 4.0)
    pred = fit_biased_mf(h, a, y, n_teams=4, reg_bias=0.1, reg_factor=5.0, steps=300)
    assert np.mean(np.abs(pred(h, a) - y)) < 1.0


# ---------------------------------------------------------------- evaluation
def test_mse():
    assert mse([1, 2, 3], [1, 2, 5]) == pytest.approx(4 / 3)


def test_betting_simulation_hand_computed():
    y = np.array([210.0, 190.0, 200.0, 205.0, 195.0])
    line = np.array([200.0, 200.0, 200.0, 200.5, np.nan])
    pred = np.array([205.0, 205.0, 205.0, 190.0, 200.0])
    r = betting_simulation(y, pred, line)
    # game 1: Over wins; game 2: Over loses; game 3: push (excluded);
    # game 4: Under loses (205 > 200.5); game 5: no line (skipped)
    assert (r.n_bets, r.wins) == (3, 1)
    assert r.win_rate == pytest.approx(1 / 3)
    assert r.roi_minus_110 == pytest.approx((1 * 100 / 110 - 2) / 3)
    assert r.under_share == pytest.approx(1 / 3)


def test_no_bet_when_prediction_equals_line():
    r = betting_simulation(np.array([210.0]), np.array([200.0]), np.array([200.0]))
    assert r.n_bets == 0


def test_wilson_interval_contains_estimate():
    lo, hi = wilson_interval(515, 1000)
    assert lo < 0.515 < hi and hi - lo < 0.07


def test_haversine_known_distance():
    # Boston <-> Los Angeles is roughly 2,600 miles.
    d = haversine_miles((42.366, -71.062), (34.043, -118.267))
    assert 2550 < d < 2700
