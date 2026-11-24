"""Metrics, including the paper's "beat the house" betting simulation."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# Standard US sportsbook price: risk 110 to win 100 -> win pays 100/110 per unit.
PAYOUT_AT_MINUS_110 = 100.0 / 110.0
BREAKEVEN_AT_MINUS_110 = 110.0 / 210.0  # 52.38 %


def mse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(y, float) - np.asarray(p, float)) ** 2))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = wins / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))


@dataclass
class BettingResult:
    n_games: int      # games that have a valid line
    n_bets: int       # bets actually placed (no bet if model == line; pushes excluded)
    wins: int
    win_rate: float
    ci_low: float
    ci_high: float
    roi_minus_110: float  # profit per unit staked at -110 pricing
    under_share: float    # fraction of placed bets that were Unders (1.0 = always Under)

    def as_dict(self) -> dict:
        return asdict(self)


def betting_simulation(y: np.ndarray, pred: np.ndarray, line: np.ndarray) -> BettingResult:
    """Bet Over when ``pred > line`` and Under when ``pred < line``.

    * Games without a valid line are skipped.
    * Pushes (final total == line) are refunded and excluded from the win rate.
    * ROI assumes every bet is priced at -110.
    """
    y, pred, line = (np.asarray(a, float) for a in (y, pred, line))
    has_line = ~np.isnan(line)
    y, pred, line = y[has_line], pred[has_line], line[has_line]

    bet_over = pred > line
    bet_under = pred < line
    placed = (bet_over | bet_under) & (y != line)  # no bet on ties; pushes refunded
    won = (bet_over & (y > line)) | (bet_under & (y < line))

    n_bets = int(placed.sum())
    wins = int((won & placed).sum())
    losses = n_bets - wins
    rate = wins / n_bets if n_bets else float("nan")
    lo, hi = wilson_interval(wins, n_bets)
    roi = (wins * PAYOUT_AT_MINUS_110 - losses) / n_bets if n_bets else float("nan")
    under_share = float(bet_under[placed].mean()) if n_bets else float("nan")
    return BettingResult(int(has_line.sum()), n_bets, wins, rate, lo, hi, float(roi), under_share)
