"""Figures and the human-readable results table."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .evaluate import BREAKEVEN_AT_MINUS_110

PAPER_TEST_MSE = {
    "Random Forest": 910.33,
    "Collaborative Filtering (biased MF)": 459.85,
    "Neural Net (MLP)": 369.84,
    "LSTM": 426.56,
    "Sportsbook line": 320.696,
}


def _pm(mean, std, fmt="{:.1f}"):
    if pd.isna(mean):
        return "-"
    return fmt.format(mean) + (f" ± {fmt.format(std)}" if std and std > 0 else "")


def write_markdown(out_dir: str, table: pd.DataFrame, n_line_games: int, n_test: int) -> None:
    lines = [
        f"Test season games: {n_test} ({n_line_games} with a valid sportsbook line). "
        f"Break-even win rate at -110 pricing: {100 * BREAKEVEN_AT_MINUS_110:.2f}%.",
        "",
        "| Model | Val MSE | Test MSE | Test MSE (games with line) | Mean error (bias) | Beats line | 95% CI | Under bets | ROI @ -110 | Paper test MSE |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        win = "-" if pd.isna(r.get("win_rate")) else f"{100 * r['win_rate']:.1f}% ({int(round(r['n_bets']))} bets)"
        ci = "-" if pd.isna(r.get("ci_low")) else f"{100 * r['ci_low']:.1f}–{100 * r['ci_high']:.1f}%"
        roi = "-" if pd.isna(r.get("roi_minus_110")) else f"{100 * r['roi_minus_110']:+.1f}%"
        under = "-" if pd.isna(r.get("under_share")) else f"{100 * r['under_share']:.0f}%"
        paper = PAPER_TEST_MSE.get(r["model"])
        lines.append(
            "| {m} | {va} | {te} | {tl} | {b} | {w} | {ci} | {u} | {roi} | {p} |".format(
                m=r["model"],
                va=_pm(r["val_mse"], r.get("val_mse_std", 0)),
                te=_pm(r["test_mse"], r.get("test_mse_std", 0)),
                tl=_pm(r["test_mse_line_games"], r.get("test_mse_line_games_std", 0)),
                b="-" if pd.isna(r["test_bias"]) else f"{r['test_bias']:+.1f}",
                w=win, ci=ci, u=under, roi=roi, p="-" if paper is None else f"{paper:.1f}",
            )
        )
    with open(os.path.join(out_dir, "results_table.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def make_figures(out_dir: str, table: pd.DataFrame, histories: dict, cf_grid_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. Test MSE per model vs. the sportsbook ---------------------------------
    t = table.dropna(subset=["test_mse"]).copy()
    book =t.loc[t["model"] == "Sportsbook line", "test_mse"].iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = t.sort_values("test_mse", ascending=False)
    colors = ["#c0392b" if m == "Sportsbook line" else "#2c6fbb" for m in order["model"]]
    ax.barh(order["model"], order["test_mse"], xerr=order.get("test_mse_std", 0), color=colors)
    ax.axvline(book, color="#c0392b", ls="--", lw=1)
    ax.set_xlabel("Test MSE (points², lower is better)")
    ax.set_title("Predicting the total points of an NBA game (2017-18 test season)")
    for y, v in enumerate(order["test_mse"]):
        ax.text(v + 3, y, f"{v:.0f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_test_mse.png"), dpi=150)
    plt.close(fig)

    # 2. CF validation MSE vs. look-back window -----------------------------------
    if os.path.exists(cf_grid_path):
        g = pd.read_csv(cf_grid_path)
        best = g.sort_values("val_mse").groupby("window_days", as_index=False).first().sort_values("window_days")
        fig, ax = plt.subplots(figsize=(6, 3.8))
        ax.plot(best["window_days"], best["val_mse"], marker="o")
        ax.set_xlabel("Look-back window (days)")
        ax.set_ylabel("Validation MSE")
        ax.set_title("Collaborative filtering: window sweep")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_cf_window.png"), dpi=150)
        plt.close(fig)

    # 3. Training curves ---------------------------------------------------------------
    if histories:
        fig, axes = plt.subplots(1, len(histories), figsize=(5.2 * len(histories), 3.8), squeeze=False)
        for ax, (name, h) in zip(axes[0], histories.items()):
            epochs = np.arange(1, len(h["val_mse"]) + 1)
            ax.plot(epochs, h["train_mse"], label="train")
            ax.plot(epochs, h["val_mse"], label="validation")
            best = int(np.argmin(h["val_mse"])) + 1
            ax.axvline(best, color="grey", ls=":", lw=1, label=f"best epoch ({best})")
            ax.set_title(name + " (seed 0)")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE (points²)")
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_training_curves.png"), dpi=150)
        plt.close(fig)
