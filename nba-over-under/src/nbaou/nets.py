"""PyTorch models: the paper's fully-connected network and LSTM (Sections 4.2-4.3).

Both regress the *standardised* game total with MSE; reported errors are always
converted back to points^2.  The epoch with the lowest validation MSE is kept
(early stopping), so the validation season is used for model selection only.
"""
from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn


# --------------------------------------------------------------------------- #
# Architectures
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """input -> 500 -> 100 -> 20 -> 1 with ReLU, exactly the paper's shape."""

    def __init__(self, in_dim: int, hidden=(500, 100, 20), dropout: float = 0.0):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LSTMNet(nn.Module):
    """One LSTM layer reads the last 3 games one by one (oldest first).

    The final hidden state is concatenated with the static features (rest,
    travel, team identities) and mapped to the predicted total by a linear layer.
    """

    def __init__(self, seq_dim: int, static_dim: int, hidden: int = 20, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden, num_layers=1, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden + static_dim, 1)

    def forward(self, seq, static):
        _, (h_n, _) = self.lstm(seq)
        z = torch.cat([self.drop(h_n[-1]), static], dim=1)
        return self.head(z).squeeze(-1)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@dataclass
class FitResult:
    val_pred: np.ndarray
    test_pred: np.ndarray
    history: dict = field(default_factory=dict)  # per-epoch train/val MSE (points^2)
    best_epoch: int = 0


def _tensors(arrs):
    return [torch.as_tensor(a, dtype=torch.float32) for a in arrs]


def fit_model(
    make_model,
    train_inputs,
    y_train,
    val_inputs,
    y_val,
    test_inputs,
    y_mu: float,
    y_sd: float,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 64,
    max_epochs: int = 150,
    patience: int = 15,
    seed: int = 0,
) -> FitResult:
    """Train with AdamW + early stopping on validation MSE.

    ``*_inputs`` are tuples of arrays (``(X,)`` for the MLP, ``(seq, static)`` for the LSTM).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr, Xva, Xte = _tensors(train_inputs), _tensors(val_inputs), _tensors(test_inputs)
    ytr = torch.as_tensor((np.asarray(y_train) - y_mu) / y_sd, dtype=torch.float32)
    n = len(ytr)
    gen = torch.Generator().manual_seed(seed)

    def predict(inputs):
        model.eval()
        with torch.no_grad():
            return model(*inputs).numpy() * y_sd + y_mu

    best, best_state, best_epoch, bad = float("inf"), None, 0, 0
    hist = {"train_mse": [], "val_mse": []}
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss = nn.functional.mse_loss(model(*[x[idx] for x in Xtr]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        tr = float(np.mean((predict(Xtr) - y_train) ** 2))
        va = float(np.mean((predict(Xva) - y_val) ** 2))
        hist["train_mse"].append(tr)
        hist["val_mse"].append(va)
        if va < best - 1e-6:
            best, best_state, best_epoch, bad = va, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return FitResult(predict(Xva), predict(Xte), hist, best_epoch)


def select_and_train(
    name: str,
    make_model_for,       # callable(hparams) -> make_model callable
    grid: list[dict],
    inputs,               # (train_inputs, val_inputs, test_inputs)
    ys,                   # (y_train, y_val)
    y_mu: float,
    y_sd: float,
    n_seeds: int = 5,
    max_epochs: int = 150,
    verbose: bool = True,
):
    """Pick hyper-parameters on validation (1 seed), then train ``n_seeds`` seeds.

    Returns (list[FitResult], best_hparams, grid_results).
    """
    tr_in, va_in, te_in = inputs
    y_tr, y_va = ys

    def run(h, seed):
        h = dict(h)
        train_kw = {k: h.pop(k) for k in ("lr", "weight_decay", "batch_size") if k in h}
        return fit_model(
            make_model_for(h), tr_in, y_tr, va_in, y_va, te_in, y_mu, y_sd,
            max_epochs=max_epochs, seed=seed, **train_kw,
        )

    results, best = [], None
    if len(grid) > 1:
        for h in grid:
            t0 = time.time()
            r = run(h, seed=0)
            score = float(np.mean((r.val_pred - y_va) ** 2))
            results.append({**h, "val_mse": score, "best_epoch": r.best_epoch})
            if verbose:
                print(f"  {name} {h}  val MSE {score:7.1f}  epoch {r.best_epoch:3d}  ({time.time() - t0:.0f}s)")
            if best is None or score < best[0]:
                best = (score, h)
        best_h = best[1]
    else:
        best_h = grid[0]

    fits = [run(best_h, seed=s) for s in range(n_seeds)]
    if verbose:
        vs = [float(np.mean((f.val_pred - y_va) ** 2)) for f in fits]
        print(f"  {name} best {best_h}: val MSE {np.mean(vs):.1f} +- {np.std(vs):.1f} over {n_seeds} seeds")
    return fits, best_h, results


def mlp_grid(quick: bool):
    if quick:
        return [dict(lr=1e-3, weight_decay=1e-1, dropout=0.3)]
    return [
        dict(lr=lr, weight_decay=wd, dropout=dp)
        for lr, wd, dp in itertools.product([1e-4, 1e-3], [1e-2, 1e-1, 1.0], [0.0, 0.3, 0.5])
    ]


def lstm_grid(quick: bool):
    if quick:
        return [dict(lr=2e-3, weight_decay=1e-2, dropout=0.2)]
    return [
        dict(lr=lr, weight_decay=1e-2, dropout=dp, hidden=hd)
        for lr, dp, hd in itertools.product([5e-4, 2e-3], [0.2, 0.5], [20, 40])
    ]
