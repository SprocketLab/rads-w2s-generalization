#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _zfit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return mu, sd


class _RegMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):  # noqa: D102
        return self.net(x).squeeze(-1)


def _oof_regress(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
    weight_decay: float,
    seed: int,
    dev: torch.device,
) -> np.ndarray:
    """Out-of-fold MLP regression: every row gets predicted by a model that never saw it.
    Inputs are z-scored per fold (stats from the train folds only, no peeking)."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    preds = np.zeros(n, dtype=np.float64)
    for fi, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(perm, test_idx, assume_unique=False)
        mu, sd = _zfit(X[train_idx])
        Xtr = torch.tensor((X[train_idx] - mu) / sd, dtype=torch.float32, device=dev)
        ytr = torch.tensor(y[train_idx], dtype=torch.float32, device=dev)
        Xte = torch.tensor((X[test_idx] - mu) / sd, dtype=torch.float32, device=dev)
        # global torch seeding, kept: a per-call Generator would change the draw
        # stream and with it every recorded mlpstep1 number
        torch.manual_seed(seed * 1000 + fi)
        model = _RegMLP(X.shape[1], hidden).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        m = Xtr.shape[0]
        model.train()
        for _ in range(epochs):
            order = torch.randperm(m, device=dev)
            for b0 in range(0, m, batch_size):
                idx = order[b0:b0 + batch_size]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(Xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            preds[test_idx] = model(Xte).double().cpu().numpy()
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return preds


def mlp_step1_rp_scores(
    weak_acts: np.ndarray,
    strong_acts: np.ndarray,
    weak_labels: np.ndarray,
    n_folds: int = 5,
    hidden: int = 256,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 30.0,
    seed: int = 0,
    device: str | None = None,
    ridge_reg: float = 0.1,
    n_ensemble: int = 3,
) -> np.ndarray:
    """rp with a cross-fitted, weight-decayed MLP ensemble for step 1 (the weak side) and
    rp's in-sample kernel-ridge projection for step 2 (the strong side). Returns [n] scores."""
    Xw = np.asarray(weak_acts, dtype=np.float32)
    y = np.asarray(weak_labels, dtype=np.float64)
    n = Xw.shape[0]
    if strong_acts.shape[0] != n or y.shape[0] != n:
        raise ValueError("weak_acts, strong_acts, weak_labels must share the first dim n")
    dev = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    yc = y - y.mean()
    # step 1: cross-fitted MLP ensemble instead of the weak-side ridge
    p = np.zeros(n, dtype=np.float64)
    for e in range(n_ensemble):
        p += _oof_regress(Xw, yc, n_folds, hidden, epochs, lr, batch_size, weight_decay,
                          seed + 77 + 131 * e, dev)
    p /= n_ensemble
    a = yc - p
    # step 2: rp's strong-side kernel ridge, in-sample
    Xs64 = np.asarray(strong_acts, dtype=np.float64)
    Xs64 = Xs64 - Xs64.mean(axis=0, keepdims=True)
    # macOS Accelerate BLAS warns on these matmuls although the results are finite, as in RADS-Lin
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Ks = Xs64 @ Xs64.T / n
        rs = ridge_reg * (np.trace(Ks) / n) + 1e-12
        v = Ks @ np.linalg.solve(Ks + rs * np.eye(n), a)
    scores = np.abs(v)
    if not np.isfinite(scores).all():
        raise ValueError("mlp_step1_rp_scores produced non-finite values (diverged fold)")
    return scores


def linstep1_rp_scores(
    weak_acts: np.ndarray,
    strong_acts: np.ndarray,
    weak_labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 0,
    ridge_reg: float = 0.1,
) -> np.ndarray:
    """rp's step-1 kernel ridge fitted out-of-fold, as in mlpstep1, with the strong side
    unchanged. It differs from rp only in the cross-fitting and from mlpstep1 only in
    linearity. Closed-form, no optimizer, runs in seconds."""
    Xw = np.asarray(weak_acts, dtype=np.float64)
    y = np.asarray(weak_labels, dtype=np.float64)
    n = len(y)
    if Xw.shape[0] != n or strong_acts.shape[0] != n:
        raise ValueError("weak_acts, strong_acts, weak_labels must share the first dim n")
    yc = y - y.mean()
    Xwc = Xw - Xw.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    preds = np.zeros(n, dtype=np.float64)
    Xsc = np.asarray(strong_acts, dtype=np.float64)
    Xsc = Xsc - Xsc.mean(axis=0, keepdims=True)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for te in np.array_split(perm, n_folds):
            tr = np.setdiff1d(perm, te, assume_unique=False)
            m = len(tr)
            Ktr = Xwc[tr] @ Xwc[tr].T / m
            r = ridge_reg * (np.trace(Ktr) / m) + 1e-12
            mu_tr = yc[tr].mean()
            alpha = np.linalg.solve(Ktr + r * np.eye(m), yc[tr] - mu_tr)
            preds[te] = (Xwc[te] @ Xwc[tr].T / m) @ alpha + mu_tr
        a = yc - preds
        Ks = Xsc @ Xsc.T / n
        rs = ridge_reg * (np.trace(Ks) / n) + 1e-12
        v = Ks @ np.linalg.solve(Ks + rs * np.eye(n), a)
    scores = np.abs(v)
    if not np.isfinite(scores).all():
        raise ValueError("linstep1_rp_scores produced non-finite values")
    return scores
