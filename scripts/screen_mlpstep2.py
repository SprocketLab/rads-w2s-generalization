#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _midranks(v):
    # tied values share their mean rank, so the result does not depend on row order
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    ranks_sorted = np.arange(1, len(v) + 1, dtype=float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks_sorted[i : j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks = np.empty(len(v))
    ranks[order] = ranks_sorted
    return ranks


def _spearman(x, y):
    return float(np.corrcoef(_midranks(x), _midranks(y))[0, 1])


class _Head(nn.Module):
    def __init__(self, d_in: int, hidden):
        super().__init__()
        if hidden == "lin":
            self.net = nn.Linear(d_in, 1)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_in, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(),
                nn.Linear(hidden, 1),
            )

    def forward(self, x):  # noqa: D102
        return self.net(x).squeeze(-1)


def insample_mlp_fit_read(Xs, a, hidden, wd, seed, epochs=60, lr=1e-3, bs=256, n_ens=3):
    """Fit heads to a on ALL rows and read predictions on the SAME rows (open book).

    The target is scaled to unit std before fitting and the predictions are
    scaled back, so the same wd means the same cap whatever the residual's
    scale. hidden may be an int or "lin" for the linear-head control."""
    dev = torch.device("cpu")
    mu = Xs.mean(0, keepdims=True); sd = Xs.std(0, keepdims=True) + 1e-8
    X = torch.tensor((Xs - mu) / sd, dtype=torch.float32, device=dev)
    scale = float(a.std()) + 1e-12
    y = torch.tensor(a / scale, dtype=torch.float32, device=dev)
    n = X.shape[0]
    v = np.zeros(n, dtype=np.float64)
    for e in range(n_ens):
        torch.manual_seed(seed + 1000 * e)
        model = _Head(X.shape[1], hidden).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.MSELoss()
        model.train()
        for _ in range(epochs):
            order = torch.randperm(n)
            for b0 in range(0, n, bs):
                idx = order[b0:b0 + bs]
                opt.zero_grad(set_to_none=True)
                loss_fn(model(X[idx]), y[idx]).backward()
                opt.step()
        model.eval()
        with torch.no_grad():
            v += model(X).double().numpy()
    return (v / n_ens) * scale
