#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def _whitener(X: np.ndarray, shrink: float, energy: float = 0.999):
    """Return (mean, W) with W built on the fold-train points: PCA basis truncated to
    `energy` variance, variances shrunk toward their mean by `shrink` for stability."""
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(1, Xc.shape[0] - 1)
    keep = np.cumsum(var) / var.sum() <= energy
    keep[0] = True
    Vt, var = Vt[keep], var[keep]
    var_s = (1.0 - shrink) * var + shrink * var.mean()
    W = Vt.T / np.sqrt(var_s + 1e-12)
    return mu, W  # z = (x - mu) @ W


def cca_alignment_scores(
    weak_acts: np.ndarray,
    strong_acts: np.ndarray,
    n_folds: int = 5,
    k_components: int = 64,
    shrink: float = 0.1,
    seed: int = 0,
    trim_frac: float = 0.0,
) -> np.ndarray:
    """Per-example out-of-fold CCA disagreement score. Returns [n] non-negative scores.

    trim_frac > 0 = ROBUST variant: fit once, drop the trim_frac fold-train points with the
    largest canonical disagreement, refit directions on the kept points, then score."""
    Xw = np.asarray(weak_acts, dtype=np.float64)
    Xs = np.asarray(strong_acts, dtype=np.float64)
    n = Xw.shape[0]
    if Xs.shape[0] != n:
        raise ValueError("weak_acts and strong_acts must share the first dim n")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)
    scores = np.zeros(n, dtype=np.float64)

    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for test_idx in folds:
            train_idx = np.setdiff1d(perm, test_idx, assume_unique=False)
            muw, Ww = _whitener(Xw[train_idx], shrink)
            mus, Ws = _whitener(Xs[train_idx], shrink)
            Zw_tr = (Xw[train_idx] - muw) @ Ww
            Zs_tr = (Xs[train_idx] - mus) @ Ws
            C = Zw_tr.T @ Zs_tr / Zw_tr.shape[0]
            U, rho, Vt = np.linalg.svd(C, full_matrices=False)
            k = min(k_components, U.shape[1])
            Uk, Vk = U[:, :k], Vt[:k].T
            if trim_frac > 0.0:
                d_tr = np.linalg.norm(Zw_tr @ Uk - Zs_tr @ Vk, axis=1)
                keep = d_tr <= np.quantile(d_tr, 1.0 - trim_frac)
                kt = train_idx[keep]
                muw, Ww = _whitener(Xw[kt], shrink)
                mus, Ws = _whitener(Xs[kt], shrink)
                Zw_k = (Xw[kt] - muw) @ Ww
                Zs_k = (Xs[kt] - mus) @ Ws
                C = Zw_k.T @ Zs_k / Zw_k.shape[0]
                U, rho, Vt = np.linalg.svd(C, full_matrices=False)
                k = min(k_components, U.shape[1])
                Uk, Vk = U[:, :k], Vt[:k].T
            zw = ((Xw[test_idx] - muw) @ Ww) @ Uk
            zs = ((Xs[test_idx] - mus) @ Ws) @ Vk
            scores[test_idx] = np.linalg.norm(zw - zs, axis=1)
    if not np.isfinite(scores).all():
        raise ValueError("cca_alignment_scores produced non-finite values (degenerate fold/whitening)")
    return scores
