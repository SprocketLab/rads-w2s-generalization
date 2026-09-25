#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def _pca(X: np.ndarray, k: int | None) -> np.ndarray:
    if not k or k >= X.shape[1]:
        return X
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :k] * S[:k]


def _center(X: np.ndarray, k: int | None) -> np.ndarray:
    return _pca(X - X.mean(axis=0, keepdims=True), k)


def representation_projection_scores(
    weak_acts: np.ndarray,
    strong_acts: np.ndarray,
    weak_labels: np.ndarray,
    reg: float = 0.1,
    n_components: int | None = None,
) -> np.ndarray:
    """Per-example score |[P_s (I - P_w) y_hat]_i|.

    weak_acts/strong_acts: [n, d] reps (pooled pre-lm_head activations in our runs).
    weak_labels: [n] weak labels y_hat (0/1; soft also accepted).
    reg: ridge knob, as a fraction of the average kernel eigenvalue (see below).
    n_components: optional PCA cut to imitate the paper's "principal" reps.
    Returns: [n] non-negative scores.
    """
    Xw = np.asarray(weak_acts, dtype=np.float64)
    Xs = np.asarray(strong_acts, dtype=np.float64)
    y = np.asarray(weak_labels, dtype=np.float64)
    n = Xw.shape[0]
    if Xs.shape[0] != n or y.shape[0] != n:
        raise ValueError("weak_acts, strong_acts, weak_labels must share the first dim n")

    # macOS + numpy>=2.0 fires fake overflow warnings on big matmuls (Accelerate
    # BLAS thing, results are fine, Linux is silent). The outputs are finite, so
    # the warning is muted here instead of scaring everyone in the logs.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Xw = _center(Xw, n_components)
        Xs = _center(Xs, n_components)
        Kw = Xw @ Xw.T / n
        Ks = Xs @ Xs.T / n
        eye = np.eye(n)
        # reg scales with trace/n (= average eigenvalue), so the same reg=0.1
        # means the same thing on a 896-dim weak rep and a 3584-dim strong rep.
        rw = reg * (np.trace(Kw) / n) + 1e-12
        rs = reg * (np.trace(Ks) / n) + 1e-12

        # center y first, otherwise the class prior walks in through the constant
        # direction and the score silently becomes "rank by the majority class"
        yc = y - y.mean()
        # step 1: (I - P_w) y = y - K_w (K_w + rw I)^{-1} y. solve() not inv(),
        # which keeps the O(n^3) as clean as it gets
        pw_y = Kw @ np.linalg.solve(Kw + rw * eye, yc)
        a = yc - pw_y
        # step 2: P_s a. same smoother, strong side, fitted in-sample.
        v = Ks @ np.linalg.solve(Ks + rs * eye, a)
    scores = np.abs(v)
    # the errstate above also mutes the warning that would surface a NaN, so this
    # is the only place a poisoned input still makes noise
    if not np.isfinite(scores).all():
        raise ValueError("representation_projection_scores produced non-finite values")
    return scores
