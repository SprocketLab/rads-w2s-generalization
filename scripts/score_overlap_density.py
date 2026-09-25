#!/usr/bin/env python3
from __future__ import annotations

import numpy as np


def l2_change_point(sorted_scores: np.ndarray) -> int:
    """Index k splitting a sorted array into [:k] and [k:] with the least total within-segment
    squared error. This is what Binseg(model="l2").predict(n_bkps=1) returns, up to that
    package's jump=5 candidate grid; a threshold is then sorted_scores[k]."""
    s = np.asarray(sorted_scores, dtype=np.float64)
    n = len(s)
    if n < 4:
        raise ValueError("change point needs at least 4 points")
    cs, cs2 = np.cumsum(s), np.cumsum(s * s)
    k = np.arange(2, n - 1)
    left = cs2[k - 1] - cs[k - 1] ** 2 / k
    right = (cs2[-1] - cs2[k - 1]) - (cs[-1] - cs[k - 1]) ** 2 / (n - k)
    return int(k[np.argmin(left + right)])


def overlap_partition(
    weak_probs: np.ndarray,
    strong_acts: np.ndarray,
    hard_frac="auto",
    metric: str = "cos",
    seed: int = 0,
    chunk: int = 1024,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (hard_idx, closeness, is_hard). closeness is defined on every row; on hard
    rows it is the closeness to the other hard rows, kept only for diagnostics."""
    p = np.asarray(weak_probs, dtype=np.float64).ravel()
    X = np.asarray(strong_acts, dtype=np.float64)
    n = len(p)
    if X.shape[0] != n:
        raise ValueError("weak_probs and strong_acts must share the first dim n")
    conf = 2.0 * np.abs(p - 0.5)
    # ties (many rows at confidence 1.0) are broken by a seeded permutation, not by row order
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.permutation(n), conf))
    if hard_frac == "auto":
        # the reference code: one change point on the sorted confidences; the tail below it is
        # hard. Its two detectors disagree by the boundary element (<= vs <); the split's own
        # left segment, exactly k rows, is used here.
        n_hard = l2_change_point(conf[order])
    else:
        if not 0.0 < float(hard_frac) < 1.0:
            raise ValueError("hard_frac must be 'auto' or inside (0, 1)")
        n_hard = max(1, int(round(float(hard_frac) * n)))
    hard_idx = np.sort(order[:n_hard])
    is_hard = np.zeros(n, dtype=bool)
    is_hard[hard_idx] = True

    # macOS + numpy>=2 raises fake overflow warnings on these matmuls (Accelerate BLAS); the
    # finiteness check at the end is the real guard
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return _closeness(X, hard_idx, is_hard, metric, chunk)


def _closeness(X, hard_idx, is_hard, metric, chunk):
    n = X.shape[0]
    if metric == "cos":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Xn = X / norms
        H = Xn[hard_idx]
        closeness = np.empty(n, dtype=np.float64)
        for s in range(0, n, chunk):
            closeness[s:s + chunk] = np.abs(Xn[s:s + chunk] @ H.T).max(axis=1)
    elif metric == "euclid":
        H = X[hard_idx]
        hh = (H * H).sum(axis=1)
        closeness = np.empty(n, dtype=np.float64)
        for s in range(0, n, chunk):
            B = X[s:s + chunk]
            d2 = (B * B).sum(axis=1)[:, None] + hh[None, :] - 2.0 * B @ H.T
            d = np.sqrt(np.clip(d2, 0.0, None))
            closeness[s:s + chunk] = 1.0 / (1.0 + d.min(axis=1))
    else:
        raise ValueError(f"metric must be 'cos' or 'euclid', got {metric!r}")
    return hard_idx, closeness, is_hard


def overlap_scores(
    weak_probs: np.ndarray,
    strong_acts: np.ndarray,
    hard_frac="auto",
    metric: str = "cos",
    seed: int = 0,
) -> np.ndarray:
    """[n] non-negative float64. Confident rows: 1 + closeness to the hard tail. Hard rows: 0."""
    hard_idx, closeness, is_hard = overlap_partition(weak_probs, strong_acts, hard_frac, metric, seed)
    scores = np.where(is_hard, 0.0, 1.0 + closeness)
    if not np.isfinite(scores).all():
        raise ValueError("overlap_scores produced non-finite output")
    return scores
