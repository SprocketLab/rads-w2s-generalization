#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np


def linear_cka(weak_acts: np.ndarray, strong_acts: np.ndarray) -> float:
    """Linear CKA between the weak and the strong representations of the same points (rows)."""
    Xw = np.asarray(weak_acts, dtype=np.float64)
    Xs = np.asarray(strong_acts, dtype=np.float64)
    Xw = Xw - Xw.mean(0, keepdims=True)
    Xs = Xs - Xs.mean(0, keepdims=True)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        cka = np.linalg.norm(Xs.T @ Xw) ** 2 / (np.linalg.norm(Xw.T @ Xw) * np.linalg.norm(Xs.T @ Xs))
    if not np.isfinite(cka):
        raise ValueError("linear_cka produced a non-finite value")
    return float(cka)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Print the linear CKA of the weak and strong pool representations in DUMP_ACTS npz files.")
    ap.add_argument("acts", nargs="+", help="npz written by DUMP_ACTS (keys weak, strong)")
    args = ap.parse_args()
    for path in args.acts:
        d = np.load(path)
        print(f"{path}: CKA {linear_cka(d['weak'], d['strong']):.4f}")


if __name__ == "__main__":
    main()
