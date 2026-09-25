#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from screen_mlpstep2 import _spearman, insample_mlp_fit_read


def main() -> None:
    ap = argparse.ArgumentParser(description="Write the mlpstep2 score as a custom-scores npz (slot cs1).")
    ap.add_argument("--acts", required=True, help="npz with weak, strong, weak_preds (DUMP_ACTS)")
    ap.add_argument("--out", required=True, help="npz to write: weak_preds, cs1")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--wd", type=float, default=10.0)
    ap.add_argument("--reg", type=float, default=0.1, help="kernel ridge reg of the step-1 residual")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    d = np.load(args.acts)
    Xw = np.asarray(d["weak"], np.float64)
    Xs = np.asarray(d["strong"], np.float64)
    wp = np.asarray(d["weak_preds"]).astype(int)
    n = len(wp)
    yc = wp.astype(np.float64) - wp.mean()
    # step 1 is rp's kernel ridge residual; step 2 is the capped in-sample MLP read of it
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        Xwc = Xw - Xw.mean(0, keepdims=True)
        Kw = Xwc @ Xwc.T / n
        rw = args.reg * (np.trace(Kw) / n) + 1e-12
        a = yc - Kw @ np.linalg.solve(Kw + rw * np.eye(n), yc)
        Xsc = Xs - Xs.mean(0, keepdims=True)
        Ks = Xsc @ Xsc.T / n
        rs = args.reg * (np.trace(Ks) / n) + 1e-12
        v_ref = Ks @ np.linalg.solve(Ks + rs * np.eye(n), a)
    if not np.isfinite(a).all():
        raise ValueError("non-finite step-1 residual")
    v = insample_mlp_fit_read(np.asarray(Xs, np.float32), a, args.hidden, args.wd, seed=args.seed, epochs=args.epochs)
    s = np.abs(v)
    if not np.isfinite(s).all():
        raise ValueError("score_mlpstep2 produced non-finite scores")
    corr_va = float(np.corrcoef(v, a)[0, 1])
    print(f"n={n} corr(v,a)={corr_va:.3f} spearman(|v|,|a|)={_spearman(s, np.abs(a)):.3f} "
          f"spearman(|v|,|v_ref|)={_spearman(s, np.abs(v_ref)):.3f} std={s.std():.3f}")
    np.savez(args.out, weak_preds=wp, cs1=s)


if __name__ == "__main__":
    main()
