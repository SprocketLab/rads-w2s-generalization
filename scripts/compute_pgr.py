#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def _read(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"no such file: {path}")


def _acc(summary, run):
    return summary["runs"].get(run, {}).get("eval", {}).get("accuracy_3class")


# Baseline fields that tell sweeps of different pools apart.
POOL = ("dataset", "format", "weak_model", "seed", "requested_sizes")


def _pool(root):
    summary = _read(root / "baseline_seed42" / "summary.json")
    return {k: summary.get(k) for k in POOL}


def anchors(root):
    """(weak, GT, base) accuracies from the baseline of one sweep root."""
    path = root / "baseline_seed42" / "summary.json"
    baseline = _read(path)
    gt = _acc(baseline, "ground_truth")
    if gt is None:
        raise SystemExit(f"{path} has no runs.ground_truth.eval.accuracy_3class: PGR needs a "
                         "ground_truth run, from this sweep or from the one --anchors names")
    return baseline["weak_label_diagnostics"]["accuracy_3class"], gt, _acc(baseline, "base")


def sweep_pgr(root, anchor_root=None):
    """Question-level PGR = (acc - weak) / (GT - weak) of every arm of one sweep root, with the
    anchors of anchor_root when given (a sweep of the same pool that trained the GT model).

    Returns the anchors (weak, GT, base), the training seeds read, and
    {arm: (mean acc, mean PGR, sd of PGR over seeds, number of seeds)}.
    """
    if anchor_root is None:
        weak, gt, base = anchors(root)
    else:
        weak, gt, base = anchors(anchor_root)
        own, theirs = _pool(root), _pool(anchor_root)
        if own != theirs:
            differ = ", ".join(k for k in POOL if own[k] != theirs[k])
            raise SystemExit(f"{root} and {anchor_root} differ in {differ}: "
                             "--anchors must name a sweep of the same pool")
    seeds = sorted(root.glob("filtering/trainseed_*/summary.json"),
                   key=lambda p: int(p.parent.name.rsplit("_", 1)[1]))
    accs = {}
    for p in seeds:
        summary = _read(p)
        for arm in summary["runs"]:
            acc = _acc(summary, arm)
            if acc is not None:
                accs.setdefault(arm, []).append(acc)
    rows = {}
    for arm, a in accs.items():
        pgr = (np.array(a) - weak) / (gt - weak)
        rows[arm] = (float(np.mean(a)), float(pgr.mean()), float(pgr.std()), len(a))
    seed_ids = [p.parent.name.rsplit("_", 1)[1] for p in seeds]
    return (weak, gt, base), seed_ids, rows


def main():
    ap = argparse.ArgumentParser(
        description="Question-level PGR = (acc - weak) / (GT - weak) of every arm in one or more "
                    "sweep roots, as the mean ± sd over training seeds. With several roots, such as "
                    "the five Random Selection halves, each arm present in every root also gets the "
                    "mean over roots ± the sample sd across roots.")
    ap.add_argument("roots", nargs="+", type=Path, help="sweep output roots, e.g. results/table1/wanli")
    ap.add_argument("--anchors", type=Path, default=None,
                    help="read weak, GT and base from this root instead, e.g. the first random half, "
                         "whose sweep alone trained the GT model")
    args = ap.parse_args()

    per_root = []
    for root in args.roots:
        (weak, gt, base), seeds, rows = sweep_pgr(root, args.anchors)
        per_root.append(rows)
        print(f"{root}  (weak {weak:.4f}, GT {gt:.4f}, seeds {' '.join(seeds)})")
        print(f"  {'arm':32s} {'acc':>7s}  PGR")
        if base is not None:
            print(f"  {'base':32s} {base:7.4f}  {(base - weak) / (gt - weak):+.4f}")
        for arm, (acc, pgr, sd, n) in rows.items():
            print(f"  {arm:32s} {acc:7.4f}  {pgr:+.4f} ± {sd:.4f}  ({n} seeds)")
    if len(per_root) > 1:
        common = [a for a in per_root[0] if all(a in rows for rows in per_root[1:])]
        print(f"\nMean over {len(per_root)} roots ± sample sd across roots"
              + ("" if common else ": no arm is in every root"))
        for arm in common:
            acc = np.mean([rows[arm][0] for rows in per_root])
            pgr = np.array([rows[arm][1] for rows in per_root])
            print(f"  {arm:32s} {acc:7.4f}  {pgr.mean():+.4f} ± {pgr.std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
