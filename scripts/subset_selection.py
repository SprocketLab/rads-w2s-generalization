#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from contracts import SplitBundle
from ridge_maps import (
    fit_centered_ridge,
    fit_pca_ridge,
    fit_raw_ridge,
    maybe_plot,
    pca_basis,
    residual_dataframe,
    result_row,
    save_best_artifact,
    write_report,
    write_summary_csv,
)
from weak_probe import extract_final_token_activations


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def slice_activations(all_activations: torch.Tensor, sizes: dict[str, int]) -> dict[str, torch.Tensor]:
    out = {}
    start = 0
    for name in ["weak_train", "strong_train", "test"]:
        end = start + sizes[name]
        out[name] = all_activations[start:end]
        start = end
    return out


def extract_all_activations(
    model_name: str,
    texts_by_split: dict[str, list[str]],
    device: torch.device,
    dtype_arg: str,
    batch_size: int,
    max_length: int | None,
    desc: str,
    layer: str | int = "end",
    pooling: str = "last",
    answer_span: bool = False,
    answer_suffix: str = "",
    span_kind: str = "answer",
) -> dict[str, torch.Tensor]:
    sizes = {name: len(texts_by_split[name]) for name in ["weak_train", "strong_train", "test"]}
    all_texts = texts_by_split["weak_train"] + texts_by_split["strong_train"] + texts_by_split["test"]
    acts = extract_final_token_activations(
        model_name,
        all_texts,
        device,
        dtype_arg,
        batch_size,
        max_length,
        desc,
        layer=layer,
        pooling=pooling,
        answer_span=answer_span,
        answer_suffix=answer_suffix,
        span_kind=span_kind,
    )
    return slice_activations(acts, sizes)


def fit_maps(
    weak_train_acts: torch.Tensor,
    weak_strong_train_acts: torch.Tensor,
    strong_train_acts_for_map: torch.Tensor,
    strong_strong_train_acts: torch.Tensor,
    splits: SplitBundle,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
):
    x = torch.cat([weak_train_acts, weak_strong_train_acts], dim=0).float().to(device)
    y = torch.cat([strong_train_acts_for_map, strong_strong_train_acts], dim=0).float().to(device)
    train_mask = torch.zeros(x.shape[0], dtype=torch.bool, device=device)
    train_mask[: weak_train_acts.shape[0]] = True

    ridge_values = parse_float_list(args.ridge_values)
    pca_dims = parse_int_list(args.pca_dims)

    results = []
    for ridge in ridge_values:
        results.append(fit_raw_ridge(x, y, train_mask, ridge))
        results.append(fit_centered_ridge(x, y, train_mask, ridge))

    if pca_dims:
        max_pca = max(pca_dims)
        x_train = x[train_mask]
        y_train = y[train_mask]
        weak_basis = pca_basis(x_train - x_train.mean(dim=0, keepdim=True), max_pca)
        strong_basis = pca_basis(y_train - y_train.mean(dim=0, keepdim=True), max_pca)
        for dim in pca_dims:
            for ridge in ridge_values:
                results.append(fit_pca_ridge(x, y, train_mask, weak_basis, strong_basis, ridge, dim))

    rows = [result_row(result, y, train_mask, args.top_k) for result in results]
    best_key = "heldout_l2_mean" if args.best_by == "heldout_mean" else "heldout_l2_median"
    best_name = min(rows, key=lambda row: row[best_key])["name"]
    best = next(result for result in results if result.name == best_name)

    map_dir = output_dir / "map"
    map_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = args.dataset
    artifact = {
        "weak_embeddings": x.detach().cpu(),
        "strong_embeddings": y.detach().cpu(),
        "map_train_mask": train_mask.detach().cpu(),
        "weak_model": args.weak_model,
        "strong_model": args.strong_model,
        "dataset": dataset_name,
        "target_split": "paper_style_weak_train_to_strong_train",
        "pooling": "final_token",
    }
    metadata_csv = map_dir / "mapping_metadata.csv"
    write_mapping_metadata(metadata_csv, splits, dataset_name)
    write_summary_csv(rows, map_dir / "stabilized_map_summary.csv")
    (map_dir / "stabilized_map_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_report(best, rows, map_dir / "stabilized_map_report.txt", args.top_k)
    residual_dataframe(best, artifact, str(metadata_csv)).to_csv(map_dir / "best_map_residuals.csv", index=False)
    save_best_artifact(best, artifact, map_dir / "best_map_artifact.pt")
    if not args.no_plots:
        maybe_plot(rows, best, artifact, map_dir / "plots", args.top_k)

    heldout_residuals = torch.linalg.norm(best.residuals.detach().cpu().float(), dim=1)[
        ~artifact["map_train_mask"].bool()
    ].numpy()
    return best, rows, heldout_residuals


def write_mapping_metadata(path: Path, splits: SplitBundle, dataset_name: str) -> None:
    fieldnames = ["id", "dataset", "split", "label", "text", "map_train"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, split, map_train in [
            ("weak_train", splits.weak_train, 1),
            ("strong_train", splits.strong_train, 0),
        ]:
            for idx in range(len(split)):
                writer.writerow(
                    {
                        "id": split[idx]["id"],
                        "dataset": dataset_name,
                        "split": split_name,
                        "label": int(split[idx]["labels"]),
                        "text": split[idx]["txt"],
                        "map_train": map_train,
                    }
                )


def hard_weak_label_balance(indices: np.ndarray, weak_preds: np.ndarray, seed: int, size: int | None = None) -> np.ndarray:
    by_label = {
        0: indices[weak_preds[indices] == 0],
        1: indices[weak_preds[indices] == 1],
    }
    if len(by_label[0]) == 0 or len(by_label[1]) == 0:
        return indices
    if size is None:
        per_label = min(len(by_label[0]), len(by_label[1]))
    else:
        per_label = max(1, size // 2)
        per_label = min(per_label, len(by_label[0]), len(by_label[1]))
    rng = np.random.default_rng(seed)
    selected = np.concatenate(
        [
            rng.choice(by_label[0], size=per_label, replace=False),
            rng.choice(by_label[1], size=per_label, replace=False),
        ]
    )
    return selected[rng.permutation(len(selected))]


def middle_residual_indices(residuals: np.ndarray, keep_middle_frac: float) -> tuple[np.ndarray, dict]:
    if not 0.0 < keep_middle_frac <= 1.0:
        raise ValueError("--residual-keep-middle-frac must be in (0, 1].")
    order = np.argsort(residuals)
    n_keep = max(1, int(round(len(order) * keep_middle_frac)))
    start = (len(order) - n_keep) // 2
    end = start + n_keep
    kept = order[start:end]
    kept_scores = residuals[kept]
    summary = {
        "matched_examples": int(len(order)),
        "kept_examples": int(len(kept)),
        "keep_middle_frac": float(keep_middle_frac),
        "dropped_low_examples": int(start),
        "dropped_high_examples": int(len(order) - end),
        "kept_residual_min": float(np.min(kept_scores)),
        "kept_residual_max": float(np.max(kept_scores)),
        "kept_residual_mean": float(np.mean(kept_scores)),
        "kept_residual_median": float(np.median(kept_scores)),
    }
    return kept, summary


def random_balanced_indices(weak_preds: np.ndarray, size: int, seed: int) -> np.ndarray:
    all_indices = np.arange(len(weak_preds))
    return hard_weak_label_balance(all_indices, weak_preds, seed, size=size)


def subset_summary(
    indices: np.ndarray,
    labels: np.ndarray,
    weak_probs: np.ndarray,
    residuals: np.ndarray,
) -> dict[str, float]:
    weak_preds = (weak_probs >= 0.5).astype(int)
    return {
        "n": int(len(indices)),
        "true_label_mean": float(np.mean(labels[indices])) if len(indices) else math.nan,
        "weak_label_mean": float(np.mean(weak_preds[indices])) if len(indices) else math.nan,
        "weak_label_accuracy": float(np.mean(weak_preds[indices] == labels[indices])) if len(indices) else math.nan,
        "weak_soft_prob_mean": float(np.mean(weak_probs[indices])) if len(indices) else math.nan,
        "residual_l2_mean": float(np.mean(residuals[indices])) if len(indices) else math.nan,
        "residual_l2_median": float(np.median(residuals[indices])) if len(indices) else math.nan,
    }
