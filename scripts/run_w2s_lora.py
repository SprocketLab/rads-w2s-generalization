#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from weak_probe import (
    extract_final_token_activations,
    fit_probe,
    predict_probe,
    resolve_device,
)
from subset_selection import (
    extract_all_activations,
    fit_maps,
    hard_weak_label_balance,
    middle_residual_indices,
    random_balanced_indices,
    subset_summary,
)
from lora_train_eval import (
    clear_memory,
    evaluate_yes_no,
    train_lora_model,
    write_predictions,
)
from score_rads_lin import representation_projection_scores
from contracts import SEED_OFFSETS, LoraExample
from score_cca import cca_alignment_scores
from score_rads import mlp_step1_rp_scores, linstep1_rp_scores
from score_excess_loss import excess_loss_scores
from score_overlap_density import overlap_scores
from score_token_length import token_length_scores
from data_preprocessing import (
    generic_mc_eval,
    csqa_rows,
    agnews_rows,
    blimp_rows,
    imdb_rows,
    qnli_rows,
    cosmosqa_rows,
    sst2_rows,
    tweetsent_rows,
    wikitoxic_rows,
    scitail_rows,
    boolq_rows,
    ethicsjustice_rows,
    quail_rows,
    riddlesense_rows,
    scinli_rows,
    load_anli_multichoice_eval,
    load_aquarat_multichoice_eval,
    load_fevernli_multichoice_eval,
    load_hellaswag_multichoice_eval,
    load_imppres_multichoice_eval,
    load_paper_style_splits,
    load_sciq_multichoice_eval,
    load_wanli_multichoice_eval,
    split_labels,
    split_texts,
    to_lora_examples,
)
from reporting import (
    accuracy_3class,
    build_summary,
    auroc_from_rows,
    eval_rows_from_probs,
    format_summary,
    prior_matched_accuracy,
    summarize_train_subset,
    write_eval3_rows,
    write_knn_diagnostics,
    write_strong_train_labels,
    write_subset_csv,
    write_text_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sciq", "anli", "hellaswag", "wanli", "fevernli", "imppres", "aquarat", "quail", "riddlesense", "scinli", "csqa", "cosmosqa", "blimp", "sst2", "qnli", "agnews", "imdb", "tweetsent", "wikitoxic", "scitail", "boolq", "ethicsjustice"], default="wanli")
    parser.add_argument(
        "--anli-round",
        choices=["r1", "r2", "r3"],
        default="r1",
        help="For ANLI only: which adversarial round to use (default r1).",
    )
    parser.add_argument(
        "--sciq-use-support",
        action="store_true",
        help=(
            "For SciQ only, include the support passage in the prompt. "
            "Default false matches the original datacentric_w2s SciQ formatter."
        ),
    )
    parser.add_argument("--weak-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--strong-model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--output-dir", default="results/w2s_lora/seed42")
    parser.add_argument("--n-train", type=int, default=6_000)
    parser.add_argument("--n-val", type=int, default=1_000)
    parser.add_argument("--n-test", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="Optional seed for LoRA initialization/training only. Keeps data splits and filters fixed.",
    )
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--activation-max-length", type=int, default=None)
    parser.add_argument("--answer-suffix", default="\nIs the candidate answer correct? Answer:")
    parser.add_argument("--weak-batch-size", type=int, default=4)
    parser.add_argument("--activation-batch-size", type=int, default=1)
    parser.add_argument("--l2-penalty", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=10_000)
    parser.add_argument("--strong-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=0,
                        help="Batch size of the yes/no test-set evaluations; 0 uses --strong-batch-size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=100)
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="If >0, train each run for this many FULL passes over its actual subset "
        "(no-filter -> all data, f50 -> the real 50%%), setting max_train_steps per run = "
        "epochs * ceil(len(subset)/(batch*grad_accum)). Overrides the fixed compute-matched "
        "--max-train-steps cap.",
    )
    parser.add_argument("--dump-acts", default="",
                        help="npz path: dump strong_train weak/strong reps + weak preds/probs + gt "
                             "for offline method development (alignment scores, diagnostics)")
    parser.add_argument("--el-batch", type=int, default=0,
                        help="batch for the excess-loss option pass; 0 = max(strong batch, 8)")
    parser.add_argument("--custom-scores-npz", default="",
                        help="npz with precomputed selection scores (keys cs1..cs4 + weak_preds guard)")
    parser.add_argument("--dump-el", default="",
                        help="npz path: dump per-example elloss, weak pick and gold option + weak "
                             "preds/probs + gt; forces el computation")
    parser.add_argument("--mlpstep1-wd", type=float, default=30.0,
                        help="weight decay for the mlpstep1 weak-side MLP (sweep peak = 30)")
    parser.add_argument("--rp-reg", type=float, default=0.1,
                        help="Kernel ridge reg for the representation-projection score (rp_high/rp_low).")
    parser.add_argument("--rp-components", type=int, default=0,
                        help="PCA components for the representation-projection score (0 = use all).")
    parser.add_argument("--score-base-on-train", action="store_true",
                        help="Also score the untuned strong (base) model's yes/no P(correct) on each "
                             "strong_train example and write it to strong_train_labels.csv "
                             "(base_prob_correct / base_confidence). One extra base forward pass.")
    parser.add_argument("--representation-layer", choices=["end", "middle"], default="end",
                        help="Transformer layer whose final-token activations feed the kNN and RP "
                             "selectors. 'end' (default) = last hidden layer. 'middle' = "
                             "len(hidden_states)//2, a training control. The weak probe / confidence "
                             "and the L2-residual maps always stay on the end layer so the weak "
                             "labels are held fixed across the comparison.")
    parser.add_argument("--representation-pooling", choices=["last", "mean"], default="last",
                        help="Token pooling for the kNN/RP representations: 'last' (default) = the "
                             "final non-pad token; 'mean' = average over the non-pad tokens (a "
                             "control). The weak probe stays on last-token regardless.")
    parser.add_argument("--representation-span", choices=["full", "answer", "context"], default="full",
                        help="Which tokens of each prompt the kNN/RP representation pools over: "
                             "'full' (default, settled choice) = the whole prompt (question + answer); "
                             "'answer' = only 'A: {candidate}' (mask the question prefix and the yes/no "
                             "suffix); 'context' = everything up to and including the 'A:' marker but "
                             "EXCLUDING the candidate word (Premise/Hypothesis/Q/A:). 'answer'/'context' "
                             "need a fast tokenizer. The weak probe/confidence stay on the full prompt "
                             "so the weak labels are held fixed across the comparison.")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--lr-decay", choices=["linear", "none"], default="linear",
                        help="LR schedule after warmup: linear decay to 0 (default) or constant.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-adapters", action="store_true")
    parser.add_argument(
        "--runs",
        default="base,ground_truth,weak_label,middle_balanced",
        help=(
            "Comma-separated run names. Anchors: base, ground_truth, weak_label. "
            "Selection arms are <score>_<band>_balanced_f<percent>: mlpstep1_high_balanced_f50, "
            "rp_high_balanced_f50, elloss_high_balanced_f50, confidence_high_balanced_f50, "
            "knn_high_balanced_f50, cca_high_balanced_f50, ccarobust_high_balanced_f50, "
            "overlap_high_balanced_f50, toklen_high_balanced_f50, random_balanced_f50, and "
            "the same with _low_; the percent follows --committee-keep-fracs. "
            "requested_runs lists every accepted name."
        ),
    )
    parser.add_argument("--residual-keep-middle-frac", type=float, default=0.5)
    parser.add_argument("--confidence-keep-frac", type=float, default=0.5)
    parser.add_argument("--knn-k", type=int, default=20)
    parser.add_argument("--knn-keep-middle-frac", type=float, default=0.5)
    parser.add_argument("--knn-mixed-center", type=float, default=0.5)
    parser.add_argument(
        "--knn-reference-cross-fit",
        action="store_true",
        help=(
            "Label weak_train reference points weak-correct/weak-wrong using "
            "cross-fitted (out-of-fold) weak-probe predictions instead of "
            "in-sample predictions. Avoids the kNN saturation that happens when "
            "the probe is near-perfect on its own training set."
        ),
    )
    parser.add_argument("--cross-fit-folds", type=int, default=5)
    parser.add_argument(
        "--committee-members",
        type=int,
        default=8,
        help=(
            "Number of bootstrap weak probes in the query-by-committee disagreement "
            "selector (committee_agree / committee_disagree)."
        ),
    )
    parser.add_argument("--committee-keep-frac", type=float, default=0.5)
    parser.add_argument(
        "--committee-keep-fracs",
        default="",
        help=(
            "Comma-separated keep fractions for a committee selection sweep, e.g. "
            "'0.1,0.2,0.3,0.45,0.6,0.8'. For each fraction f (pct=round(100f)) it adds runs "
            "committee_agree_balanced_f{pct}, committee_agree_unbalanced_f{pct}, "
            "committee_disagree_balanced_f{pct}, and a matched random_balanced_f{pct}. "
            "Empty = no sweep."
        ),
    )
    parser.add_argument(
        "--purity-grid",
        default="",
        help=(
            "Comma-separated target purities in percent (e.g. 50,60,80,90,100; the token "
            "'nat' targets the subset's own natural purity) for the purity_grid runs: one "
            "random subset at the random-control budget, labels set to each target purity. "
            "Diagnostic; uses gold."
        ),
    )
    parser.add_argument(
        "--purity-mode",
        choices=["relabel", "iid"],
        default="relabel",
        help=(
            "relabel: start from the natural weak labels and flip toward/away from gold "
            "(errors stay structured). iid: start from gold and corrupt uniformly chosen "
            "rows (errors are position-random at every purity)."
        ),
    )
    parser.add_argument("--purity-draws", type=int, default=1,
                        help="independent corruption draws per purity target (iid mode)")
    parser.add_argument("--random-control-size", type=int, default=None)
    parser.add_argument("--ridge-values", default="100.0")
    parser.add_argument("--pca-dims", default="")
    parser.add_argument("--best-by", choices=["heldout_mean", "heldout_median"], default="heldout_median")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--save-activations", action="store_true")
    parser.add_argument(
        "--eval-3class",
        action="store_true",
        help="Also report per-question multiple-choice accuracy: score every candidate "
        "answer of each test question and argmax P(correct).",
    )
    parser.add_argument(
        "--n-eval-questions",
        type=int,
        default=2041,
        help="Number of test questions for --eval-3class (each expands to its candidate answers).",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT"),
        help="Send each arm to this Weights & Biases project as it finishes. Left unset, "
        "the run never imports wandb, so the training path is untouched.",
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument(
        "--wandb-group",
        default=os.environ.get("WANDB_GROUP"),
        help="Groups the arms of one sweep together in the wandb UI. Defaults to the "
        "output directory name, which is already unique per sweep.",
    )
    return parser.parse_args()


def requested_runs(args: argparse.Namespace) -> list[str]:
    runs = [item.strip() for item in args.runs.split(",") if item.strip()]
    allowed = {
        "base",
        "ground_truth",
        "weak_label",
        "purity_grid",
        "purity_natural",
        "middle_unbalanced",
        "middle_balanced",
        "confidence_middle_unbalanced",
        "confidence_middle_balanced",
        "confidence_high_unbalanced",
        "confidence_high_balanced",
        "knn_middle_unbalanced",
        "knn_middle_balanced",
        "knn_mixed_unbalanced",
        "knn_mixed_balanced",
        "knn_high_unbalanced",
        "knn_high_balanced",
        "committee_agree_unbalanced",
        "committee_agree_balanced",
        "committee_disagree_unbalanced",
        "committee_disagree_balanced",
    }
    def _is_frac_run(name: str) -> bool:
        for pref in (
            "committee_agree_balanced_f",
            "committee_agree_unbalanced_f",
            "committee_disagree_balanced_f",
            "knn_high_balanced_f",
            "knn_low_balanced_f",
            "confidence_high_balanced_f",
            "confidence_low_balanced_f",
            "rp_high_balanced_f",
            "rp_low_balanced_f",
            "cca_high_balanced_f",
            "cca_low_balanced_f",
            "mlpstep1_high_balanced_f",
            "mlpstep1_low_balanced_f",
            "linstep1_high_balanced_f",
            "linstep1_low_balanced_f",
            "cs1_high_balanced_f", "cs1_low_balanced_f",
            "cs2_high_balanced_f", "cs2_low_balanced_f",
            "cs3_high_balanced_f", "cs3_low_balanced_f",
            "cs4_high_balanced_f", "cs4_low_balanced_f",
            "ccarobust_high_balanced_f",
            "ccarobust_low_balanced_f",
            "l2resid_high_balanced_f",
            "l2resid_low_balanced_f",
            "elloss_high_balanced_f",
            "elloss_low_balanced_f",
            "overlap_high_balanced_f",
            "overlap_low_balanced_f",
            "toklen_high_balanced_f",
            "toklen_low_balanced_f",
            "random_balanced_f",
            "random_gt_balanced_f",
            "weak_correct_only_balanced_f",
        ):
            if name.startswith(pref) and name[len(pref):].isdigit():
                return True
        return False

    unknown = sorted(r for r in set(runs) - allowed if not _is_frac_run(r))
    if unknown:
        raise SystemExit(f"Unknown run(s): {', '.join(unknown)}")
    return runs


def score_band_indices(scores: np.ndarray, keep_frac: float, mode: str) -> tuple[np.ndarray, dict[str, float | str]]:
    if not 0.0 < keep_frac <= 1.0:
        raise ValueError("keep_frac must be in (0, 1].")
    if mode not in {"middle", "high", "low"}:
        raise ValueError(f"Unknown score band mode: {mode}")

    order = np.argsort(scores)
    n_keep = max(1, int(round(len(order) * keep_frac)))
    if mode == "middle":
        start = (len(order) - n_keep) // 2
        end = start + n_keep
        kept = order[start:end]
        dropped_low = start
        dropped_high = len(order) - end
    elif mode == "low":
        kept = order[:n_keep]
        dropped_low = 0
        dropped_high = len(order) - n_keep
    else:
        kept = order[-n_keep:]
        dropped_low = len(order) - n_keep
        dropped_high = 0

    kept_scores = scores[kept]
    return kept, {
        "mode": mode,
        "matched_examples": int(len(order)),
        "kept_examples": int(len(kept)),
        "keep_frac": float(keep_frac),
        "dropped_low_examples": int(dropped_low),
        "dropped_high_examples": int(dropped_high),
        "kept_score_min": float(np.min(kept_scores)),
        "kept_score_max": float(np.max(kept_scores)),
        "kept_score_mean": float(np.mean(kept_scores)),
        "kept_score_median": float(np.median(kept_scores)),
    }


def score_closest_indices(
    scores: np.ndarray,
    keep_frac: float,
    center: float,
    mode_name: str,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Keep examples whose score is closest to a target value.

    For kNN filtering, the intended "overlap-like" points are not
    necessarily the middle quantile of the score distribution. They are points
    whose neighborhoods are mixed between weak-correct and weak-wrong reference
    examples. With a correct-neighbor-rate score, that means closest to 0.5.
    """

    if not 0.0 < keep_frac <= 1.0:
        raise ValueError("keep_frac must be in (0, 1].")

    order = np.argsort(np.abs(scores - center))
    n_keep = max(1, int(round(len(order) * keep_frac)))
    kept = order[:n_keep]
    kept_scores = scores[kept]
    return kept, {
        "mode": mode_name,
        "matched_examples": int(len(order)),
        "kept_examples": int(len(kept)),
        "keep_frac": float(keep_frac),
        "center": float(center),
        "kept_score_min": float(np.min(kept_scores)),
        "kept_score_max": float(np.max(kept_scores)),
        "kept_score_mean": float(np.mean(kept_scores)),
        "kept_score_median": float(np.median(kept_scores)),
        "kept_abs_distance_to_center_mean": float(np.mean(np.abs(kept_scores - center))),
        "kept_abs_distance_to_center_median": float(np.median(np.abs(kept_scores - center))),
    }


def cross_fitted_weak_probs(
    activations: torch.Tensor,
    labels: np.ndarray,
    folds: int,
    l2_penalty: float,
    max_iter: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """Out-of-fold weak-probe probabilities on the weak_train reference set.

    Each weak_train point is scored by a probe that was NOT trained on it: the
    set is split into ``folds`` folds, and for each fold a probe is fit on the
    other folds and used to predict the held-out fold. This gives honest
    weak-correct / weak-wrong reference labels for the kNN filter and avoids the
    in-sample saturation that occurs when the probe predicts its own training
    data (an overfit probe labels almost every reference point weak-correct,
    leaving no weak-wrong neighbors for the mixed-neighborhood signal).
    """
    n = activations.shape[0]
    folds = max(2, min(folds, n))
    order = np.random.default_rng(seed).permutation(n)
    labels_t = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
    probs = np.zeros(n, dtype=np.float32)
    for held_out in np.array_split(order, folds):
        held_out_t = torch.as_tensor(held_out, dtype=torch.long)
        train_mask = torch.ones(n, dtype=torch.bool)
        train_mask[held_out_t] = False
        train_idx_t = torch.nonzero(train_mask, as_tuple=False).squeeze(1)
        probe = fit_probe(
            activations[train_idx_t],
            labels_t[train_idx_t],
            l2_penalty,
            max_iter,
            device,
        )
        probs[held_out] = predict_probe(probe, activations[held_out_t], device)
    return probs


def committee_disagreement_on_strong_train(
    weak_train_activations: torch.Tensor,
    weak_train_labels: np.ndarray,
    strong_train_activations: torch.Tensor,
    members: int,
    l2_penalty: float,
    max_iter: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Query-by-committee disagreement for each strong_train point.

    Trains ``members`` weak probes on bootstrap resamples of weak_train and
    predicts every strong_train point with each one. The committee's per-point
    disagreement (standard deviation of the member probabilities) operationalizes
    the active-learning "region of disagreement": high disagreement means the
    plausible weak probes cannot agree on the label (boundary / unreliable weak
    label), low disagreement means they concur (reliable weak label). strong_train
    is out-of-sample for every member, so the estimate is honest by construction.
    """
    n = weak_train_activations.shape[0]
    members = max(2, members)
    rng = np.random.default_rng(seed)
    labels_t = torch.as_tensor(np.asarray(weak_train_labels), dtype=torch.float32)
    member_probs = []
    for _ in range(members):
        boot = rng.integers(0, n, size=n)
        boot_t = torch.as_tensor(boot, dtype=torch.long)
        probe = fit_probe(
            weak_train_activations[boot_t],
            labels_t[boot_t],
            l2_penalty,
            max_iter,
            device,
        )
        member_probs.append(predict_probe(probe, strong_train_activations, device))
    stacked = np.stack(member_probs, axis=0)
    mean_prob = stacked.mean(axis=0)
    disagreement = stacked.std(axis=0)
    return mean_prob, disagreement


def compute_weak_train_knn_stats(
    reference_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    reference_weak_correct: np.ndarray,
    k: int,
) -> dict[str, np.ndarray]:
    if k <= 0:
        raise ValueError("--knn-k must be positive.")
    if len(reference_embeddings) == 0:
        raise ValueError("Need at least one weak_train reference embedding.")
    k = min(k, len(reference_embeddings))

    ref = torch.nn.functional.normalize(reference_embeddings.to(torch.float32), p=2, dim=1)
    query = torch.nn.functional.normalize(query_embeddings.to(torch.float32), p=2, dim=1)
    sims = query @ ref.T
    top_values, top_indices = torch.topk(sims, k=k, dim=1)

    correct = torch.tensor(reference_weak_correct.astype(np.float32))
    neighbor_correct = correct[top_indices]
    correct_count = neighbor_correct.sum(dim=1)
    correct_rate = correct_count / float(k)
    return {
        "neighbor_indices": top_indices.cpu().numpy(),
        "neighbor_cosine_mean": top_values.mean(dim=1).cpu().numpy(),
        "neighbor_cosine_min": top_values.min(dim=1).values.cpu().numpy(),
        "knn_correct_count": correct_count.cpu().numpy(),
        "knn_correct_rate": correct_rate.cpu().numpy(),
    }


def log_arm_to_wandb(
    args: argparse.Namespace, run_name: str, report: dict, output_dir: Path
) -> None:
    """Send one finished arm to wandb, or do nothing when no project is configured.

    An arm is the unit a sweep compares, so that is what I log; a 200-step curve
    arriving every couple of minutes is what makes a running sweep watchable
    without touching the training loop. PGR is absent on purpose: it needs the weak
    and ground-truth anchors, which a filtering invocation does not hold.

    A logging failure must never cost a finished arm, hence the blanket catch.
    """
    if not args.wandb_project:
        return
    try:
        import wandb

        eval_summary = report.get("eval") or {}
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"{output_dir.name}/{run_name}@seed{args.train_seed}",
            group=args.wandb_group or output_dir.name,
            job_type=run_name,
            config={
                "arm": run_name,
                "dataset": args.dataset,
                "weak_model": args.weak_model,
                "strong_model": args.strong_model,
                "data_seed": args.seed,
                "train_seed": args.train_seed,
                "lr": args.lr,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "max_length": args.max_length,
                "train_examples": report.get("train_examples"),
            },
            tags=["live", args.dataset],
            reinit=True,
        )
        for step, loss in enumerate(report.get("losses") or [], start=1):
            wandb.log({"train_loss": float(loss)}, step=step)
        for key in ("accuracy", "auroc", "accuracy_prior_matched", "accuracy_3class"):
            if key in eval_summary:
                wandb.summary[key] = eval_summary[key]
        if "elapsed_sec" in report:
            wandb.summary["elapsed_sec"] = report["elapsed_sec"]
        run.finish()
    except Exception as exc:  # noqa: BLE001
        print(f"[wandb] skipped {run_name}: {exc}")


def run_lora_eval(
    args: argparse.Namespace,
    run_name: str,
    train_examples: list[LoraExample],
    train_labels: list[int],
    eval_examples: list[LoraExample],
    output_dir: Path,
    eval3_examples: list[LoraExample] | None = None,
    curriculum_order=None,
    sample_weights=None,
    max_train_steps: int | None = None,
) -> tuple[dict, list[dict]]:
    if args.train_seed is not None:
        torch.manual_seed(args.train_seed)
        np.random.seed(args.train_seed)
        random.seed(args.train_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.train_seed)
    if max_train_steps is not None:
        # train_lora_model reads the step cap off the namespace; give it a copy
        # rather than mutating the shared one
        args = argparse.Namespace(**{**vars(args), "max_train_steps": max_train_steps})
    model, tokenizer, report = train_lora_model(
        args, train_examples, train_labels, run_name, output_dir,
        curriculum_order=curriculum_order, sample_weights=sample_weights,
    )
    eval_summary, rows = evaluate_yes_no(
        model,
        tokenizer,
        eval_examples,
        args.eval_batch_size or args.strong_batch_size,
        args.device,
        args.max_length,
        f"eval {run_name}",
    )
    eval_summary["auroc"] = auroc_from_rows(rows)
    eval_summary["accuracy_prior_matched"] = prior_matched_accuracy(rows)
    if eval3_examples:
        _, rows3 = evaluate_yes_no(
            model,
            tokenizer,
            eval3_examples,
            args.eval_batch_size or args.strong_batch_size,
            args.device,
            args.max_length,
            f"eval3 {run_name}",
        )
        eval_summary["accuracy_3class"] = accuracy_3class(eval3_examples, rows3)
        write_eval3_rows(output_dir / f"eval3_{run_name}.csv", eval3_examples, rows3)
    del model
    del tokenizer
    clear_memory()
    return {
        **report,
        "train_seed": args.train_seed,
        "train_subset": summarize_train_subset(train_examples, train_labels),
        "eval": eval_summary,
    }, rows


def _load_causal_lm_for_scoring(model_name: str, torch_dtype: str, device):
    """Load an arbitrary model name as a causal LM + tokenizer for yes/no scoring."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from lora_train_eval import resolve_dtype

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Our prompt puts the question, the candidate answer and the yes/no cue at the END,
    # so right truncation (the tokenizer default) would cut exactly the part that
    # distinguishes one candidate from another. Truncate the passage instead.
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=resolve_dtype(torch_dtype), low_cpu_mem_usage=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.to(device)
    return model, tokenizer


def compute_el_kway_scores(
    args: argparse.Namespace, strong_train_ds, device
) -> dict:
    """Excess loss per strong_train question (in row order): loss_strong - loss_weak at
    the weak model's own pick, where each model's option distribution is its yes/no
    P(correct) over the K answer options, normalized (score_excess_loss). Loads the weak
    and the untuned-strong model once each (no probe); independent of LoRA training, so
    it depends on the data seed alone and serves both bands.

    Returns {"elloss", "weak_pick", "gold_opt"}: the score, the weak model's pick and the
    gold option, each row-aligned to strong_train.
    """
    from lora_train_eval import load_strong_model_and_tokenizer

    if "mc_options" not in strong_train_ds.column_names:
        raise SystemExit(
            "The excess-loss selectors need per-question options; this dataset's "
            "formatter does not emit 'mc_options'. Add it to the format_* function before "
            "requesting them."
        )
    source_ids = list(strong_train_ds["source_id"])
    mc_options = list(strong_train_ds["mc_options"])
    opt_examples = [
        LoraExample(
            id=f"{src}::opt{k}",
            source_id=src,
            text=str(otext),
            label=0,
            answer_suffix=args.answer_suffix,
        )
        for src, opts in zip(source_ids, mc_options)
        for k, otext in enumerate(opts)
    ]
    # The K-option pass defaults to batches of at least 8, larger than the training batch, so on
    # long prompts it can OOM where training does not. Let the caller cap it.
    el_batch = int(args.el_batch) if args.el_batch else max(int(args.strong_batch_size), 8)

    def _option_probs_by_q(model, tokenizer, desc):
        _, rows = evaluate_yes_no(
            model, tokenizer, opt_examples, el_batch, device, args.max_length, desc
        )
        by_q: dict[str, list[float]] = {}
        for ex, row in zip(opt_examples, rows):
            by_q.setdefault(ex.source_id, []).append(float(row["prob_label1"]))
        return by_q

    weak_model, weak_tok = _load_causal_lm_for_scoring(args.weak_model, args.torch_dtype, device)
    weak_by_q = _option_probs_by_q(weak_model, weak_tok, "el K-option (weak)")
    del weak_model
    clear_memory()

    strong_model, strong_tok = load_strong_model_and_tokenizer(args, trainable_lora=False)
    strong_by_q = _option_probs_by_q(strong_model, strong_tok, "el K-option (base strong)")
    del strong_model
    clear_memory()

    mc_correct = (list(strong_train_ds["mc_correct"])
                  if "mc_correct" in strong_train_ds.column_names else [-1] * len(source_ids))
    n_options = {src: len(opts) for src, opts in zip(source_ids, mc_options)}
    elloss_by_q, pick_by_q = excess_loss_scores(weak_by_q, strong_by_q, n_options)
    return {
        "elloss": np.asarray([elloss_by_q.get(src, 0.0) for src in source_ids], dtype=np.float64),
        "weak_pick": np.asarray([pick_by_q.get(src, -1) for src in source_ids], dtype=np.int64),
        "gold_opt": np.asarray(mc_correct, dtype=np.int64),
    }


def build_keep_fraction_arms(
    *,
    seed: int,
    run_subsets: dict,
    scores: dict,
    custom_scores: dict,
    committee_keep_fracs: list[float],
    committee_disagreement_strong: np.ndarray,
    weak_confidences_strong: np.ndarray,
    weak_preds_strong: np.ndarray,
    weak_labels: list,
    strong_examples: list,
    residuals: np.ndarray,
    knn_stats: dict,
    random_balanced_indices: np.ndarray,
    n_strong_examples: int,
) -> None:
    """Add the per-keep-fraction arms (the _f<pct> family) to run_subsets in place.

    The loop mutates run_subsets in place and returns nothing.
    The seeds are offset by the enumerate index fi, so the iteration order over
    committee_keep_fracs is part of the contract; do not sort or parallelize it.
    """
    rp_scores = scores.get('rp')
    elloss_scores = scores.get('elloss')
    cca_scores = scores.get('cca')
    cca_robust_scores = scores.get('cca_robust')
    mlpstep1_scores = scores.get('mlpstep1')
    linstep1_scores = scores.get('linstep1')
    overlap_scores_arr = scores.get('overlap')
    toklen_scores = scores.get('toklen')

    gold_strong = np.asarray([int(e.label) for e in strong_examples])
    for fi, frac in enumerate(committee_keep_fracs):
        pct = int(round(frac * 100))
        agree_idx, _ = score_band_indices(committee_disagreement_strong, frac, "low")
        disagree_idx, _ = score_band_indices(committee_disagreement_strong, frac, "high")
        agree_bal = hard_weak_label_balance(agree_idx, weak_preds_strong, seed + SEED_OFFSETS["kf_committee_agree"] + fi)
        disagree_bal = hard_weak_label_balance(disagree_idx, weak_preds_strong, seed + SEED_OFFSETS["kf_committee_disagree"] + fi)
        rand_bal = random_balanced_indices(
            weak_preds_strong, int(round(frac * n_strong_examples)), seed + SEED_OFFSETS["kf_committee_top"] + fi
        )
        run_subsets[f"committee_agree_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in agree_bal],
            [weak_labels[int(i)] for i in agree_bal],
        )
        run_subsets[f"committee_agree_unbalanced_f{pct}"] = (
            [strong_examples[int(i)] for i in agree_idx],
            [weak_labels[int(i)] for i in agree_idx],
        )
        run_subsets[f"committee_disagree_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in disagree_bal],
            [weak_labels[int(i)] for i in disagree_bal],
        )
        run_subsets[f"random_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in rand_bal],
            [weak_labels[int(i)] for i in rand_bal],
        )
        # Budget-matched oracle: the SAME random half as the arm above, trained on gold
        # labels instead of weak ones. The full-data gold anchor confounds budget with
        # labelling; this one holds the rows fixed so the difference is the labels alone.
        run_subsets[f"random_gt_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in rand_bal],
            [int(strong_examples[int(i)].label) for i in rand_bal],
        )
        # The other budget-matched oracle: the rows the weak model labels
        # correctly, drawn down to this budget and balanced the way random_balanced is. Its
        # labels are the weak ones, which on these rows are the gold ones, so it is the ceiling
        # for a selector that keeps this fraction and never keeps a wrong label. Where the
        # correct rows fall short of the budget it keeps all of them.
        correct_bal = hard_weak_label_balance(
            np.where(weak_preds_strong == gold_strong)[0], weak_preds_strong,
            seed + SEED_OFFSETS["kf_weak_correct"] + fi, size=int(round(frac * n_strong_examples)),
        )
        run_subsets[f"weak_correct_only_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in correct_bal],
            [weak_labels[int(i)] for i in correct_bal],
        )
        knn_high_idx, _ = score_band_indices(knn_stats["knn_correct_rate"], frac, "high")
        knn_high_bal = hard_weak_label_balance(knn_high_idx, weak_preds_strong, seed + SEED_OFFSETS["kf_knn_high"] + fi)
        run_subsets[f"knn_high_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in knn_high_bal],
            [weak_labels[int(i)] for i in knn_high_bal],
        )
        conf_high_idx, _ = score_band_indices(weak_confidences_strong, frac, "high")
        conf_high_bal = hard_weak_label_balance(conf_high_idx, weak_preds_strong, seed + SEED_OFFSETS["kf_confidence_high"] + fi)
        run_subsets[f"confidence_high_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in conf_high_bal],
            [weak_labels[int(i)] for i in conf_high_bal],
        )
        conf_low_idx, _ = score_band_indices(weak_confidences_strong, frac, "low")
        conf_low_bal = hard_weak_label_balance(conf_low_idx, weak_preds_strong, seed + SEED_OFFSETS["kf_confidence_low"] + fi)
        run_subsets[f"confidence_low_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in conf_low_bal],
            [weak_labels[int(i)] for i in conf_low_bal],
        )
        knn_low_idx, _ = score_band_indices(knn_stats["knn_correct_rate"], frac, "low")
        knn_low_bal = hard_weak_label_balance(knn_low_idx, weak_preds_strong, seed + SEED_OFFSETS["kf_knn_low"] + fi)
        run_subsets[f"knn_low_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in knn_low_bal],
            [weak_labels[int(i)] for i in knn_low_bal],
        )
        rp_high_idx, _ = score_band_indices(rp_scores, frac, "high")
        rp_high_bal = hard_weak_label_balance(rp_high_idx, weak_preds_strong, seed + SEED_OFFSETS["rp_high"] + fi)
        run_subsets[f"rp_high_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in rp_high_bal],
            [weak_labels[int(i)] for i in rp_high_bal],
        )
        rp_low_idx, _ = score_band_indices(rp_scores, frac, "low")
        rp_low_bal = hard_weak_label_balance(rp_low_idx, weak_preds_strong, seed + SEED_OFFSETS["rp_low"] + fi)
        run_subsets[f"rp_low_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in rp_low_bal],
            [weak_labels[int(i)] for i in rp_low_bal],
        )
        if cca_scores is not None:
            cc_high_idx, _ = score_band_indices(cca_scores, frac, "high")
            cc_high_bal = hard_weak_label_balance(cc_high_idx, weak_preds_strong, seed + SEED_OFFSETS["cca_high"] + fi)
            run_subsets[f"cca_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in cc_high_bal],
                [weak_labels[int(i)] for i in cc_high_bal],
            )
            cc_low_idx, _ = score_band_indices(cca_scores, frac, "low")
            cc_low_bal = hard_weak_label_balance(cc_low_idx, weak_preds_strong, seed + SEED_OFFSETS["cca_low"] + fi)
            run_subsets[f"cca_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in cc_low_bal],
                [weak_labels[int(i)] for i in cc_low_bal],
            )
        # linear/affine alignment residual (the existing ridge weak->strong map residuals):
        # what the affine map misses.
        _lin_resid = np.asarray(residuals, dtype=np.float64).reshape(-1)
        lr_high_idx, _ = score_band_indices(_lin_resid, frac, "high")
        lr_high_bal = hard_weak_label_balance(lr_high_idx, weak_preds_strong, seed + SEED_OFFSETS["l2resid_high"] + fi)
        run_subsets[f"l2resid_high_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in lr_high_bal],
            [weak_labels[int(i)] for i in lr_high_bal],
        )
        lr_low_idx, _ = score_band_indices(_lin_resid, frac, "low")
        lr_low_bal = hard_weak_label_balance(lr_low_idx, weak_preds_strong, seed + SEED_OFFSETS["l2resid_low"] + fi)
        run_subsets[f"l2resid_low_balanced_f{pct}"] = (
            [strong_examples[int(i)] for i in lr_low_bal],
            [weak_labels[int(i)] for i in lr_low_bal],
        )
        for _rname, _rscores in (("ccarobust", cca_robust_scores),):
            if _rscores is None:
                continue
            _hi_idx, _ = score_band_indices(_rscores, frac, "high")
            _hi_bal = hard_weak_label_balance(_hi_idx, weak_preds_strong, seed + SEED_OFFSETS[f"{_rname}_high"] + fi)
            run_subsets[f"{_rname}_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in _hi_bal],
                [weak_labels[int(i)] for i in _hi_bal],
            )
            _lo_idx, _ = score_band_indices(_rscores, frac, "low")
            _lo_bal = hard_weak_label_balance(_lo_idx, weak_preds_strong, seed + SEED_OFFSETS[f"{_rname}_low"] + fi)
            run_subsets[f"{_rname}_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in _lo_bal],
                [weak_labels[int(i)] for i in _lo_bal],
            )
        for _ci, (_slot, _sc) in enumerate(sorted(custom_scores.items())):
            _hi, _ = score_band_indices(_sc, frac, "high")
            _hb = hard_weak_label_balance(_hi, weak_preds_strong, seed + SEED_OFFSETS["custom_high"] + 1000 * _ci + fi)
            run_subsets[f"{_slot}_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in _hb],
                [weak_labels[int(i)] for i in _hb],
            )
            _lo, _ = score_band_indices(_sc, frac, "low")
            _lb = hard_weak_label_balance(_lo, weak_preds_strong, seed + SEED_OFFSETS["custom_low"] + 1000 * _ci + fi)
            run_subsets[f"{_slot}_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in _lb],
                [weak_labels[int(i)] for i in _lb],
            )
        if linstep1_scores is not None:
            ls_high_idx, _ = score_band_indices(linstep1_scores, frac, "high")
            ls_high_bal = hard_weak_label_balance(ls_high_idx, weak_preds_strong, seed + SEED_OFFSETS["linstep1_high"] + fi)
            run_subsets[f"linstep1_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ls_high_bal],
                [weak_labels[int(i)] for i in ls_high_bal],
            )
            ls_low_idx, _ = score_band_indices(linstep1_scores, frac, "low")
            ls_low_bal = hard_weak_label_balance(ls_low_idx, weak_preds_strong, seed + SEED_OFFSETS["linstep1_low"] + fi)
            run_subsets[f"linstep1_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ls_low_bal],
                [weak_labels[int(i)] for i in ls_low_bal],
            )
        if mlpstep1_scores is not None:
            ms_high_idx, _ = score_band_indices(mlpstep1_scores, frac, "high")
            ms_high_bal = hard_weak_label_balance(ms_high_idx, weak_preds_strong, seed + SEED_OFFSETS["mlpstep1_high"] + fi)
            run_subsets[f"mlpstep1_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ms_high_bal],
                [weak_labels[int(i)] for i in ms_high_bal],
            )
            ms_low_idx, _ = score_band_indices(mlpstep1_scores, frac, "low")
            ms_low_bal = hard_weak_label_balance(ms_low_idx, weak_preds_strong, seed + SEED_OFFSETS["mlpstep1_low"] + fi)
            run_subsets[f"mlpstep1_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ms_low_bal],
                [weak_labels[int(i)] for i in ms_low_bal],
            )
        # The excess loss, loss_strong - loss_weak at the weak model's own pick. Both bands
        # run, as for every score, even though the source of this one prescribes keep-high.
        if elloss_scores is not None:
            ell_high_idx, _ = score_band_indices(elloss_scores, frac, "high")
            ell_high_bal = hard_weak_label_balance(
                ell_high_idx, weak_preds_strong, seed + SEED_OFFSETS["elloss_high"] + fi
            )
            run_subsets[f"elloss_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ell_high_bal],
                [weak_labels[int(i)] for i in ell_high_bal],
            )
            ell_low_idx, _ = score_band_indices(elloss_scores, frac, "low")
            ell_low_bal = hard_weak_label_balance(
                ell_low_idx, weak_preds_strong, seed + SEED_OFFSETS["elloss_low"] + fi
            )
            run_subsets[f"elloss_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ell_low_bal],
                [weak_labels[int(i)] for i in ell_low_bal],
            )
        # Overlap-detection baseline. The score is built so that the high band is the paper's
        # overlap set at this budget and the low band its non-overlap set (the confidence tail
        # plus the far confident rows); the tail itself never enters the high band.
        if overlap_scores_arr is not None:
            ov_high_idx, _ = score_band_indices(overlap_scores_arr, frac, "high")
            ov_high_bal = hard_weak_label_balance(
                ov_high_idx, weak_preds_strong, seed + SEED_OFFSETS["overlap_high"] + fi
            )
            run_subsets[f"overlap_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ov_high_bal],
                [weak_labels[int(i)] for i in ov_high_bal],
            )
            ov_low_idx, _ = score_band_indices(overlap_scores_arr, frac, "low")
            ov_low_bal = hard_weak_label_balance(
                ov_low_idx, weak_preds_strong, seed + SEED_OFFSETS["overlap_low"] + fi
            )
            run_subsets[f"overlap_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in ov_low_bal],
                [weak_labels[int(i)] for i in ov_low_bal],
            )
        # whole-prompt token length: the selector that reads neither model. High keeps the
        # longest prompts, low the shortest. Same balancing and same kept-set size as every
        # other band, so it is a like-for-like floor on the by-method map.
        if toklen_scores is not None:
            tl_high_idx, _ = score_band_indices(toklen_scores, frac, "high")
            tl_high_bal = hard_weak_label_balance(tl_high_idx, weak_preds_strong, seed + SEED_OFFSETS["toklen_high"] + fi)
            run_subsets[f"toklen_high_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in tl_high_bal],
                [weak_labels[int(i)] for i in tl_high_bal],
            )
            tl_low_idx, _ = score_band_indices(toklen_scores, frac, "low")
            tl_low_bal = hard_weak_label_balance(tl_low_idx, weak_preds_strong, seed + SEED_OFFSETS["toklen_low"] + fi)
            run_subsets[f"toklen_low_balanced_f{pct}"] = (
                [strong_examples[int(i)] for i in tl_low_bal],
                [weak_labels[int(i)] for i in tl_low_bal],
            )


def compute_selection_scores(
    *,
    args,
    runs: list[str],
    rep_weak_acts: dict,
    rep_strong_acts: dict,
    weak_preds_strong,
    weak_probs_strong,
    device,
) -> tuple[dict, dict]:
    """Compute every representation-side selection score the requested runs need.

    Each score is computed only when some run
    asks for it, so an ordinary sweep pays for rp alone. Returns the scores by
    short name plus the externally supplied ones from --custom-scores-npz.

    The activation dicts stay alive after this returns; main frees them right
    after the call, before the excess-loss branch loads two models. Do not move
    that free in here: it would only unbind the parameters and free nothing.
    """
    rp_scores = representation_projection_scores(
        np.asarray(rep_weak_acts["strong_train"], dtype=np.float64),
        np.asarray(rep_strong_acts["strong_train"], dtype=np.float64),
        weak_preds_strong,
        reg=args.rp_reg,
        n_components=(args.rp_components or None),
    )
    # CCA alignment-disagreement score (cross-fitted regularized linear CCA); label-free.
    cca_scores = None
    if any(r.startswith("cca_") for r in runs):
        print("[cca] fitting cross-fitted CCA alignment (5 folds)...")
        cca_scores = cca_alignment_scores(
            np.asarray(rep_weak_acts["strong_train"], dtype=np.float64),
            np.asarray(rep_strong_acts["strong_train"], dtype=np.float64),
            n_folds=5, seed=args.seed,
        )
        print(f"[cca] scores: mean {cca_scores.mean():.3f} std {cca_scores.std():.3f}")
    # Robust variant (trim-refit): CCA that drops far outliers and refits.
    cca_robust_scores = None
    if any(r.startswith("ccarobust_") for r in runs):
        print("[ccarobust] fitting robust (trim-refit) CCA...")
        cca_robust_scores = cca_alignment_scores(
            np.asarray(rep_weak_acts["strong_train"], dtype=np.float64),
            np.asarray(rep_strong_acts["strong_train"], dtype=np.float64),
            n_folds=5, seed=args.seed, trim_frac=0.1,
        )
        print(f"[ccarobust] mean {cca_robust_scores.mean():.3f} std {cca_robust_scores.std():.3f}")
    # mlpstep1: a cross-fitted, weight-decayed MLP ensemble for step 1 (weight decay from
    # --mlpstep1-wd) and rp's in-sample kernel ridge for step 2.
    mlpstep1_scores = None
    if any(r.startswith("mlpstep1_") for r in runs):
        print(f"[mlpstep1] fitting MLP(wd={args.mlpstep1_wd})+ridge rp (5 folds x 3 ensemble)...")
        mlpstep1_scores = mlp_step1_rp_scores(
            np.asarray(rep_weak_acts["strong_train"], dtype=np.float32),
            np.asarray(rep_strong_acts["strong_train"], dtype=np.float32),
            weak_preds_strong,
            n_folds=5, seed=args.seed, device=str(device), ridge_reg=args.rp_reg,
            weight_decay=args.mlpstep1_wd,
        )
        print(f"[mlpstep1] scores: mean {mlpstep1_scores.mean():.4f} std {mlpstep1_scores.std():.4f}")
    # Pure cross-fit ablation: rp's step-1 kernel ridge fitted out-of-fold, step 2 unchanged.
    linstep1_scores = None
    if any(r.startswith("linstep1_") for r in runs):
        print("[linstep1] fitting oof kernel-ridge step-1 rp (5 folds, closed form)...")
        linstep1_scores = linstep1_rp_scores(
            np.asarray(rep_weak_acts["strong_train"], dtype=np.float64),
            np.asarray(rep_strong_acts["strong_train"], dtype=np.float64),
            weak_preds_strong,
            n_folds=5, seed=args.seed, ridge_reg=args.rp_reg,
        )
        print(f"[linstep1] scores: mean {linstep1_scores.mean():.4f} std {linstep1_scores.std():.4f}")
    # The overlap-detection baseline (arXiv:2412.03881, Alg. 2): the weak-confidence tail is cut
    # at the paper's change point, the rest is scored by |cos| to that tail in the untuned
    # strong representation. Selection is train-seed independent, so it is computed once here.
    overlap = None
    if any(r.startswith("overlap_") for r in runs):
        print("[overlap] change-point tail on weak confidence, |cos| to the tail in strong reps...")
        Xs = np.asarray(rep_strong_acts["strong_train"], dtype=np.float64)
        overlap = overlap_scores(weak_probs_strong, Xs, hard_frac="auto", metric="cos", seed=args.seed)
        n_hard = int((overlap == 0).sum())
        print(f"[overlap] hard tail {n_hard}/{len(overlap)} rows ({n_hard / len(overlap):.1%}); "
              f"closeness on the rest: mean {overlap[overlap > 0].mean() - 1:.3f}")
    # Precomputed custom scores (offline-designed selectors, e.g. the step-2 screen's
    # picks). The npz must carry weak_preds for the alignment guard.
    custom_scores = {}
    if args.custom_scores_npz and any(r.startswith("cs") for r in runs):
        _cs = np.load(args.custom_scores_npz)
        _stored = np.asarray(_cs["weak_preds"]).astype(int)
        _own = np.asarray(weak_preds_strong).astype(int)
        _rate = float((_stored == _own).mean()) if len(_stored) == len(_own) else 0.0
        if _rate < 0.95:
            raise ValueError(f"custom-scores npz misaligned with strong_train (weak_preds match {_rate:.3f})")
        if _rate < 0.995:
            print(f"[custom] WARNING: weak_preds match {_rate:.3f} -- cross-machine probe drift tolerated")
        for _slot in ("cs1", "cs2", "cs3", "cs4"):
            if _slot in _cs.files:
                custom_scores[_slot] = np.asarray(_cs[_slot], dtype=np.float64)
        print(f"[custom] loaded {sorted(custom_scores)} from {args.custom_scores_npz}")

    return (
        {
            "rp": rp_scores,
            "cca": cca_scores,
            "cca_robust": cca_robust_scores,
            "mlpstep1": mlpstep1_scores,
            "linstep1": linstep1_scores,
            "overlap": overlap,
        },
        custom_scores,
    )


def prepare_multichoice_eval(*, args, device, output_dir, weak_probe) -> tuple[list | None, float | None]:
    """Build the multiple-choice eval set and score the weak probe on it.

    Returns (eval3_examples, weak_eval3_acc). The weak accuracy here is the lower
    bound of the bounds table: the same argmax-over-candidates metric the strong
    side is scored with, so the two are directly comparable. Both are None when
    --eval-3class is off.
    """
    eval3_examples = None
    if args.eval_3class:
        if args.dataset == "sciq":
            multichoice_rows = load_sciq_multichoice_eval(
                args.n_eval_questions, args.seed + 2, args.sciq_use_support
            )
        elif args.dataset == "anli":
            multichoice_rows = load_anli_multichoice_eval(
                args.n_eval_questions, args.seed + 2, args.anli_round
            )
        elif args.dataset == "hellaswag":
            multichoice_rows = load_hellaswag_multichoice_eval(args.n_eval_questions, args.seed + 2)
        elif args.dataset == "wanli":
            multichoice_rows = load_wanli_multichoice_eval(args.n_eval_questions, args.seed + 2)
        elif args.dataset == "fevernli":
            multichoice_rows = load_fevernli_multichoice_eval(args.n_eval_questions, args.seed + 2)
        elif args.dataset == "imppres":
            multichoice_rows = load_imppres_multichoice_eval(args.n_eval_questions, args.seed, args.n_test)
        elif args.dataset == "aquarat":
            multichoice_rows = load_aquarat_multichoice_eval(args.n_eval_questions, args.seed + 2)
        elif args.dataset == "quail":
            multichoice_rows = generic_mc_eval(quail_rows("validation", args.seed + 2), "quail", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "riddlesense":
            multichoice_rows = generic_mc_eval(riddlesense_rows("validation", args.seed + 2), "riddlesense", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "scinli":
            multichoice_rows = generic_mc_eval(scinli_rows("test", args.seed + 2), "scinli", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "csqa":
            multichoice_rows = generic_mc_eval(csqa_rows("validation", args.seed + 2), "csqa", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "cosmosqa":
            multichoice_rows = generic_mc_eval(cosmosqa_rows("validation", args.seed + 2), "cosmosqa", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "blimp":
            multichoice_rows = generic_mc_eval(blimp_rows("validation", args.seed)[: args.n_eval_questions], "blimp", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "sst2":
            multichoice_rows = generic_mc_eval(sst2_rows("validation", args.seed + 2), "sst2", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "qnli":
            multichoice_rows = generic_mc_eval(qnli_rows("validation", args.seed + 2), "qnli", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "agnews":
            multichoice_rows = generic_mc_eval(agnews_rows("test", args.seed + 2)[: args.n_eval_questions], "agnews", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "imdb":
            multichoice_rows = generic_mc_eval(imdb_rows("test", args.seed + 2)[: args.n_eval_questions], "imdb", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "tweetsent":
            multichoice_rows = generic_mc_eval(tweetsent_rows("test", args.seed + 2)[: args.n_eval_questions], "tweetsent", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "wikitoxic":
            multichoice_rows = generic_mc_eval(wikitoxic_rows("test", args.seed + 2)[: args.n_eval_questions], "wikitoxic", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "scitail":
            multichoice_rows = generic_mc_eval(scitail_rows("test", args.seed + 2)[: args.n_eval_questions], "scitail", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "boolq":
            multichoice_rows = generic_mc_eval(boolq_rows("validation", args.seed + 2)[: args.n_eval_questions], "boolq", args.n_eval_questions, args.seed + 2)
        elif args.dataset == "ethicsjustice":
            multichoice_rows = generic_mc_eval(ethicsjustice_rows("test", args.seed + 2)[: args.n_eval_questions], "ethicsjustice", args.n_eval_questions, args.seed + 2)
        else:
            multichoice_rows = []
        if multichoice_rows:
            eval3_examples = [
                LoraExample(
                    id=r["id"],
                    source_id=r["source_id"],
                    text=r["txt"],
                    label=int(r["labels"]),
                    answer_suffix=args.answer_suffix,
                )
                for r in multichoice_rows
            ]
            n_questions = len({r["source_id"] for r in multichoice_rows})
            print(
                f"[multichoice] built {len(eval3_examples)} candidate rows over "
                f"{n_questions} {args.dataset} questions"
            )
    # Weak-model multiple-choice accuracy = the LOWER BOUND for the bounds table /
    # multichoice PGR. Score the candidate set with the (already-fitted) weak probe and
    # argmax over each question's candidates, same as the strong-side multichoice metric.
    weak_eval3_acc = None
    if eval3_examples:
        weak_eval3_acts = extract_final_token_activations(
            args.weak_model,
            [ex.text for ex in eval3_examples],
            device,
            args.torch_dtype,
            args.activation_batch_size,
            args.activation_max_length,
            "extract weak eval3 activations",
        )
        weak_eval3_rows = [
            {"prob_label1": float(p)} for p in predict_probe(weak_probe, weak_eval3_acts, device)
        ]
        weak_eval3_acc = accuracy_3class(eval3_examples, weak_eval3_rows)
        write_eval3_rows(output_dir / "eval3_weak.csv", eval3_examples, weak_eval3_rows)
        del weak_eval3_acts
        clear_memory()
        print(f"[multichoice] weak-probe lower bound accuracy_3class = {weak_eval3_acc:.3f}")

    return eval3_examples, weak_eval3_acc




def flip_labels_to_purity(labels: list, gold: list, target: float, rng) -> list:
    """Flip binary labels toward or away from gold until mean(labels == gold) hits target.

    Raising purity flips randomly chosen incorrect rows to gold; lowering flips
    randomly chosen correct rows to 1 - gold. The flip count is rounded, so the
    achieved purity lands within 1/(2n) of the target.
    """
    labels = [int(x) for x in labels]
    n = len(labels)
    correct = [i for i in range(n) if labels[i] == gold[i]]
    wrong = [i for i in range(n) if labels[i] != gold[i]]
    want = int(round(target * n))
    if want > len(correct):
        for i in rng.choice(wrong, size=want - len(correct), replace=False):
            labels[int(i)] = int(gold[int(i)])
    elif want < len(correct):
        for i in rng.choice(correct, size=len(correct) - want, replace=False):
            labels[int(i)] = 1 - int(gold[int(i)])
    return labels


def build_run_plan(
    *,
    args,
    runs,
    strong_examples,
    strong_train_labels,
    weak_preds_strong,
    weak_probs_strong,
    committee_disagreement_strong,
    residuals,
    knn_stats,
    scores,
    custom_scores,
) -> dict:
    """Turn the scores and diagnostics into the arm subsets a sweep trains on.

    Everything between scoring and the train loop: the residual and confidence
    bands, every run_subsets entry, and the random controls. Returns one dict; "bands" inside it carries the per-band filters
    and balanced indices the report writers read.
    """
    middle_indices, middle_filter = middle_residual_indices(residuals, args.residual_keep_middle_frac)
    middle_balanced_indices = hard_weak_label_balance(
        middle_indices,
        weak_preds_strong,
        args.seed,
    )
    weak_confidences_strong = 2.0 * np.abs(weak_probs_strong - 0.5)
    confidence_middle_indices, confidence_middle_filter = score_band_indices(
        weak_confidences_strong,
        args.confidence_keep_frac,
        "middle",
    )
    confidence_middle_balanced_indices = hard_weak_label_balance(
        confidence_middle_indices,
        weak_preds_strong,
        args.seed + SEED_OFFSETS["confidence_middle"],
    )
    confidence_high_indices, confidence_high_filter = score_band_indices(
        weak_confidences_strong,
        args.confidence_keep_frac,
        "high",
    )
    confidence_high_balanced_indices = hard_weak_label_balance(
        confidence_high_indices,
        weak_preds_strong,
        args.seed + SEED_OFFSETS["confidence_high"],
    )
    knn_middle_indices, knn_middle_filter = score_band_indices(
        knn_stats["knn_correct_rate"],
        args.knn_keep_middle_frac,
        "middle",
    )
    knn_middle_balanced_indices = hard_weak_label_balance(
        knn_middle_indices,
        weak_preds_strong,
        args.seed + SEED_OFFSETS["knn_middle"],
    )
    knn_mixed_indices, knn_mixed_filter = score_closest_indices(
        knn_stats["knn_correct_rate"],
        args.knn_keep_middle_frac,
        args.knn_mixed_center,
        "mixed",
    )
    knn_mixed_balanced_indices = hard_weak_label_balance(
        knn_mixed_indices,
        weak_preds_strong,
        args.seed + SEED_OFFSETS["knn_mixed"],
    )
    knn_high_indices, _ = score_band_indices(
        knn_stats["knn_correct_rate"],
        args.knn_keep_middle_frac,
        "high",
    )
    knn_high_balanced_indices = hard_weak_label_balance(
        knn_high_indices,
        weak_preds_strong,
        args.seed + SEED_OFFSETS["knn_high"],
    )
    committee_agree_indices, committee_agree_filter = score_band_indices(
        committee_disagreement_strong,
        args.committee_keep_frac,
        "low",
    )
    committee_agree_balanced_indices = hard_weak_label_balance(
        committee_agree_indices,
        weak_preds_strong,
        args.seed + SEED_OFFSETS["committee_agree"],
    )
    committee_disagree_indices, committee_disagree_filter = score_band_indices(
        committee_disagreement_strong,
        args.committee_keep_frac,
        "high",
    )
    committee_disagree_balanced_indices = hard_weak_label_balance(
        committee_disagree_indices,
        weak_preds_strong,
        args.seed + SEED_OFFSETS["committee_disagree"],
    )
    random_control_size = args.random_control_size or len(middle_balanced_indices)

    weak_labels = weak_preds_strong.tolist()
    # Fixed training orders by run name; every arm here shuffles, so the dict stays
    # empty and train_lora_model's curriculum_order plumbing is never exercised.
    curriculum_orders: dict[str, list[int]] = {}
    # Per-example loss weights by run name; every arm here trains unweighted, so the
    # dict stays empty and train_lora_model's sample_weights plumbing is never exercised.
    run_weights: dict[str, np.ndarray] = {}
    run_subsets: dict[str, tuple[list[LoraExample], list[int]]] = {
        "ground_truth": (strong_examples, strong_train_labels.tolist()),
        "weak_label": (strong_examples, weak_labels),
        "middle_unbalanced": (
            [strong_examples[int(i)] for i in middle_indices],
            [weak_labels[int(i)] for i in middle_indices],
        ),
        "middle_balanced": (
            [strong_examples[int(i)] for i in middle_balanced_indices],
            [weak_labels[int(i)] for i in middle_balanced_indices],
        ),
        "confidence_middle_unbalanced": (
            [strong_examples[int(i)] for i in confidence_middle_indices],
            [weak_labels[int(i)] for i in confidence_middle_indices],
        ),
        "confidence_middle_balanced": (
            [strong_examples[int(i)] for i in confidence_middle_balanced_indices],
            [weak_labels[int(i)] for i in confidence_middle_balanced_indices],
        ),
        "confidence_high_unbalanced": (
            [strong_examples[int(i)] for i in confidence_high_indices],
            [weak_labels[int(i)] for i in confidence_high_indices],
        ),
        "confidence_high_balanced": (
            [strong_examples[int(i)] for i in confidence_high_balanced_indices],
            [weak_labels[int(i)] for i in confidence_high_balanced_indices],
        ),
        "knn_middle_unbalanced": (
            [strong_examples[int(i)] for i in knn_middle_indices],
            [weak_labels[int(i)] for i in knn_middle_indices],
        ),
        "knn_middle_balanced": (
            [strong_examples[int(i)] for i in knn_middle_balanced_indices],
            [weak_labels[int(i)] for i in knn_middle_balanced_indices],
        ),
        "knn_mixed_unbalanced": (
            [strong_examples[int(i)] for i in knn_mixed_indices],
            [weak_labels[int(i)] for i in knn_mixed_indices],
        ),
        "knn_mixed_balanced": (
            [strong_examples[int(i)] for i in knn_mixed_balanced_indices],
            [weak_labels[int(i)] for i in knn_mixed_balanced_indices],
        ),
        "knn_high_unbalanced": (
            [strong_examples[int(i)] for i in knn_high_indices],
            [weak_labels[int(i)] for i in knn_high_indices],
        ),
        "knn_high_balanced": (
            [strong_examples[int(i)] for i in knn_high_balanced_indices],
            [weak_labels[int(i)] for i in knn_high_balanced_indices],
        ),
        "committee_agree_unbalanced": (
            [strong_examples[int(i)] for i in committee_agree_indices],
            [weak_labels[int(i)] for i in committee_agree_indices],
        ),
        "committee_agree_balanced": (
            [strong_examples[int(i)] for i in committee_agree_balanced_indices],
            [weak_labels[int(i)] for i in committee_agree_balanced_indices],
        ),
        "committee_disagree_unbalanced": (
            [strong_examples[int(i)] for i in committee_disagree_indices],
            [weak_labels[int(i)] for i in committee_disagree_indices],
        ),
        "committee_disagree_balanced": (
            [strong_examples[int(i)] for i in committee_disagree_balanced_indices],
            [weak_labels[int(i)] for i in committee_disagree_balanced_indices],
        ),
    }
    # Optional committee selection sweep over keep fractions (label-complexity /
    # data-efficiency curve): for each f, keep the most-reliable (low-disagreement)
    # or most-boundary (high-disagreement) f-fraction, plus a matched random_balanced.
    committee_keep_fracs = [
        float(x) for x in (args.committee_keep_fracs or "").split(",") if x.strip()
    ]
    n_strong_examples = len(strong_examples)
    build_keep_fraction_arms(
        seed=args.seed,
        run_subsets=run_subsets,
        scores=scores,
        custom_scores=custom_scores,
        committee_keep_fracs=committee_keep_fracs,
        committee_disagreement_strong=committee_disagreement_strong,
        weak_confidences_strong=weak_confidences_strong,
        weak_preds_strong=weak_preds_strong,
        weak_labels=weak_labels,
        strong_examples=strong_examples,
        residuals=residuals,
        knn_stats=knn_stats,
        random_balanced_indices=random_balanced_indices,
        n_strong_examples=n_strong_examples,
    )

    if "purity_grid" in runs or "purity_natural" in runs:
        # one fixed random subset, labels moved to each target purity: selection
        # never sees the scores, so the line isolates label quality.
        # Natural point, relabel line, and iid line all share this subset, so
        # the comparisons are paired.
        gold_all = [int(x) for x in strong_train_labels]
        base_sel = random_balanced_indices(
            weak_preds_strong, random_control_size, args.seed + SEED_OFFSETS["purity_flip"]
        )
        sub_examples = [strong_examples[int(i)] for i in base_sel]
        sub_weak = [weak_labels[int(i)] for i in base_sel]
        sub_gold = [gold_all[int(i)] for i in base_sel]
        nat_purity = float(np.mean([w == g for w, g in zip(sub_weak, sub_gold)]))
    if "purity_natural" in runs:
        runs = [name for name in runs if name != "purity_natural"]
        run_subsets["random_purity_natural"] = (sub_examples, list(sub_weak))
        runs.append("random_purity_natural")
    if "purity_grid" in runs:
        runs = [name for name in runs if name != "purity_grid"]
        targets = []
        for tok in [x.strip() for x in args.purity_grid.split(",") if x.strip()]:
            targets.append(("nat", nat_purity) if tok == "nat" else (tok, int(tok) / 100.0))
        n_draws = max(1, args.purity_draws) if args.purity_mode == "iid" else 1
        for tag, target in targets:
            pt_off = int(round(target * 100))
            for draw in range(n_draws):
                if args.purity_mode == "iid":
                    # errors at uniformly random positions: start from gold, corrupt
                    rng = np.random.default_rng(
                        args.seed + SEED_OFFSETS["purity_iid"] + pt_off + 200 * draw
                    )
                    n_sub = len(sub_gold)
                    labels = list(sub_gold)
                    n_wrong = int(round((1.0 - target) * n_sub))
                    for i in rng.choice(n_sub, size=n_wrong, replace=False):
                        labels[int(i)] = 1 - labels[int(i)]
                    run_name = f"random_purity_iid_p{tag}" + (f"_d{draw}" if n_draws > 1 else "")
                else:
                    rng = np.random.default_rng(args.seed + SEED_OFFSETS["purity_flip"] + pt_off)
                    labels = flip_labels_to_purity(sub_weak, sub_gold, target, rng)
                    run_name = f"random_purity_p{tag}"
                run_subsets[run_name] = (sub_examples, labels)
                runs.append(run_name)
    return {
        "runs": runs,
        "run_subsets": run_subsets,
        "curriculum_orders": curriculum_orders,
        "run_weights": run_weights,
        "bands": {
            "middle_filter": middle_filter,
            "confidence_middle_filter": confidence_middle_filter,
            "confidence_high_filter": confidence_high_filter,
            "knn_middle_filter": knn_middle_filter,
            "knn_mixed_filter": knn_mixed_filter,
            "committee_agree_filter": committee_agree_filter,
            "committee_disagree_filter": committee_disagree_filter,
            "middle_balanced_indices": middle_balanced_indices,
            "confidence_middle_balanced_indices": confidence_middle_balanced_indices,
            "confidence_high_balanced_indices": confidence_high_balanced_indices,
            "knn_middle_balanced_indices": knn_middle_balanced_indices,
            "knn_mixed_balanced_indices": knn_mixed_balanced_indices,
            "committee_agree_balanced_indices": committee_agree_balanced_indices,
            "committee_disagree_balanced_indices": committee_disagree_balanced_indices,
        },
    }



def write_run_outputs(
    *,
    args,
    output_dir,
    plan,
    run_reports,
    prediction_columns,
    best_map,
    best_row,
    committee_disagreement_strong,
    device,
    eval_examples,
    knn_stats,
    lora_examples,
    residuals,
    rp_scores,
    splits,
    start,
    strong_examples,
    strong_train_labels,
    test_labels,
    weak_acts,
    weak_correct_weak_train,
    weak_eval3_acc,
    weak_eval_rows,
    weak_preds_strong,
    weak_preds_test,
    weak_probs_strong,
) -> None:
    """Write everything a finished run leaves behind.

    The prediction CSVs, the kept-set summaries and diagnostics, summary.json,
    the text report, and the optional activation save. Consumes the plan dict
    from build_run_plan and the per-run reports from the train loop; nothing
    flows back out.
    """
    run_subsets = plan["run_subsets"]
    middle_filter = plan["bands"]["middle_filter"]
    confidence_middle_filter = plan["bands"]["confidence_middle_filter"]
    confidence_high_filter = plan["bands"]["confidence_high_filter"]
    knn_middle_filter = plan["bands"]["knn_middle_filter"]
    knn_mixed_filter = plan["bands"]["knn_mixed_filter"]
    committee_agree_filter = plan["bands"]["committee_agree_filter"]
    committee_disagree_filter = plan["bands"]["committee_disagree_filter"]
    middle_balanced_indices = plan["bands"]["middle_balanced_indices"]
    confidence_middle_balanced_indices = plan["bands"]["confidence_middle_balanced_indices"]
    confidence_high_balanced_indices = plan["bands"]["confidence_high_balanced_indices"]
    knn_middle_balanced_indices = plan["bands"]["knn_middle_balanced_indices"]
    knn_mixed_balanced_indices = plan["bands"]["knn_mixed_balanced_indices"]
    committee_agree_balanced_indices = plan["bands"]["committee_agree_balanced_indices"]
    committee_disagree_balanced_indices = plan["bands"]["committee_disagree_balanced_indices"]
    write_predictions(output_dir / "eval_predictions.csv", eval_examples, prediction_columns, weak_eval_rows)
    # Optional: untuned-strong (base) yes/no P(correct) on each strong_train example, so analyze_rp
    # can test whether rp selects points the strong BASE can already (nearly) learn (overlap).
    base_train_probs = None
    if args.score_base_on_train:
        from lora_train_eval import load_strong_model_and_tokenizer
        base_model, base_tok = load_strong_model_and_tokenizer(args, trainable_lora=False)
        _, base_rows = evaluate_yes_no(
            base_model, base_tok, lora_examples["strong_train"],
            max(int(args.strong_batch_size), 8), device, args.max_length, "score base on strong_train",
        )
        base_train_probs = np.asarray([float(r["prob_label1"]) for r in base_rows], dtype=np.float64)
        del base_model
        clear_memory()
    write_strong_train_labels(output_dir / "strong_train_labels.csv", strong_examples, weak_probs_strong, residuals, knn_stats, rp_scores=rp_scores, base_probs=base_train_probs)
    write_knn_diagnostics(
        output_dir / "knn_diagnostics.csv",
        strong_examples,
        strong_train_labels,
        weak_probs_strong,
        residuals,
        knn_stats,
    )
    write_subset_csv(output_dir / "train_subsets.csv", {name: run_subsets[name] for name in run_subsets if name in run_reports})

    subset_summaries = {
        name: summarize_train_subset(*subset)
        for name, subset in run_subsets.items()
    }
    subset_summaries["middle_residual_diagnostics"] = subset_summary(
        middle_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["confidence_middle_diagnostics"] = subset_summary(
        confidence_middle_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["confidence_high_diagnostics"] = subset_summary(
        confidence_high_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["knn_middle_diagnostics"] = subset_summary(
        knn_middle_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["knn_mixed_diagnostics"] = subset_summary(
        knn_mixed_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["committee_agree_diagnostics"] = subset_summary(
        committee_agree_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )
    subset_summaries["committee_disagree_diagnostics"] = subset_summary(
        committee_disagree_balanced_indices,
        strong_train_labels,
        weak_probs_strong,
        residuals,
    )

    fmt = format_summary(args.dataset, args.sciq_use_support, args.anli_round)
    summary = build_summary(
        args=args,
        best_map=best_map,
        best_row=best_row,
        committee_agree_filter=committee_agree_filter,
        committee_disagree_filter=committee_disagree_filter,
        committee_disagreement_strong=committee_disagreement_strong,
        confidence_high_filter=confidence_high_filter,
        confidence_middle_filter=confidence_middle_filter,
        fmt=fmt,
        knn_middle_filter=knn_middle_filter,
        knn_mixed_filter=knn_mixed_filter,
        knn_stats=knn_stats,
        middle_filter=middle_filter,
        output_dir=output_dir,
        run_reports=run_reports,
        splits=splits,
        start=start,
        strong_train_labels=strong_train_labels,
        subset_summaries=subset_summaries,
        test_labels=test_labels,
        weak_correct_weak_train=weak_correct_weak_train,
        weak_eval3_acc=weak_eval3_acc,
        weak_preds_strong=weak_preds_strong,
        weak_preds_test=weak_preds_test,
        weak_probs_strong=weak_probs_strong,
    )

    if args.save_activations:
        torch.save(
            {
                "weak_activations": weak_acts,
                "splits": {
                    name: getattr(splits, name).to_pandas()
                    for name in ["weak_train", "strong_train", "val", "test"]
                },
            },
            output_dir / "activations.pt",
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_text_report(output_dir / "paper_style_lora_report.txt", summary)
    print(json.dumps(summary, indent=2))
    print(f"Wrote outputs to {output_dir}")


def main() -> None:
    args = parse_args()
    runs = requested_runs(args)
    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this script on a CUDA GPU machine.")
    args.device = str(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    splits = load_paper_style_splits(args)
    texts = split_texts(splits)

    lora_examples = {
        "weak_train": to_lora_examples(splits.weak_train, args.answer_suffix),
        "strong_train": to_lora_examples(splits.strong_train, args.answer_suffix),
        "test": to_lora_examples(splits.test, args.answer_suffix),
    }

    weak_acts = extract_all_activations(
        args.weak_model,
        texts,
        device,
        args.torch_dtype,
        args.activation_batch_size,
        args.activation_max_length,
        "extract weak activations",
    )
    weak_probe = fit_probe(
        weak_acts["weak_train"],
        torch.tensor(split_labels(splits.weak_train), dtype=torch.float32),
        args.l2_penalty,
        args.max_iter,
        device,
    )
    weak_probs_weak_train = predict_probe(weak_probe, weak_acts["weak_train"], device)
    weak_probs_strong = predict_probe(weak_probe, weak_acts["strong_train"], device)
    weak_probs_test = predict_probe(weak_probe, weak_acts["test"], device)
    weak_train_labels = split_labels(splits.weak_train)
    weak_preds_weak_train = (weak_probs_weak_train >= 0.5).astype(int)
    if args.knn_reference_cross_fit:
        cross_fit_probs_weak_train = cross_fitted_weak_probs(
            weak_acts["weak_train"],
            weak_train_labels,
            args.cross_fit_folds,
            args.l2_penalty,
            args.max_iter,
            device,
            args.seed,
        )
        reference_preds_weak_train = (cross_fit_probs_weak_train >= 0.5).astype(int)
        weak_correct_weak_train = (reference_preds_weak_train == weak_train_labels).astype(int)
        print(
            "[cross-fit] weak_train reference accuracy: "
            f"in-sample={float(np.mean(weak_preds_weak_train == weak_train_labels)):.3f}, "
            f"cross-fitted={float(np.mean(weak_correct_weak_train)):.3f} "
            f"(folds={max(2, min(args.cross_fit_folds, len(weak_train_labels)))})"
        )
    else:
        weak_correct_weak_train = (weak_preds_weak_train == weak_train_labels).astype(int)
    weak_preds_strong = (weak_probs_strong >= 0.5).astype(int)
    weak_preds_test = (weak_probs_test >= 0.5).astype(int)

    committee_mean_strong, committee_disagreement_strong = committee_disagreement_on_strong_train(
        weak_acts["weak_train"],
        weak_train_labels,
        weak_acts["strong_train"],
        args.committee_members,
        args.l2_penalty,
        args.max_iter,
        device,
        args.seed + SEED_OFFSETS["committee_fit"],
    )

    strong_train_labels = split_labels(splits.strong_train)
    test_labels = split_labels(splits.test)
    weak_eval_rows = eval_rows_from_probs(lora_examples["test"], weak_probs_test)

    strong_acts = extract_all_activations(
        args.strong_model,
        texts,
        device,
        args.torch_dtype,
        args.activation_batch_size,
        args.activation_max_length,
        "extract strong activations",
    )
    best_map, map_rows, residuals = fit_maps(
        weak_acts["weak_train"],
        weak_acts["strong_train"],
        strong_acts["weak_train"],
        strong_acts["strong_train"],
        splits,
        args,
        device,
        output_dir,
    )
    # kNN / RP may use a different activation LAYER (--representation-layer), token POOLING
    # (--representation-pooling), and/or SPAN (--representation-span: full prompt vs answer-only,
    # masking the question) as controls. The weak probe/confidence and the L2-residual maps stay
    # on the default end-layer / last-token / full-prompt reps (extracted above); only the kNN and
    # RP representations move, so the weak labels (and the oracle) are held fixed across the
    # comparison. The default (end, last, full) reuses the already-extracted acts (no extra cost).
    answer_span = args.representation_span in ("answer", "context")
    rep_is_default = (
        args.representation_layer == "end"
        and args.representation_pooling == "last"
        and not answer_span
    )
    if rep_is_default:
        rep_weak_acts, rep_strong_acts = weak_acts, strong_acts
    else:
        rep_tag = f"{args.representation_layer},{args.representation_pooling},{args.representation_span}"
        rep_weak_acts = extract_all_activations(
            args.weak_model, texts, device, args.torch_dtype,
            args.activation_batch_size, args.activation_max_length,
            f"extract weak activations ({rep_tag})",
            layer=args.representation_layer, pooling=args.representation_pooling,
            answer_span=answer_span, answer_suffix=args.answer_suffix,
            span_kind=args.representation_span,
        )
        rep_strong_acts = extract_all_activations(
            args.strong_model, texts, device, args.torch_dtype,
            args.activation_batch_size, args.activation_max_length,
            f"extract strong activations ({rep_tag})",
            layer=args.representation_layer, pooling=args.representation_pooling,
            answer_span=answer_span, answer_suffix=args.answer_suffix,
            span_kind=args.representation_span,
        )
    knn_stats = compute_weak_train_knn_stats(
        rep_strong_acts["weak_train"],
        rep_strong_acts["strong_train"],
        weak_correct_weak_train,
        args.knn_k,
    )
    if args.dump_acts:
        # the probe's pre-sigmoid margin: identical to logit(p) in exact math,
        # but free of the clip censoring at saturated rows
        weak_margins_strong = (
            weak_probe(weak_acts["strong_train"].to(device)).detach().cpu().numpy()
        )
        np.savez_compressed(
            args.dump_acts,
            weak=np.asarray(rep_weak_acts["strong_train"], dtype=np.float32),
            strong=np.asarray(rep_strong_acts["strong_train"], dtype=np.float32),
            weak_preds=np.asarray(weak_preds_strong, dtype=np.int64),
            weak_probs=np.asarray(weak_probs_strong, dtype=np.float32),
            weak_margin=np.asarray(weak_margins_strong, dtype=np.float32),
            gt=np.asarray(strong_train_labels, dtype=np.int64),
        )
        print(f"[dump-acts] wrote {args.dump_acts}")

    scores, custom_scores = compute_selection_scores(
        args=args,
        runs=runs,
        rep_weak_acts=rep_weak_acts,
        rep_strong_acts=rep_strong_acts,
        weak_preds_strong=weak_preds_strong,
        weak_probs_strong=weak_probs_strong,
        device=device,
    )
    rp_scores = scores["rp"]
    # rep_* alias weak_acts/strong_acts in the default path, so free them
    # unconditionally; deleting only strong_acts would leave the alias holding
    # the whole array through every LoRA run
    del rep_weak_acts, rep_strong_acts, strong_acts
    clear_memory()
    # excess loss: each model's yes/no head over the K options (no probe). Expensive (loads
    # the weak + untuned-strong models once), so only when an excess-loss run is requested;
    # independent of LoRA training, so it depends on the data seed alone.
    elloss_scores = None
    if args.dump_el or any(r.strip().startswith("elloss_") for r in args.runs.split(",")):
        el_out = compute_el_kway_scores(args, splits.strong_train, args.device)
        elloss_scores = el_out["elloss"]
    scores["elloss"] = elloss_scores
    if args.dump_el:
        np.savez_compressed(
            args.dump_el,
            elloss=el_out["elloss"],
            weak_pick=el_out["weak_pick"],
            gold_opt=el_out["gold_opt"],
            weak_preds=np.asarray(weak_preds_strong, dtype=np.int64),
            weak_probs=np.asarray(weak_probs_strong, dtype=np.float32),
            gt=np.asarray(strong_train_labels, dtype=np.int64),
        )
        print(f"[dump-el] wrote {args.dump_el}")

    best_row = next(row for row in map_rows if row["name"] == best_map.name)
    strong_examples = lora_examples["strong_train"]
    # Token length needs the prompts rather than the activations, so it is computed here
    # instead of in compute_selection_scores. Only the tokenizer is loaded, no weights.
    if any(r.strip().startswith("toklen_") for r in args.runs.split(",")):
        from transformers import AutoTokenizer

        _tl_tok = AutoTokenizer.from_pretrained(args.strong_model)
        scores["toklen"] = token_length_scores(
            [ex.prompt for ex in strong_examples], lambda s: _tl_tok(s)["input_ids"]
        )
        print(
            f"[toklen] whole-prompt tokens: mean {scores['toklen'].mean():.1f} "
            f"min {scores['toklen'].min():.0f} max {scores['toklen'].max():.0f}"
        )
        del _tl_tok
    eval_examples = lora_examples["test"]
    eval3_examples, weak_eval3_acc = prepare_multichoice_eval(
        args=args, device=device, output_dir=output_dir, weak_probe=weak_probe
    )
    plan = build_run_plan(
        args=args,
        runs=runs,
        strong_examples=strong_examples,
        strong_train_labels=strong_train_labels,
        weak_preds_strong=weak_preds_strong,
        weak_probs_strong=weak_probs_strong,
        committee_disagreement_strong=committee_disagreement_strong,
        residuals=residuals,
        knn_stats=knn_stats,
        scores=scores,
        custom_scores=custom_scores,
    )
    runs = plan["runs"]
    run_subsets = plan["run_subsets"]
    curriculum_orders = plan["curriculum_orders"]
    run_weights = plan["run_weights"]

    prediction_columns: dict[str, list[dict]] = {}
    run_reports: dict[str, dict] = {}

    if "base" in runs:
        from lora_train_eval import load_strong_model_and_tokenizer

        model, tokenizer = load_strong_model_and_tokenizer(args, trainable_lora=False)
        base_summary, base_rows = evaluate_yes_no(
            model,
            tokenizer,
            eval_examples,
            args.eval_batch_size or args.strong_batch_size,
            args.device,
            args.max_length,
            "eval base strong",
        )
        base_summary["auroc"] = auroc_from_rows(base_rows)
        base_summary["accuracy_prior_matched"] = prior_matched_accuracy(base_rows)
        if eval3_examples:
            _, base_rows3 = evaluate_yes_no(
                model,
                tokenizer,
                eval3_examples,
                args.eval_batch_size or args.strong_batch_size,
                args.device,
                args.max_length,
                "eval3 base",
            )
            base_summary["accuracy_3class"] = accuracy_3class(eval3_examples, base_rows3)
            write_eval3_rows(output_dir / "eval3_base.csv", eval3_examples, base_rows3)
        prediction_columns["base"] = base_rows
        run_reports["base"] = {"eval": base_summary}
        del model
        del tokenizer
        clear_memory()

    eff_batch = max(1, args.strong_batch_size * args.gradient_accumulation_steps)
    for run_name in runs:
        if run_name == "base":
            continue
        train_examples, train_labels = run_subsets[run_name]
        run_steps = None
        if args.epochs > 0:
            # Train this run for `epochs` full passes over its ACTUAL subset
            # (no-filter -> all data; f50 -> the real 50%), instead of the fixed
            # compute-matched cap. no-filter therefore gets proportionally more steps.
            run_steps = args.epochs * max(1, math.ceil(len(train_examples) / eff_batch))
        report, rows = run_lora_eval(
            args, run_name, train_examples, train_labels, eval_examples, output_dir, eval3_examples,
            curriculum_order=curriculum_orders.get(run_name),
            sample_weights=run_weights.get(run_name),
            max_train_steps=run_steps,
        )
        prediction_columns[run_name] = rows
        run_reports[run_name] = report
        log_arm_to_wandb(args, run_name, report, output_dir)

    write_run_outputs(
        args=args,
        output_dir=output_dir,
        plan=plan,
        run_reports=run_reports,
        prediction_columns=prediction_columns,
        best_map=best_map,
        best_row=best_row,
        committee_disagreement_strong=committee_disagreement_strong,
        device=device,
        eval_examples=eval_examples,
        knn_stats=knn_stats,
        lora_examples=lora_examples,
        residuals=residuals,
        rp_scores=rp_scores,
        splits=splits,
        start=start,
        strong_examples=strong_examples,
        strong_train_labels=strong_train_labels,
        test_labels=test_labels,
        weak_acts=weak_acts,
        weak_correct_weak_train=weak_correct_weak_train,
        weak_eval3_acc=weak_eval3_acc,
        weak_eval_rows=weak_eval_rows,
        weak_preds_strong=weak_preds_strong,
        weak_preds_test=weak_preds_test,
        weak_probs_strong=weak_probs_strong,
    )


if __name__ == "__main__":
    main()
