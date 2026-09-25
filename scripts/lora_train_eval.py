#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gc
import math
import time
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Example:
    id: str
    text: str
    label: int

    @property
    def prompt(self) -> str:
        return f"{self.text}\nAnswer:"


def require_peft():
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: peft. Install it in the remote environment with:\n"
            "  python3 -m pip install -r requirements-finetune.txt"
        ) from exc
    return LoraConfig, TaskType, get_peft_model


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def parse_lora_target_modules(value: str) -> list[str]:
    modules = [item.strip() for item in value.split(",") if item.strip()]
    if not modules:
        raise ValueError("--lora-target-modules cannot be empty.")
    return modules


def batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def answer_text(label: int) -> str:
    return " yes" if int(label) == 1 else " no"


def encode_prompt_answer(tokenizer, prompt: str, answer: str, max_length: int) -> dict[str, list[int]]:
    """Encode prompt+answer while always keeping answer tokens.

    Passages can be long. Ordinary truncation of the full string can cut the final
    " yes"/" no" answer, which leaves every label at -100 and the causal-LM loss
    undefined. The answer tokens are reserved first and the prompt is truncated from
    the left, so the question and candidate answer near its end are kept.
    """

    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        raise ValueError(f"Answer produced no tokens: {answer!r}")

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    prompt_budget = max_length - len(answer_ids)
    if prompt_budget <= 0:
        raise ValueError(f"max_length={max_length} is too short for answer {answer!r}")
    if len(prompt_ids) > prompt_budget:
        prompt_ids = prompt_ids[-prompt_budget:]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def pad_encoded_batch(tokenizer, encoded_rows: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in encoded_rows)
    pad_id = tokenizer.pad_token_id
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for row in encoded_rows:
        pad_len = max_len - len(row["input_ids"])
        batch["input_ids"].append(row["input_ids"] + [pad_id] * pad_len)
        batch["attention_mask"].append(row["attention_mask"] + [0] * pad_len)
        batch["labels"].append(row["labels"] + [-100] * pad_len)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


class PromptAnswerDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[Example], labels: list[int], weights=None):
        self.examples = examples
        self.labels = [int(label) for label in labels]
        # Optional per-example loss weights (continuous-weighting / loss-adaptation). When
        # None the item is a 2-tuple and behaves exactly as before (uniform weighting).
        self.weights = None if weights is None else [float(w) for w in weights]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        prompt, answer = self.examples[idx].prompt, answer_text(self.labels[idx])
        if self.weights is None:
            return prompt, answer
        return prompt, answer, self.weights[idx]


class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch) -> dict[str, torch.Tensor]:
        encoded_rows = [
            encode_prompt_answer(self.tokenizer, item[0], item[1], self.max_length)
            for item in batch
        ]
        out = pad_encoded_batch(self.tokenizer, encoded_rows)
        # Only carry per-sample weights when the dataset provides them (3-tuples). This keeps
        # the native/PairDataset path (2-tuples) -- which calls model(**batch) -- untouched.
        if len(batch[0]) > 2:
            out["sample_weight"] = torch.tensor([item[2] for item in batch], dtype=torch.float32)
        return out


def load_strong_model_and_tokenizer(args: argparse.Namespace, trainable_lora: bool):
    dtype = resolve_dtype(args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.strong_model)
    # Our prompt puts the question, the candidate answer and the yes/no cue at the END,
    # so right truncation (the tokenizer default) would cut exactly the part that
    # distinguishes one candidate from another. Truncate the passage instead.
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.strong_model, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if trainable_lora:
        LoraConfig, TaskType, get_peft_model = require_peft()
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=parse_lora_target_modules(
                getattr(args, "lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
            ),
        )
        model = get_peft_model(model, lora_config)
    model.to(args.device)
    return model, tokenizer


def count_params(model) -> tuple[int, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable


def weighted_token_ce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per-sample answer-token cross-entropy, averaged with per-sample weights.

    Reproduces the model's built-in causal-LM loss (mean CE over the answer tokens, where
    labels != -100) but at the SAMPLE level, then takes a weighted mean over the batch. With
    all weights == 1 this equals the standard mean loss; non-uniform weights up/down-weight
    each example's gradient -- the continuous-weighting / loss-adaptation knob.
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    mask = labels != -100
    safe = labels.clamp(min=0)
    tok_ce = -logp.gather(-1, safe.unsqueeze(-1)).squeeze(-1) * mask
    per_sample = tok_ce.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    return (per_sample * weights).sum() / weights.sum().clamp(min=1e-8)


def train_lora_model(
    args: argparse.Namespace,
    train_examples: list[Example],
    train_labels: list[int],
    run_name: str,
    output_dir: Path,
    curriculum_order=None,
    sample_weights=None,
) -> tuple[object, object, dict]:
    start_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_strong_model_and_tokenizer(args, trainable_lora=True)
    model.train()
    total_params, trainable_params = count_params(model)
    # Curriculum: when an order is given, present examples in that fixed order each pass
    # (easy/reliable -> hard) instead of random shuffling. The effective batch, #steps and
    # data are unchanged -- only the ordering differs (the control is the shuffled run).
    if curriculum_order is not None:
        train_examples = [train_examples[i] for i in curriculum_order]
        train_labels = [train_labels[i] for i in curriculum_order]
        if sample_weights is not None:
            sample_weights = [sample_weights[i] for i in curriculum_order]
    loader = DataLoader(
        PromptAnswerDataset(train_examples, train_labels, sample_weights),
        batch_size=args.strong_batch_size,
        shuffle=(curriculum_order is None),
        collate_fn=Collator(tokenizer, args.max_length),
    )
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=float(getattr(args, "weight_decay", 0.0)),
    )
    warmup_steps = int(getattr(args, "warmup_steps", 0))
    lr_decay = str(getattr(args, "lr_decay", "linear")).lower()
    total_steps = max(1, int(args.max_train_steps))

    def _lr_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if lr_decay == "linear":
            # linear decay from 1.0 (end of warmup) to ~0 at the final step, so training
            # settles instead of oscillating at constant lr (which made small noisy
            # subsets occasionally land on a bad final model).
            denom = max(1, total_steps - warmup_steps)
            return max(0.0, float(total_steps - step) / float(denom))
        return 1.0  # "none" -> constant after warmup

    scheduler = None
    if warmup_steps > 0 or lr_decay != "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_factor)
    data_iter = cycle(loader)
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for step in tqdm(range(args.max_train_steps), desc=f"train {run_name}"):
        step_losses = []
        for _ in range(args.gradient_accumulation_steps):
            batch = next(data_iter)
            weights = batch.pop("sample_weight", None)
            batch = {key: value.to(args.device) for key, value in batch.items()}
            if weights is None:
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
            else:
                outputs = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False
                )
                loss = weighted_token_ce(
                    outputs.logits, batch["labels"], weights.to(args.device)
                ) / args.gradient_accumulation_steps
            loss.backward()
            step_losses.append(float(loss.detach().cpu().item() * args.gradient_accumulation_steps))
        max_grad_norm = float(getattr(args, "max_grad_norm", 0.0))
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(sum(step_losses) / len(step_losses))

    if args.save_adapters:
        adapter_dir = output_dir / f"{run_name}_adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    report = {
        "run_name": run_name,
        "train_examples": len(train_examples),
        "train_label_mean": float(np.mean(train_labels)) if train_labels else math.nan,
        "max_train_steps": args.max_train_steps,
        "batch_size": args.strong_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr": args.lr,
        "weight_decay": float(getattr(args, "weight_decay", 0.0)),
        "warmup_steps": int(getattr(args, "warmup_steps", 0)),
        "lr_decay": str(getattr(args, "lr_decay", "linear")).lower(),
        "max_grad_norm": float(getattr(args, "max_grad_norm", 0.0)),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": parse_lora_target_modules(
            getattr(args, "lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
        ),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_param_fraction": trainable_params / total_params if total_params else math.nan,
        "losses": losses,
        "elapsed_sec": time.time() - start_time,
        "cuda_memory": cuda_memory_report(),
    }
    return model, tokenizer, report


@torch.no_grad()
def score_candidates(
    model,
    tokenizer,
    prompts: list[str],
    candidates: list[str],
    device: str,
    max_length: int,
) -> torch.Tensor:
    encoded = pad_encoded_batch(
        tokenizer,
        [
            encode_prompt_answer(tokenizer, prompt, candidate, max_length)
            for prompt, candidate in zip(prompts, candidates)
        ],
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    outputs = model(**encoded, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    targets = encoded["input_ids"][:, 1:]
    labels = encoded["labels"][:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_scores = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    scores = []
    for row_idx in range(targets.shape[0]):
        candidate_mask = labels[row_idx] != -100
        if not candidate_mask.any():
            raise RuntimeError("Candidate answer was fully masked during scoring.")
        scores.append(token_scores[row_idx][candidate_mask].sum())
    return torch.stack(scores).detach().cpu()


def single_token_ids(tokenizer, candidates: list[str]) -> list[int] | None:
    """Token ids of the candidates when each one is a single token, else None."""
    ids = [tokenizer(c, add_special_tokens=False)["input_ids"] for c in candidates]
    if any(len(i) != 1 for i in ids):
        return None
    return [i[0] for i in ids]


@torch.no_grad()
def score_single_token_candidates(
    model,
    tokenizer,
    prompts: list[str],
    candidate_ids: list[int],
    appended: str,
    device: str,
    max_length: int,
) -> torch.Tensor:
    """Log-probs of several single-token candidates from one forward pass, [n_prompts, n_candidates].

    The row fed to the model is the prompt with `appended` (one of the candidates) after it, the
    same row the two-pass scorer builds, so truncation and the logits at the last prompt position
    are unchanged; the causal mask keeps those logits independent of the appended token, and every
    candidate's log-prob is read off that one position. Without labels the model computes no loss,
    which spares the full-sequence float32 logits the two-pass scorer paid for on every pass.
    """
    if len(tokenizer(appended, add_special_tokens=False)["input_ids"]) != 1:
        raise ValueError(f"appended candidate must be a single token: {appended!r}")
    rows = [encode_prompt_answer(tokenizer, prompt, appended, max_length) for prompt in prompts]
    # the answer is one token, so the position that predicts it is the second-to-last real token
    last_prompt = torch.tensor([len(row["input_ids"]) - 2 for row in rows], dtype=torch.long, device=device)
    encoded = pad_encoded_batch(tokenizer, rows)
    outputs = model(
        input_ids=encoded["input_ids"].to(device),
        attention_mask=encoded["attention_mask"].to(device),
        use_cache=False,
    )
    at_prompt_end = outputs.logits[torch.arange(len(rows), device=device), last_prompt]
    log_probs = F.log_softmax(at_prompt_end, dim=-1)
    return log_probs[:, candidate_ids].detach().cpu()


@torch.no_grad()
def evaluate_yes_no(
    model,
    tokenizer,
    examples: list[Example],
    batch_size: int,
    device: str,
    max_length: int,
    desc: str,
) -> tuple[dict, list[dict]]:
    model.eval()
    rows = []
    # both answers are single tokens for the Qwen and Llama tokenizers, so one forward pass per
    # prompt gives both scores; the two-pass path stays for a tokenizer where they are not
    one_pass = single_token_ids(tokenizer, [" no", " yes"])
    for batch in tqdm(list(batched(examples, batch_size)), desc=desc):
        prompts = [ex.prompt for ex in batch]
        if one_pass is None:
            yes_scores = score_candidates(model, tokenizer, prompts, [" yes"] * len(batch), device, max_length)
            no_scores = score_candidates(model, tokenizer, prompts, [" no"] * len(batch), device, max_length)
        else:
            no_scores, yes_scores = score_single_token_candidates(
                model, tokenizer, prompts, one_pass, " yes", device, max_length
            ).unbind(dim=1)
        probs = torch.softmax(torch.stack([no_scores, yes_scores], dim=1), dim=1)[:, 1]
        preds = (probs >= 0.5).long()
        for ex, prob, pred in zip(batch, probs.tolist(), preds.tolist()):
            rows.append(
                {
                    "id": ex.id,
                    "label": ex.label,
                    "prob_label1": float(prob),
                    "pred": int(pred),
                    "correct": int(pred == ex.label),
                }
            )
    acc = sum(row["correct"] for row in rows) / len(rows) if rows else math.nan
    confidence = [2.0 * abs(row["prob_label1"] - 0.5) for row in rows]
    summary = {
        "n": len(rows),
        "accuracy": acc,
        "confidence_mean": float(np.mean(confidence)) if confidence else math.nan,
        "confidence_median": float(np.median(confidence)) if confidence else math.nan,
    }
    return summary, rows


def cuda_memory_report() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {
            "allocated_gb": None,
            "reserved_gb": None,
            "max_allocated_gb": None,
            "max_reserved_gb": None,
        }
    denom = 1024**3
    return {
        "allocated_gb": torch.cuda.memory_allocated() / denom,
        "reserved_gb": torch.cuda.memory_reserved() / denom,
        "max_allocated_gb": torch.cuda.max_memory_allocated() / denom,
        "max_reserved_gb": torch.cuda.max_memory_reserved() / denom,
    }


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_predictions(path: Path, examples: list[Example], columns: dict[str, list[dict]], weak_eval_rows: list[dict]) -> None:
    weak_by_id = {row["id"]: row for row in weak_eval_rows}
    column_by_id = {
        name: {row["id"]: row for row in rows}
        for name, rows in columns.items()
    }
    fieldnames = ["id", "label", "text", "weak_prob_label1", "weak_pred", "weak_correct"]
    for name in columns:
        fieldnames.extend([f"{name}_prob_label1", f"{name}_pred", f"{name}_correct"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for ex in examples:
            row = {"id": ex.id, "label": ex.label, "text": ex.text}
            weak = weak_by_id.get(ex.id, {})
            row.update(
                {
                    "weak_prob_label1": weak.get("prob_label1"),
                    "weak_pred": weak.get("pred"),
                    "weak_correct": weak.get("correct"),
                }
            )
            for name, by_id in column_by_id.items():
                pred_row = by_id.get(ex.id, {})
                row[f"{name}_prob_label1"] = pred_row.get("prob_label1")
                row[f"{name}_pred"] = pred_row.get("pred")
                row[f"{name}_correct"] = pred_row.get("correct")
            writer.writerow(row)
