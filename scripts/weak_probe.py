#!/usr/bin/env python3

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str):
    if dtype_arg == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_arg]


def batched_indices(n_items: int, batch_size: int):
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)


def answer_span_char_bounds(
    prompt: str, marker: str, answer_suffix: str, span_kind: str = "answer"
) -> tuple[int, int]:
    """Character range [start, end) of a sub-span of a candidate-correctness prompt.

    Prompt form: ``Premise: ...\\nHypothesis: ...\\nQ: ...? A: {candidate}`` optionally
    followed by ``answer_suffix`` (the fixed yes/no instruction). Two sub-spans:
      * ``span_kind="answer"``  -> ``A: {candidate}`` (from the LAST ``marker`` through the end
        of the text, excluding the question prefix AND the instruction suffix).
      * ``span_kind="context"`` -> everything from the start THROUGH the marker ``A:`` but
        EXCLUDING the candidate word (i.e. ``Premise: ... Q: ...? A:``).
    Returns (-1, -1) if the marker is absent (caller falls back to the full sequence).

    NOTE: only strip ``answer_suffix`` when the prompt actually ends with it. The
    representation extraction tokenizes suffix-less ``txt`` (the candidate is already at the
    end), so subtracting the suffix length there would truncate the ``A:`` marker and wrongly
    trigger the full-sequence fallback (the earlier answer-only-span bug).
    """
    end = len(prompt) - len(answer_suffix) if (answer_suffix and prompt.endswith(answer_suffix)) else len(prompt)
    marker_start = prompt.rfind(marker, 0, end)
    if marker_start < 0 or end <= marker_start:
        return -1, -1
    if span_kind == "context":
        return 0, marker_start + len(marker)
    return marker_start, end


def build_answer_span_mask(
    offsets: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_texts: list[str],
    marker: str,
    answer_suffix: str,
    span_kind: str = "answer",
) -> tuple[torch.Tensor, int]:
    """0/1 mask selecting only the tokens inside each prompt's answer span (see
    :func:`answer_span_char_bounds`). ``offsets`` is the fast-tokenizer offset mapping
    [B, T, 2]; special/pad tokens have (0, 0) and are excluded. Rows whose answer span is
    empty (marker missing, or truncated away) fall back to the full attention mask. Returns
    (span_mask, n_fallback) so the caller can warn when truncation ate the answer."""
    span_mask = torch.zeros_like(attention_mask)
    n_fallback = 0
    for b, prompt in enumerate(batch_texts):
        start_char, end_char = answer_span_char_bounds(prompt, marker, answer_suffix, span_kind)
        if start_char < 0:
            span_mask[b] = attention_mask[b]
            n_fallback += 1
            continue
        row = span_mask[b]
        for t in range(offsets.shape[1]):
            s, e = int(offsets[b, t, 0]), int(offsets[b, t, 1])
            if e <= s:  # special / pad token -> (0, 0)
                continue
            if s >= start_char and e <= end_char:
                row[t] = 1
        if int((row * attention_mask[b]).sum()) == 0:  # span truncated away
            span_mask[b] = attention_mask[b]
            n_fallback += 1
    return span_mask, n_fallback


@torch.no_grad()
def extract_final_token_activations(
    model_name: str,
    texts: list[str],
    device: torch.device,
    dtype_arg: str,
    batch_size: int,
    max_length: int | None,
    desc: str,
    layer: str | int = "end",
    pooling: str = "last",
    answer_span: bool = False,
    answer_marker: str = "A:",
    answer_suffix: str = "",
    span_kind: str = "answer",
) -> torch.Tensor:
    """Per-prompt hidden state at the requested transformer layer and token pooling.

    layer: "end" (last hidden layer -- the default the weak probe uses), "middle"
    (len(hidden_states)//2; the representation methods kNN/RP can use this as a
    control), or an explicit integer index into ``outputs.hidden_states``.
    pooling: "last" (the final non-pad token -- the probe / W2S-paper lineage) or "mean"
    (average over the non-pad tokens; kNN/RP can use this as a control).
    answer_span: if True, pool only over the ANSWER part of each prompt ("A: {candidate}",
    masking the question prefix and the trailing yes/no instruction ``answer_suffix``) --
    the answer-only ablation. Needs a fast tokenizer (offset mapping). Rows whose answer
    span is missing/truncated fall back to the full sequence.
    """
    dtype = resolve_dtype(dtype_arg)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Our prompt puts the question, the candidate answer and the yes/no cue at the END,
    # so right truncation (the tokenizer default) would cut exactly the part that
    # distinguishes one candidate from another. Truncate the passage instead.
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "num_labels": 2,
        "low_cpu_mem_usage": True,
    }
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "single_label_classification"
    model.to(device)
    model.eval()

    activations = []
    span_fallbacks = 0
    for start, end in tqdm(list(batched_indices(len(texts), batch_size)), desc=desc):
        batch_texts = texts[start:end]
        tokenize_kwargs: dict[str, Any] = {
            "return_tensors": "pt",
            "padding": True,
            "truncation": True,
        }
        if max_length is not None:
            tokenize_kwargs["max_length"] = max_length
        if answer_span:
            tokenize_kwargs["return_offsets_mapping"] = True
        inputs = tokenizer(batch_texts, **tokenize_kwargs)
        offsets = inputs.pop("offset_mapping", None)
        inputs = inputs.to(device)
        # pool_mask selects which tokens contribute: the full attention mask, or (answer_span)
        # only the answer-part tokens of each prompt.
        if answer_span:
            span_mask, n_fb = build_answer_span_mask(
                offsets, inputs["attention_mask"].cpu(), batch_texts, answer_marker, answer_suffix, span_kind
            )
            span_fallbacks += n_fb
            pool_mask = (span_mask.to(device) * inputs["attention_mask"]).to(torch.long)
        else:
            pool_mask = inputs["attention_mask"]
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states
        if layer == "end":
            layer_idx = len(hidden) - 1
        elif layer == "middle":
            layer_idx = len(hidden) // 2
        else:
            layer_idx = int(layer)
        last_hidden = hidden[layer_idx]
        if pooling == "mean":
            mask = pool_mask.unsqueeze(-1).to(last_hidden.dtype)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:  # "last": the final selected (non-pad / in-span) token
            arange = torch.arange(pool_mask.shape[1], device=device).unsqueeze(0)
            last_indices = (pool_mask * arange).argmax(dim=1)
            batch_indices = torch.arange(last_hidden.shape[0], device=device)
            pooled = last_hidden[batch_indices, last_indices]
        activations.append(pooled.detach().to(torch.float32).cpu())
    if answer_span and span_fallbacks:
        print(f"[answer-span] {span_fallbacks}/{len(texts)} prompts fell back to full sequence "
              f"(marker '{answer_marker}' missing or truncated away).")

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return torch.cat(activations, dim=0)


class LogisticProbe(torch.nn.Module):
    """Minimal copy of the original repo's LBFGS logistic probe."""

    def __init__(self, input_dim: int, device: torch.device):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1, device=device)
        self.linear.bias.data.zero_()
        self.linear.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

    @torch.enable_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor, l2_penalty: float, max_iter: int) -> float:
        optimizer = torch.optim.LBFGS(
            self.parameters(),
            line_search_fn="strong_wolfe",
            max_iter=max_iter,
        )
        y = y.to(torch.float32)
        loss = torch.inf

        def closure():
            nonlocal loss
            optimizer.zero_grad()
            logits = self(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            reg_loss = loss + l2_penalty * self.linear.weight.square().sum()
            reg_loss.backward()
            return float(reg_loss)

        optimizer.step(closure)
        return float(loss)

    @torch.no_grad()
    def predict_prob(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(x))


def fit_probe(x: torch.Tensor, y: torch.Tensor, l2_penalty: float, max_iter: int, device: torch.device) -> LogisticProbe:
    probe = LogisticProbe(x.shape[1], device)
    probe.fit(x.to(device), y.to(device), l2_penalty, max_iter)
    return probe


@torch.no_grad()
def predict_probe(probe: LogisticProbe, x: torch.Tensor, device: torch.device) -> np.ndarray:
    return probe.predict_prob(x.to(device)).detach().cpu().numpy()
