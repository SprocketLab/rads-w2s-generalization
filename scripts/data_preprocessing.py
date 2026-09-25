#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset

from contracts import LoraExample, SplitBundle


def balance_binary_dataset(ds: Dataset, seed: int) -> Dataset:
    labels = [int(label) for label in ds["labels"]]
    counts = {0: labels.count(0), 1: labels.count(1)}
    if not counts[0] or not counts[1]:
        return ds

    majority_label = 0 if counts[0] > counts[1] else 1
    minority_label = 1 - majority_label
    minority_count = counts[minority_label]
    minority_ds = ds.filter(lambda ex: int(ex["labels"]) == minority_label)
    majority_ds = (
        ds.filter(lambda ex: int(ex["labels"]) == majority_label)
        .shuffle(seed=seed)
        .select(range(minority_count))
    )
    return concatenate_datasets([minority_ds, majority_ds]).shuffle(seed=seed)


def to_lora_examples(split, answer_suffix: str) -> list[LoraExample]:
    examples = []
    for idx in range(len(split)):
        examples.append(
            LoraExample(
                id=split[idx]["id"],
                source_id=split[idx]["source_id"],
                text=split[idx]["txt"],
                label=int(split[idx]["labels"]),
                answer_suffix=answer_suffix,
            )
        )
    return examples


def format_sciq_paper_style(ex: dict, row_id: int, rng: random.Random, use_support: bool) -> dict:
    """Format SciQ as binary candidate-answer correctness.

    The no-support default matches the original datacentric_w2s `sciq`
    formatter: `Q: {question} A: {candidate}`. The support-passage variant is
    kept as an explicit ablation because it can make the binary task easier.
    """

    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = ex["correct_answer"]
    else:
        distractors = [ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        ans = rng.choice(distractors)

    if use_support:
        support = (ex.get("support") or "").strip()
        support_block = f"Support: {support}\n\n" if support else ""
    else:
        support_block = ""
    txt = f"{support_block}Q: {ex['question']} A: {ans}"
    options = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
    return {
        "id": f"sciq-{row_id}",
        "source_id": f"sciq-{row_id}",
        "txt": txt,
        "labels": hard_label,
        "gt_labels": hard_label,
        "mc_options": [f"{support_block}Q: {ex['question']} A: {o}" for o in options],
        "mc_correct": 0,
    }


def load_and_process_sciq_split(split: str, n_docs: int, seed: int, use_support: bool) -> Dataset:
    raw = load_dataset("allenai/sciq", split=split).shuffle(seed=seed)
    if len(raw) < n_docs:
        print(f"sciq/{split} has < {n_docs} raw docs, using all {len(raw)}")
    raw = raw.select(range(min(n_docs, len(raw))))
    rng = random.Random(seed)
    formatted_rows = [
        format_sciq_paper_style(ex, row_id=i, rng=rng, use_support=use_support)
        for i, ex in enumerate(raw)
    ]
    ds = Dataset.from_list(formatted_rows)
    ds = ds.filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    return ds


def load_paper_sciq_splits(
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    use_support: bool,
) -> SplitBundle:
    train_pool = load_and_process_sciq_split("train", n_train + n_val, seed, use_support)
    test = load_and_process_sciq_split("test", n_test, seed + 2, use_support)

    val_count = min(n_val, len(train_pool))
    val = train_pool.select(range(val_count))
    train = train_pool.select(range(val_count, len(train_pool)))
    train_halves = train.train_test_split(test_size=0.5, seed=seed)

    return SplitBundle(
        weak_train=train_halves["train"],
        strong_train=train_halves["test"],
        val=val,
        test=test,
    )


def load_sciq_multichoice_eval(n_questions: int, seed: int, use_support: bool) -> list[dict]:
    """Per-question multiple-choice eval for SciQ (4 options).

    For each test question, emit all 4 candidates (correct_answer + 3 distractors),
    sharing source_id, label=1 on the correct one, same prompt format as training
    (no support by default). argmax of P(correct) over a question's 4 candidates gives
    a true 4-way multiple-choice prediction (scored by accuracy_3class's argmax)."""
    raw = load_dataset("allenai/sciq", split="test").shuffle(seed=seed)
    rng = random.Random(seed + 7)
    out: list[dict] = []
    n = 0
    for ex in raw:
        if n >= n_questions:
            break
        if use_support:
            support = (ex.get("support") or "").strip()
            support_block = f"Support: {support}\n\n" if support else ""
        else:
            support_block = ""
        options = [ex["correct_answer"], ex["distractor1"], ex["distractor2"], ex["distractor3"]]
        # Shuffle the emission order: argmax tie-breaks pick the FIRST max, so a fixed
        # correct-first order would systematically resolve fp16 probability ties in favor
        # of the correct answer and inflate accuracy_3class.
        order = rng.sample(range(4), 4)
        for slot, oi in enumerate(order):
            txt = f"{support_block}Q: {ex['question']} A: {options[oi]}"
            out.append(
                {
                    "id": f"sciq-mc-{n}-{slot}",
                    "source_id": f"sciq-mc-{n}",
                    "txt": txt,
                    "labels": int(oi == 0),  # options[0] is the correct_answer
                }
            )
        n += 1
    return out


def format_hellaswag_paper_style(ex: dict, row_id: int, rng: random.Random) -> dict:
    """Format HellaSwag as binary candidate-ending correctness: '{context} {ending}'."""
    endings = list(ex["endings"])
    correct = int(ex["label"])
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        ans = endings[correct]
    else:
        wrong = [e for i, e in enumerate(endings) if i != correct]
        ans = rng.choice(wrong)
    ctx = (ex.get("ctx") or "").strip()
    txt = f"{ctx} {ans}".strip()
    return {
        "id": f"hellaswag-{row_id}",
        "source_id": f"hellaswag-{row_id}",
        "txt": txt,
        "labels": hard_label,
        "gt_labels": hard_label,
        "mc_options": [f"{ctx} {e}".strip() for e in endings],
        "mc_correct": correct,
    }


def load_and_process_hellaswag_split(split: str, n_docs: int, seed: int) -> Dataset:
    raw = load_dataset("Rowan/hellaswag", split=split).shuffle(seed=seed)
    raw = raw.filter(lambda ex: ex["label"] != "")  # test labels are hidden -> drop them
    if len(raw) < n_docs:
        print(f"hellaswag/{split} has < {n_docs} labeled docs, using all {len(raw)}")
    raw = raw.select(range(min(n_docs, len(raw))))
    rng = random.Random(seed)
    rows = [format_hellaswag_paper_style(ex, row_id=i, rng=rng) for i, ex in enumerate(raw)]
    ds = Dataset.from_list(rows)
    ds = ds.filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    return ds


def load_paper_hellaswag_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    # HellaSwag test labels are hidden -> train pool from "train", eval from "validation".
    train_pool = load_and_process_hellaswag_split("train", n_train + n_val, seed)
    test = load_and_process_hellaswag_split("validation", n_test, seed + 2)
    val_count = min(n_val, len(train_pool))
    val = train_pool.select(range(val_count))
    train = train_pool.select(range(val_count, len(train_pool)))
    train_halves = train.train_test_split(test_size=0.5, seed=seed)
    return SplitBundle(
        weak_train=train_halves["train"],
        strong_train=train_halves["test"],
        val=val,
        test=test,
    )


def load_hellaswag_multichoice_eval(n_questions: int, seed: int) -> list[dict]:
    """Per-question 4-way multiple-choice eval for HellaSwag (pick the best of 4 endings)."""
    raw = load_dataset("Rowan/hellaswag", split="validation").shuffle(seed=seed)
    rng = random.Random(seed + 7)
    out: list[dict] = []
    n = 0
    for ex in raw:
        if n >= n_questions:
            break
        if ex["label"] == "":
            continue
        ctx = (ex.get("ctx") or "").strip()
        endings = list(ex["endings"])
        correct = int(ex["label"])
        order = rng.sample(range(len(endings)), len(endings))
        for slot, oi in enumerate(order):
            out.append(
                {
                    "id": f"hellaswag-mc-{n}-{slot}",
                    "source_id": f"hellaswag-mc-{n}",
                    "txt": f"{ctx} {endings[oi]}".strip(),
                    "labels": int(oi == correct),
                }
            )
        n += 1
    return out


ANLI_LABELS = {0: "entailment", 1: "neutral", 2: "contradiction"}


def load_anli_multichoice_eval(n_questions: int, seed: int, anli_round: str) -> list[dict]:
    """Per-question multiple-choice eval for ANLI (3 relations).

    For each test example, emit all 3 candidate relations (entailment / neutral /
    contradiction), sharing source_id, label=1 on the gold relation, same prompt
    format as training. Option order shuffled (seeded) so argmax tie-breaks are not
    biased toward a fixed position. argmax of P(correct) over the 3 -> 3-way NLI."""
    raw = load_dataset("facebook/anli", split=f"test_{anli_round}").shuffle(seed=seed)
    rng = random.Random(seed + 7)
    out: list[dict] = []
    n = 0
    for ex in raw:
        if n >= n_questions:
            break
        gold = int(ex["label"])
        if gold not in (0, 1, 2):
            continue
        order = rng.sample(range(3), 3)
        for slot, ri in enumerate(order):
            txt = (
                f"Premise: {ex['premise']}\n"
                f"Hypothesis: {ex['hypothesis']}\n"
                f"Q: What is the relationship from the premise to the hypothesis? A: {ANLI_LABELS[ri]}"
            )
            out.append(
                {
                    "id": f"anli-mc-{n}-{slot}",
                    "source_id": f"anli-mc-{n}",
                    "txt": txt,
                    "labels": int(ri == gold),
                }
            )
        n += 1
    return out


def format_anli_paper_style(ex: dict, row_id: int, rng: random.Random) -> dict:
    """Format ANLI (3-class NLI) as binary candidate-relation correctness.

    A candidate relation is sampled (50% the gold relation, 50% a wrong one) and
    the model judges whether it is correct, mirroring the SciQ candidate-answer
    reduction. ANLI is adversarial, so the weak model is near chance here -- the
    noisy-weak-label regime where overlap selection should matter if it is real.
    """
    gold = int(ex["label"])
    hard_label = int(rng.random() < 0.5)
    if hard_label:
        candidate = ANLI_LABELS[gold]
    else:
        wrong = [name for key, name in ANLI_LABELS.items() if key != gold]
        candidate = rng.choice(wrong)
    txt = (
        f"Premise: {ex['premise']}\n"
        f"Hypothesis: {ex['hypothesis']}\n"
        f"Q: What is the relationship from the premise to the hypothesis? A: {candidate}"
    )
    return {
        "id": f"anli-{row_id}",
        "source_id": f"anli-{row_id}",
        "txt": txt,
        "labels": hard_label,
        "gt_labels": hard_label,
        "mc_options": [
            f"Premise: {ex['premise']}\n"
            f"Hypothesis: {ex['hypothesis']}\n"
            f"Q: What is the relationship from the premise to the hypothesis? A: {ANLI_LABELS[k]}"
            for k in (0, 1, 2)
        ],
        "mc_correct": gold,
    }


def load_and_process_anli_split(split: str, n_docs: int, seed: int) -> Dataset:
    raw = load_dataset("facebook/anli", split=split).shuffle(seed=seed)
    raw = raw.filter(lambda ex: int(ex["label"]) in (0, 1, 2))
    rng = random.Random(seed)
    formatted_rows = [
        format_anli_paper_style(ex, row_id=i, rng=rng)
        for i, ex in enumerate(raw)
    ]
    ds = Dataset.from_list(formatted_rows)
    ds = ds.filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    if len(ds) < n_docs:
        print(f"anli/{split} has < {n_docs} docs after balancing, using all {len(ds)}")
    return ds.select(range(min(n_docs, len(ds))))


def load_paper_anli_splits(
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    anli_round: str,
) -> SplitBundle:
    train_pool = load_and_process_anli_split(f"train_{anli_round}", n_train + n_val, seed)
    test = load_and_process_anli_split(f"test_{anli_round}", n_test, seed + 2)

    val_count = min(n_val, len(train_pool))
    val = train_pool.select(range(val_count))
    train = train_pool.select(range(val_count, len(train_pool)))
    train_halves = train.train_test_split(test_size=0.5, seed=seed)

    return SplitBundle(
        weak_train=train_halves["train"],
        strong_train=train_halves["test"],
        val=val,
        test=test,
    )


WANLI_REL2IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}


def format_wanli_paper_style(ex: dict, row_id: int, rng: random.Random) -> dict:
    gold = WANLI_REL2IDX.get(ex["gold"])
    if gold is None:
        return {"id": "", "source_id": "", "txt": "", "labels": 0, "gt_labels": 0,
                "mc_options": [], "mc_correct": 0}
    hard_label = int(rng.random() < 0.5)
    candidate = ANLI_LABELS[gold] if hard_label else rng.choice(
        [name for key, name in ANLI_LABELS.items() if key != gold]
    )
    base = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\n"
    q = "Q: What is the relationship from the premise to the hypothesis? A:"
    return {
        "id": f"wanli-{row_id}", "source_id": f"wanli-{row_id}",
        "txt": f"{base}{q} {candidate}",
        "labels": hard_label, "gt_labels": hard_label,
        "mc_options": [f"{base}{q} {ANLI_LABELS[k]}" for k in (0, 1, 2)],
        "mc_correct": gold,
    }


# --- WANLI: worker-and-AI adversarial NLI (3-way), same candidate-relation reduction as ANLI.
def load_and_process_wanli_split(split: str, n_docs: int, seed: int) -> Dataset:
    raw = load_dataset("alisawuffles/WANLI", split=split).shuffle(seed=seed)
    raw = raw.filter(lambda ex: ex["gold"] in WANLI_REL2IDX)
    rng = random.Random(seed)
    rows = [format_wanli_paper_style(ex, i, rng) for i, ex in enumerate(raw)]
    ds = Dataset.from_list(rows).filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    if len(ds) < n_docs:
        print(f"wanli/{split} has < {n_docs} docs after balancing, using all {len(ds)}")
    return ds.select(range(min(n_docs, len(ds))))


def load_paper_wanli_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    train_pool = load_and_process_wanli_split("train", n_train + n_val, seed)
    test = load_and_process_wanli_split("test", n_test, seed + 2)
    val_count = min(n_val, len(train_pool))
    val = train_pool.select(range(val_count))
    train = train_pool.select(range(val_count, len(train_pool)))
    halves = train.train_test_split(test_size=0.5, seed=seed)
    return SplitBundle(weak_train=halves["train"], strong_train=halves["test"], val=val, test=test)


def load_wanli_multichoice_eval(n_questions: int, seed: int) -> list[dict]:
    raw = load_dataset("alisawuffles/WANLI", split="test").shuffle(seed=seed)
    rng = random.Random(seed + 7)
    out: list[dict] = []
    n = 0
    for ex in raw:
        if n >= n_questions:
            break
        gold = WANLI_REL2IDX.get(ex["gold"])
        if gold is None:
            continue
        base = f"Premise: {ex['premise']}\nHypothesis: {ex['hypothesis']}\n"
        q = "Q: What is the relationship from the premise to the hypothesis? A:"
        for slot, ri in enumerate(rng.sample(range(3), 3)):
            out.append({"id": f"wanli-mc-{n}-{slot}", "source_id": f"wanli-mc-{n}",
                        "txt": f"{base}{q} {ANLI_LABELS[ri]}", "labels": int(ri == gold)})
        n += 1
    return out


FEVER_REL2IDX = {"SUPPORTS": 0, "NOT ENOUGH INFO": 1, "REFUTES": 2}


FEVER_LABELS = ["supported", "not enough info", "refuted"]


def format_fevernli_paper_style(ex: dict, row_id: int, rng: random.Random) -> dict:
    gold = FEVER_REL2IDX.get(str(ex["fever_gold_label"]))
    ev = str(ex.get("premise", "")).strip()
    cl = str(ex.get("hypothesis", "")).strip()
    if gold is None or not ev or not cl:
        return {"id": "", "source_id": "", "txt": "", "labels": 0, "gt_labels": 0,
                "mc_options": [], "mc_correct": 0}
    hard_label = int(rng.random() < 0.5)
    candidate = FEVER_LABELS[gold] if hard_label else rng.choice(
        [FEVER_LABELS[k] for k in range(3) if k != gold]
    )
    base = f"Evidence: {ev}\nClaim: {cl}\n"
    q = "Q: Based on the evidence, the claim is? A:"
    return {
        "id": f"fevernli-{row_id}", "source_id": f"fevernli-{row_id}",
        "txt": f"{base}{q} {candidate}",
        "labels": hard_label, "gt_labels": hard_label,
        "mc_options": [f"{base}{q} {FEVER_LABELS[k]}" for k in range(3)],
        "mc_correct": gold,
    }


def load_and_process_fevernli_split(split: str, n_docs: int, seed: int) -> Dataset:
    raw = load_dataset("pietrolesci/nli_fever", split=split).shuffle(seed=seed)
    raw = raw.filter(lambda ex: str(ex["fever_gold_label"]) in FEVER_REL2IDX)
    rng = random.Random(seed)
    rows = [format_fevernli_paper_style(ex, i, rng) for i, ex in enumerate(raw)]
    ds = Dataset.from_list(rows).filter(lambda ex: ex["txt"] != "")
    ds = balance_binary_dataset(ds, seed)
    if len(ds) < n_docs:
        print(f"fevernli/{split} has < {n_docs} docs after balancing, using all {len(ds)}")
    return ds.select(range(min(n_docs, len(ds))))


def load_paper_fevernli_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    # nli_fever: labeled 'train' + 'dev' (test is unlabeled). Carve test from dev.
    train_pool = load_and_process_fevernli_split("train", n_train + n_val, seed)
    test = load_and_process_fevernli_split("dev", n_test, seed + 2)
    val_count = min(n_val, len(train_pool))
    val = train_pool.select(range(val_count))
    train = train_pool.select(range(val_count, len(train_pool)))
    halves = train.train_test_split(test_size=0.5, seed=seed)
    return SplitBundle(weak_train=halves["train"], strong_train=halves["test"], val=val, test=test)


def load_fevernli_multichoice_eval(n_questions: int, seed: int) -> list[dict]:
    raw = load_dataset("pietrolesci/nli_fever", split="dev").shuffle(seed=seed)
    rng = random.Random(seed + 7)
    out: list[dict] = []
    n = 0
    for ex in raw:
        if n >= n_questions:
            break
        gold = FEVER_REL2IDX.get(str(ex["fever_gold_label"]))
        ev = str(ex.get("premise", "")).strip()
        cl = str(ex.get("hypothesis", "")).strip()
        if gold is None or not ev or not cl:
            continue
        base = f"Evidence: {ev}\nClaim: {cl}\n"
        q = "Q: Based on the evidence, the claim is? A:"
        for slot, ri in enumerate(rng.sample(range(3), 3)):
            out.append({"id": f"fevernli-mc-{n}-{slot}", "source_id": f"fevernli-mc-{n}",
                        "txt": f"{base}{q} {FEVER_LABELS[ri]}", "labels": int(ri == gold)})
        n += 1
    return out


IMPPRES_CONFIGS = [
    "presupposition_all_n_presupposition", "presupposition_both_presupposition",
    "presupposition_change_of_state", "presupposition_cleft_existence",
    "presupposition_cleft_uniqueness", "presupposition_only_presupposition",
    "presupposition_possessed_definites_existence", "presupposition_possessed_definites_uniqueness",
    "presupposition_question_presupposition",
]


def _imppres_rows(seed: int):
    """Merged presupposition-section rows -> (premise, hypothesis, gold 0/1/2), shuffled."""
    import random as _random

    rows = []
    for cfg in IMPPRES_CONFIGS:
        dd = load_dataset("facebook/imppres", cfg)
        ds = dd[list(dd.keys())[0]]
        for ex in ds:
            try:
                gold = int(str(ex.get("gold_label", "")).strip())
            except ValueError:
                continue
            prem = str(ex.get("premise", "")).strip()
            hyp = str(ex.get("hypothesis", "")).strip()
            if gold not in (0, 1, 2) or not prem or not hyp:
                continue
            rows.append((prem, hyp, gold))
    _random.Random(seed).shuffle(rows)
    return rows


def format_imppres_paper_style(prem: str, hyp: str, gold: int, row_id: int, rng: random.Random) -> dict:
    hard_label = int(rng.random() < 0.5)
    candidate = ANLI_LABELS[gold] if hard_label else rng.choice(
        [name for key, name in ANLI_LABELS.items() if key != gold]
    )
    base = f"Premise: {prem}\nHypothesis: {hyp}\n"
    q = "Q: What is the relationship from the premise to the hypothesis? A:"
    return {
        "id": f"imppres-{row_id}", "source_id": f"imppres-{row_id}",
        "txt": f"{base}{q} {candidate}",
        "labels": hard_label, "gt_labels": hard_label,
        "mc_options": [f"{base}{q} {ANLI_LABELS[k]}" for k in (0, 1, 2)],
        "mc_correct": gold,
    }


def load_paper_imppres_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    # ImpPres is an eval-only templated corpus (~17k presupposition rows); carve
    # disjoint train/test pools from one shuffle. Templated data -> watch for
    # template memorization in Stage-2 (GT arm may saturate).
    rows = _imppres_rows(seed)
    test_rows = rows[:max(1, n_test * 2)]
    pool_rows = rows[max(1, n_test * 2):]
    rng = random.Random(seed)
    fmt_pool = [format_imppres_paper_style(p, h, g, i, rng) for i, (p, h, g) in enumerate(pool_rows)]
    fmt_test = [format_imppres_paper_style(p, h, g, 10_000_000 + i, rng) for i, (p, h, g) in enumerate(test_rows)]
    pool = balance_binary_dataset(Dataset.from_list(fmt_pool).filter(lambda ex: ex["txt"] != ""), seed)
    test = balance_binary_dataset(Dataset.from_list(fmt_test).filter(lambda ex: ex["txt"] != ""), seed + 2)
    pool = pool.select(range(min(n_train + n_val, len(pool))))
    test = test.select(range(min(n_test, len(test))))
    val_count = min(n_val, len(pool))
    val = pool.select(range(val_count))
    train = pool.select(range(val_count, len(pool)))
    halves = train.train_test_split(test_size=0.5, seed=seed)
    return SplitBundle(weak_train=halves["train"], strong_train=halves["test"], val=val, test=test)


def load_imppres_multichoice_eval(n_questions: int, seed: int, n_test: int) -> list[dict]:
    # The questions come from the head of the split's own shuffle, the rows
    # load_paper_imppres_splits keeps out of the training pool. The corpus repeats
    # a few premise-hypothesis pairs, so a head row whose pair recurs in the pool
    # is skipped as well.
    rows = _imppres_rows(seed)
    pool_pairs = {(p, h) for p, h, _ in rows[max(1, n_test * 2):]}
    rows = [r for r in rows[:max(1, n_test * 2)] if r[:2] not in pool_pairs][:n_questions]
    rng = random.Random(seed + 7)
    out: list[dict] = []
    for n_i, (prem, hyp, gold) in enumerate(rows):
        base = f"Premise: {prem}\nHypothesis: {hyp}\n"
        q = "Q: What is the relationship from the premise to the hypothesis? A:"
        for slot, ri in enumerate(rng.sample(range(3), 3)):
            out.append({"id": f"imppres-mc-{n_i}-{slot}", "source_id": f"imppres-mc-{n_i}",
                        "txt": f"{base}{q} {ANLI_LABELS[ri]}", "labels": int(ri == gold)})
    return out


def _aquarat_rows(splits, seed: int):
    """AQuA-RAT rows -> (question, [5 option texts], gold_idx), merged over the given splits."""
    import random as _random

    rows = []
    for sp in splits:
        ds = load_dataset("deepmind/aqua_rat", "raw", split=sp)
        for ex in ds:
            opts_raw = [str(o).strip() for o in ex.get("options", [])]
            options = [o.split(")", 1)[1].strip() if ")" in o else o for o in opts_raw]
            gold = "ABCDE".find(str(ex.get("correct", "")).strip())
            q = str(ex.get("question", "")).strip()
            if not (0 <= gold < len(options)) or not q or len(options) != 5:
                continue
            rows.append((q, options, gold))
    _random.Random(seed).shuffle(rows)
    return rows


def format_aquarat_paper_style(q: str, options: list, gold: int, row_id: int, rng: random.Random) -> dict:
    hard_label = int(rng.random() < 0.5)
    cand = options[gold] if hard_label else options[rng.choice([k for k in range(len(options)) if k != gold])]
    base = f"Q: {q} A:"
    return {
        "id": f"aquarat-{row_id}", "source_id": f"aquarat-{row_id}",
        "txt": f"{base} {cand}",
        "labels": hard_label, "gt_labels": hard_label,
        "mc_options": [f"{base} {o}" for o in options],
        "mc_correct": gold,
    }


def load_paper_aquarat_splits(n_train: int, n_val: int, n_test: int, seed: int) -> SplitBundle:
    # 97k crowd-amplified train pool; official test+validation (254+254) merged as the test set.
    train_rows = _aquarat_rows(["train"], seed)[: (n_train + n_val) * 2]
    test_rows = _aquarat_rows(["test", "validation"], seed + 2)
    rng = random.Random(seed)
    pool = [format_aquarat_paper_style(q, o, g, i, rng) for i, (q, o, g) in enumerate(train_rows)]
    tst = [format_aquarat_paper_style(q, o, g, 10_000_000 + i, rng) for i, (q, o, g) in enumerate(test_rows)]
    pool_ds = balance_binary_dataset(Dataset.from_list(pool).filter(lambda ex: ex["txt"] != ""), seed)
    test_ds = balance_binary_dataset(Dataset.from_list(tst).filter(lambda ex: ex["txt"] != ""), seed + 2)
    pool_ds = pool_ds.select(range(min(n_train + n_val, len(pool_ds))))
    test_ds = test_ds.select(range(min(n_test, len(test_ds))))
    val_count = min(n_val, len(pool_ds))
    val = pool_ds.select(range(val_count))
    train = pool_ds.select(range(val_count, len(pool_ds)))
    halves = train.train_test_split(test_size=0.5, seed=seed)
    return SplitBundle(weak_train=halves["train"], strong_train=halves["test"], val=val, test=test_ds)


def load_aquarat_multichoice_eval(n_questions: int, seed: int) -> list[dict]:
    rows = _aquarat_rows(["test", "validation"], seed)[:n_questions]
    rng = random.Random(seed + 7)
    out: list[dict] = []
    for n_i, (q, options, gold) in enumerate(rows):
        base = f"Q: {q} A:"
        for slot, ri in enumerate(rng.sample(range(len(options)), len(options))):
            out.append({"id": f"aquarat-mc-{n_i}-{slot}", "source_id": f"aquarat-mc-{n_i}",
                        "txt": f"{base} {options[ri]}", "labels": int(ri == gold)})
    return out


def quail_rows(split: str, seed: int):
    import random as _random

    ds = load_dataset("textmachinelab/quail", "default", split=split, revision="refs/convert/parquet")
    rows = []
    for ex in ds:
        options = [str(o).strip() for o in ex.get("answers", [])]
        gold = ex.get("correct_answer_id", -1)
        c = str(ex.get("context", "")).strip()
        q = str(ex.get("question", "")).strip()
        if not isinstance(gold, int) or not (0 <= gold < len(options)) or not c or not q:
            continue
        rows.append((f"{c}\nQ: {q} A:", options, gold))
    _random.Random(seed).shuffle(rows)
    return rows


def riddlesense_rows(split: str, seed: int):
    import random as _random

    ds = load_dataset("INK-USC/riddle_sense", "default", split=split, revision="refs/convert/parquet")
    rows = []
    for ex in ds:
        labels = list(ex["choices"]["label"])
        options = [str(t).strip() for t in ex["choices"]["text"]]
        key = str(ex["answerKey"]).strip()
        q = str(ex.get("question", "")).strip()
        if key not in labels or not options or not q:
            continue
        rows.append((f"Q: {q} A:", options, labels.index(key)))
    _random.Random(seed).shuffle(rows)
    return rows


def csqa_rows(split: str, seed: int):
    import random as _random

    ds = load_dataset("tau/commonsense_qa", split=split, revision="refs/convert/parquet")
    rows = []
    for ex in ds:
        labels = list(ex["choices"]["label"])
        options = [str(t).strip() for t in ex["choices"]["text"]]
        key = str(ex["answerKey"]).strip()
        q = str(ex.get("question", "")).strip()
        if key not in labels or not options or not q:
            continue
        rows.append((f"Q: {q} A:", options, labels.index(key)))
    _random.Random(seed).shuffle(rows)
    return rows


def _format_mc_rows_paper_style(rows, prefix: str, rng: random.Random, id_base: int = 0):
    out = []
    for i, (base, options, gold) in enumerate(rows):
        hard_label = int(rng.random() < 0.5)
        cand = options[gold] if hard_label else options[rng.choice([k for k in range(len(options)) if k != gold])]
        out.append({
            "id": f"{prefix}-{id_base + i}", "source_id": f"{prefix}-{id_base + i}",
            "txt": f"{base} {cand}",
            "labels": hard_label, "gt_labels": hard_label,
            "mc_options": [f"{base} {o}" for o in options],
            "mc_correct": gold,
        })
    return out


def _generic_mc_splits(train_rows, test_rows, prefix, n_train, n_val, n_test, seed) -> SplitBundle:
    rng = random.Random(seed)
    pool = _format_mc_rows_paper_style(train_rows, prefix, rng)
    tst = _format_mc_rows_paper_style(test_rows, prefix, rng, id_base=10_000_000)
    pool_ds = balance_binary_dataset(Dataset.from_list(pool).filter(lambda ex: ex["txt"] != ""), seed)
    test_ds = balance_binary_dataset(Dataset.from_list(tst).filter(lambda ex: ex["txt"] != ""), seed + 2)
    pool_ds = pool_ds.select(range(min(n_train + n_val, len(pool_ds))))
    test_ds = test_ds.select(range(min(n_test, len(test_ds))))
    val_count = min(n_val, len(pool_ds))
    val = pool_ds.select(range(val_count))
    train = pool_ds.select(range(val_count, len(pool_ds)))
    halves = train.train_test_split(test_size=0.5, seed=seed)
    return SplitBundle(weak_train=halves["train"], strong_train=halves["test"], val=val, test=test_ds)


def generic_mc_eval(rows, prefix, n_questions, seed) -> list[dict]:
    rng = random.Random(seed + 7)
    out: list[dict] = []
    for n_i, (base, options, gold) in enumerate(rows[:n_questions]):
        for slot, ri in enumerate(rng.sample(range(len(options)), len(options))):
            out.append({"id": f"{prefix}-mc-{n_i}-{slot}", "source_id": f"{prefix}-mc-{n_i}",
                        "txt": f"{base} {options[ri]}", "labels": int(ri == gold)})
    return out


def scinli_rows(split: str, seed: int):
    import random as _random

    ds = load_dataset("tasksource/scinli", split=split)
    opts = ["entailment", "contrasting", "reasoning", "neutral"]
    lab2idx = {o: i for i, o in enumerate(opts)}
    rows = []
    for ex in ds:
        gold = lab2idx.get(str(ex.get("label", "")).strip().lower())
        s1 = str(ex.get("sentence1", "")).strip()
        s2 = str(ex.get("sentence2", "")).strip()
        if gold is None or not s1 or not s2:
            continue
        base = (f"Sentence 1: {s1}\nSentence 2: {s2}\n"
                "Q: What is the relationship from sentence 1 to sentence 2? A:")
        rows.append((base, list(opts), gold))
    _random.Random(seed).shuffle(rows)
    return rows


def load_paper_scinli_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(scinli_rows("train", seed), scinli_rows("test", seed + 2),
                              "scinli", n_train, n_val, n_test, seed)


def load_paper_quail_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(quail_rows("train", seed), quail_rows("validation", seed + 2),
                              "quail", n_train, n_val, n_test, seed)


def load_paper_riddlesense_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(riddlesense_rows("train", seed), riddlesense_rows("validation", seed + 2),
                              "riddlesense", n_train, n_val, n_test, seed)


def load_paper_csqa_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(csqa_rows("train", seed), csqa_rows("validation", seed + 2),
                              "csqa", n_train, n_val, n_test, seed)


def cosmosqa_rows(split: str, seed: int):
    import random as _random

    ds = load_dataset("allenai/cosmos_qa", split=split, revision="refs/convert/parquet")
    rows = []
    for ex in ds:
        options = [str(ex.get(f"answer{k}", "")).strip() for k in range(4)]
        gold = int(ex.get("label", -1))
        ctx = str(ex.get("context", "")).strip()
        q = str(ex.get("question", "")).strip()
        if not ctx or not q or not all(options) or gold not in (0, 1, 2, 3):
            continue
        rows.append((f"Context: {ctx}\nQ: {q} A:", options, gold))
    _random.Random(seed).shuffle(rows)
    return rows


def load_paper_cosmosqa_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(cosmosqa_rows("train", seed), cosmosqa_rows("validation", seed + 2),
                              "cosmosqa", n_train, n_val, n_test, seed)


def blimp_rows(split: str, seed: int):
    """All 67 BLiMP paradigms pooled; each pair becomes a 2-option acceptability item.
    BLiMP ships a single split per paradigm, so the shuffled tail serves as the eval side."""
    import random as _random

    from datasets import get_dataset_config_names

    rows = []
    for cfg in sorted(get_dataset_config_names("nyu-mll/blimp")):
        ds = load_dataset("nyu-mll/blimp", cfg, split="train")
        for ex in ds:
            good = str(ex.get("sentence_good", "")).strip()
            bad = str(ex.get("sentence_bad", "")).strip()
            if not good or not bad:
                continue
            rows.append(("Q: Which sentence is grammatical? A:", [good, bad], 0))
    _random.Random(seed).shuffle(rows)
    cut = max(1, int(len(rows) * 0.85))
    return rows[:cut] if split == "train" else rows[cut:]


def load_paper_blimp_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(blimp_rows("train", seed), blimp_rows("validation", seed),
                              "blimp", n_train, n_val, n_test, seed)


def sst2_rows(split: str, seed: int):
    import random as _random

    ds = load_dataset("nyu-mll/glue", "sst2", split=split)
    rows = []
    for ex in ds:
        s = str(ex.get("sentence", "")).strip()
        gold = int(ex.get("label", -1))
        if not s or gold not in (0, 1):
            continue
        base = f"Sentence: {s}\nQ: What is the sentiment of this sentence? A:"
        rows.append((base, ["positive", "negative"], 0 if gold == 1 else 1))
    _random.Random(seed).shuffle(rows)
    return rows


def load_paper_sst2_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(sst2_rows("train", seed), sst2_rows("validation", seed + 2),
                              "sst2", n_train, n_val, n_test, seed)


def _glue_pair_rows(config: str, split: str, seed: int, fields, question: str, opts):
    import random as _random

    ds = load_dataset("nyu-mll/glue", config, split=split)
    f1, f2 = fields
    rows = []
    for ex in ds:
        a = str(ex.get(f1, "")).strip()
        b = str(ex.get(f2, "")).strip()
        gold = int(ex.get("label", -1))
        if not a or not b or gold not in (0, 1):
            continue
        base = f"Sentence 1: {a}\nSentence 2: {b}\nQ: {question} A:"
        rows.append((base, list(opts), gold))
    _random.Random(seed).shuffle(rows)
    return rows


def qnli_rows(split: str, seed: int):
    return _glue_pair_rows("qnli", split, seed, ("question", "sentence"),
                           "Does sentence 2 answer the question in sentence 1?", ["yes", "no"])


def load_paper_qnli_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(qnli_rows("train", seed), qnli_rows("validation", seed + 2),
                              "qnli", n_train, n_val, n_test, seed)


def agnews_rows(split: str, seed: int):
    import random as _random

    ds = load_dataset("fancyzhx/ag_news", split=split)
    opts = ["world news", "sports", "business", "science and technology"]
    rows = []
    for ex in ds:
        txt = str(ex.get("text", "")).strip()
        gold = int(ex.get("label", -1))
        if not txt or gold not in (0, 1, 2, 3):
            continue
        base = f"Article: {txt}\nQ: What is the topic of this article? A:"
        rows.append((base, list(opts), gold))
    _random.Random(seed).shuffle(rows)
    return rows


def load_paper_agnews_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(agnews_rows("train", seed), agnews_rows("test", seed + 2),
                              "agnews", n_train, n_val, n_test, seed)


def imdb_rows(split: str, seed: int):
    """Full reviews. An earlier version cut them to 800 characters, which truncated 63% of
    the corpus and kept only a third of the 512-token budget the IMDb literature settled on;
    worse, it was a head-only cut, the one strategy measured worst on this dataset because
    sentiment often lands in the closing lines. Length is now bounded by max_length with
    left truncation, so what gets dropped is the opening of the passage rather than the
    question and the candidate answer."""
    import random as _random

    ds = load_dataset("stanfordnlp/imdb", split=split)
    rows = []
    for ex in ds:
        txt = str(ex.get("text", "")).strip()
        gold = int(ex.get("label", -1))
        if not txt or gold not in (0, 1):
            continue
        base = f"Review: {txt}\nQ: What is the sentiment of this review? A:"
        rows.append((base, ["negative", "positive"], gold))
    _random.Random(seed).shuffle(rows)
    return rows


def load_paper_imdb_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(imdb_rows("train", seed), imdb_rows("test", seed + 2),
                              "imdb", n_train, n_val, n_test, seed)


def tweetsent_rows(split: str, seed: int):
    """TweetEval sentiment (SemEval-2017 Task 4A), three classes. The classes are uneven
    (neutral is 45% of train) and I leave them so: the binary pairs are balanced by
    construction anyway, and the question-level score keeps the benchmark's own mix."""
    import random as _random

    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split)
    opts = ["negative", "neutral", "positive"]
    rows = []
    for ex in ds:
        txt = str(ex.get("text", "")).strip()
        gold = int(ex.get("label", -1))
        if not txt or gold not in (0, 1, 2):
            continue
        base = f"Tweet: {txt}\nQ: What is the sentiment of this tweet? A:"
        rows.append((base, list(opts), gold))
    _random.Random(seed).shuffle(rows)
    return rows


def load_paper_tweetsent_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(tweetsent_rows("train", seed), tweetsent_rows("test", seed + 2),
                              "tweetsent", n_train, n_val, n_test, seed)


def wikitoxic_rows(split: str, seed: int):
    """Wikipedia talk-page comments from the Jigsaw toxic-comment challenge, in the cleaned
    wiki_toxic release. About 10% of each split is toxic, so I cut every split to equal
    classes: otherwise answering "non-toxic" everywhere scores 0.90 and the gap between the
    weak and the gold model has no room to show. The release also ships a balanced_train
    split, but its parquet copy, the only form current `datasets` will load, lists just
    train, validation and test. Long comments stay whole and are bounded by max_length with
    left truncation, as for IMDb."""
    import random as _random

    ds = load_dataset("OxAISH-AL-LLM/wiki_toxic", "default", split=split, revision="refs/convert/parquet")
    rows = []
    for ex in ds:
        txt = str(ex.get("comment_text", "")).strip()
        gold = int(ex.get("label", -1))
        if not txt or gold not in (0, 1):
            continue
        base = f"Comment: {txt}\nQ: Is this comment toxic or non-toxic? A:"
        rows.append((base, ["non-toxic", "toxic"], gold))
    _random.Random(seed).shuffle(rows)
    by_class = [[r for r in rows if r[2] == k] for k in (0, 1)]
    m = min(len(c) for c in by_class)
    rows = by_class[0][:m] + by_class[1][:m]
    _random.Random(seed + 1).shuffle(rows)
    return rows


def load_paper_wikitoxic_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(wikitoxic_rows("train", seed), wikitoxic_rows("test", seed + 2),
                              "wikitoxic", n_train, n_val, n_test, seed)


def scitail_rows(split: str, seed: int):
    """SciTail, science-exam entailment in the tsv release (entails / neutral). A few train premises
    are pasted web pages of tens of thousands of characters; rows with a premise over 1500 characters
    are dropped rather than left to the truncation, which would keep only their tail."""
    import random as _random

    ds = load_dataset("allenai/scitail", "tsv_format", split=split)
    rows = []
    for ex in ds:
        p_ = str(ex.get("premise", "")).strip()
        h = str(ex.get("hypothesis", "")).strip()
        lab = str(ex.get("label", "")).strip()
        if not p_ or not h or lab not in ("entails", "neutral") or len(p_) > 1500:
            continue
        base = f"Premise: {p_}\nHypothesis: {h}\nQ: Does the premise entail the hypothesis? A:"
        rows.append((base, ["no", "yes"], 1 if lab == "entails" else 0))
    _random.Random(seed).shuffle(rows)
    return rows


def load_paper_scitail_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(scitail_rows("train", seed), scitail_rows("test", seed + 2),
                              "scitail", n_train, n_val, n_test, seed)


def boolq_rows(split: str, seed: int):
    """BoolQ, yes/no questions over a Wikipedia passage. 62% of the answers are yes; the pairs are
    balanced by construction and the question-level score is left at the benchmark's own mix."""
    import random as _random

    ds = load_dataset("google/boolq", split=split)
    rows = []
    for ex in ds:
        q = str(ex.get("question", "")).strip()
        p_ = str(ex.get("passage", "")).strip()
        ans = ex.get("answer")
        if not q or not p_ or ans is None:
            continue
        base = f"Passage: {p_}\nQuestion: {q}?\nQ: Is the answer yes or no? A:"
        rows.append((base, ["no", "yes"], 1 if bool(ans) else 0))
    _random.Random(seed).shuffle(rows)
    return rows


def load_paper_boolq_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(boolq_rows("train", seed), boolq_rows("validation", seed + 2),
                              "boolq", n_train, n_val, n_test, seed)


def ethicsjustice_rows(split: str, seed: int):
    """ETHICS, justice subset: a first-person claim of desert or impartiality, labelled reasonable (1)
    or not (0). One of the four ETHICS subsets in the original weak-to-strong paper's dataset list."""
    import random as _random

    # the repository is a loading script, which current `datasets` refuses; its parquet branch
    # names the config as a folder, which the builder does not resolve, so the file is named here
    ds = load_dataset("parquet", split="d", data_files={"d": f"hf://datasets/hendrycks/ethics@refs%2Fconvert%2Fparquet/justice/{split}/0000.parquet"})
    rows = []
    for ex in ds:
        sc = str(ex.get("scenario", "")).strip()
        gold = int(ex.get("label", -1))
        if not sc or gold not in (0, 1):
            continue
        base = f"Statement: {sc}\nQ: Is this claim reasonable or unreasonable? A:"
        rows.append((base, ["unreasonable", "reasonable"], gold))
    _random.Random(seed).shuffle(rows)
    return rows


def load_paper_ethicsjustice_splits(n_train, n_val, n_test, seed) -> SplitBundle:
    return _generic_mc_splits(ethicsjustice_rows("train", seed), ethicsjustice_rows("test", seed + 2),
                              "ethicsjustice", n_train, n_val, n_test, seed)


def load_paper_style_splits(args: argparse.Namespace) -> SplitBundle:
    if args.dataset == "sciq":
        return load_paper_sciq_splits(
            args.n_train,
            args.n_val,
            args.n_test,
            args.seed,
            args.sciq_use_support,
        )
    if args.dataset == "anli":
        return load_paper_anli_splits(
            args.n_train, args.n_val, args.n_test, args.seed, args.anli_round
        )
    if args.dataset == "hellaswag":
        return load_paper_hellaswag_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "wanli":
        return load_paper_wanli_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "fevernli":
        return load_paper_fevernli_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "imppres":
        return load_paper_imppres_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "aquarat":
        return load_paper_aquarat_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "quail":
        return load_paper_quail_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "riddlesense":
        return load_paper_riddlesense_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "scinli":
        return load_paper_scinli_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "csqa":
        return load_paper_csqa_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "cosmosqa":
        return load_paper_cosmosqa_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "blimp":
        return load_paper_blimp_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "sst2":
        return load_paper_sst2_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "qnli":
        return load_paper_qnli_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "agnews":
        return load_paper_agnews_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "imdb":
        return load_paper_imdb_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "tweetsent":
        return load_paper_tweetsent_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "wikitoxic":
        return load_paper_wikitoxic_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "scitail":
        return load_paper_scitail_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "boolq":
        return load_paper_boolq_splits(args.n_train, args.n_val, args.n_test, args.seed)
    if args.dataset == "ethicsjustice":
        return load_paper_ethicsjustice_splits(args.n_train, args.n_val, args.n_test, args.seed)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def split_texts(splits: SplitBundle) -> dict[str, list[str]]:
    return {
        "weak_train": list(splits.weak_train["txt"]),
        "strong_train": list(splits.strong_train["txt"]),
        "test": list(splits.test["txt"]),
    }


def split_labels(split) -> np.ndarray:
    return np.array(split["labels"], dtype=int)
