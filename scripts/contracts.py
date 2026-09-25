#!/usr/bin/env python3
from dataclasses import dataclass

from datasets import Dataset


@dataclass
class LoraExample:
    id: str
    source_id: str
    text: str
    label: int
    answer_suffix: str

    @property
    def prompt(self) -> str:
        return f"{self.text}{self.answer_suffix}"


@dataclass(frozen=True)
class SplitBundle:
    weak_train: Dataset
    strong_train: Dataset
    val: Dataset
    test: Dataset


CUSTOM_SLOTS = ("cs1", "cs2", "cs3", "cs4")

# Every RNG stream is args.seed + offset (+ fi for per-keep-fraction arms,
# + 1000*slot for the custom slots, + target percent and draw for purity, noted inline).
# One entry per stream that exists; the uniqueness assert below is the whole
# point, so add new arms here first and let a collision fail at import time.
SEED_OFFSETS = {
    "confidence_middle": 2000,
    "confidence_high": 3000,
    "knn_middle": 4000,
    "knn_mixed": 5000,
    "knn_high": 5500,
    "committee_agree": 7000,
    "committee_disagree": 8000,
    "committee_fit": 9000,
    "kf_committee_agree": 10000,
    "kf_committee_disagree": 11000,
    "kf_committee_top": 12000,
    "kf_knn_high": 13000,
    "kf_knn_low": 13500,
    "kf_confidence_high": 14000,
    "kf_confidence_low": 14500,
    "rp_high": 16000,
    "rp_low": 16500,
    "cca_high": 22000,
    "cca_low": 22500,
    "l2resid_high": 23000,
    "l2resid_low": 23500,
    "ccarobust_high": 26000,
    "ccarobust_low": 26500,
    "mlpstep1_high": 27000,
    "mlpstep1_low": 27500,
    "linstep1_high": 29000,
    "linstep1_low": 29500,
    "custom_high": 31000,           # + 1000*slot, slots 0..3 reach 34000
    "custom_low": 31500,            # + 1000*slot, slots 0..3 reach 34500
    "purity_flip": 36000,           # + target percent per grid point
    "purity_iid": 37000,            # + target percent + 200*draw
    "toklen_high": 38000,
    "toklen_low": 38500,
    "elloss_high": 39000,
    "elloss_low": 39500,
    "overlap_high": 40000,
    "overlap_low": 40500,
    "kf_weak_correct": 41000,       # + fi per keep fraction
}

_vals = list(SEED_OFFSETS.values())
assert len(set(_vals)) == len(_vals), "seed offsets collide; pick a fresh stream"
