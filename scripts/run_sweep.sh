#!/usr/bin/env bash
set -euo pipefail

# The canonical sweep: one dataset, the paper's selection arms at the 50% budget, three
# training seeds. Sets the defaults and hands the dataset to run_seed_loop.sh,
# which runs the baseline once and then one filtering run per training seed.
#
# Every setting is an environment variable; scripts/reproduce.sh sets them for each result
# of the paper. Quick look:
#   DATASET=wanli TRAIN_SEEDS=42 bash scripts/run_sweep.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export DATASET="${DATASET:-wanli}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/${DATASET}}"
export TRAIN_SEEDS="${TRAIN_SEEDS:-42 123 456}"
export N_TRAIN="${N_TRAIN:-6000}"
export N_VAL="${N_VAL:-1000}"
export N_TEST="${N_TEST:-1000}"
# Every reported run scored all of its test questions; the wrapper alone would stop at 2041.
export N_EVAL_QUESTIONS="${N_EVAL_QUESTIONS:-$N_TEST}"
export EVAL_3CLASS="${EVAL_3CLASS:-1}"
export KNN_REFERENCE_CROSS_FIT="${KNN_REFERENCE_CROSS_FIT:-1}"
export CROSS_FIT_FOLDS="${CROSS_FIT_FOLDS:-5}"
export COMMITTEE_MEMBERS="${COMMITTEE_MEMBERS:-8}"
export COMMITTEE_KEEP_FRACS="${COMMITTEE_KEEP_FRACS:-0.5}"
# EPOCHS>0 trains each arm for that many passes over its actual subset (no-filter -> all
# rows, f50 -> the kept half) instead of the fixed MAX_TRAIN_STEPS cap; 0 restores the cap.
export EPOCHS="${EPOCHS:-2}"

# The Table 1 arms at the 50% budget. elloss_* loads the weak and the untuned strong model
# once more for the excess-loss forward passes.
export FILTER_RUNS="${FILTER_RUNS:-mlpstep1_high_balanced_f50,rp_high_balanced_f50,elloss_high_balanced_f50,ccarobust_high_balanced_f50,random_balanced_f50,overlap_high_balanced_f50,confidence_high_balanced_f50,toklen_high_balanced_f50,knn_high_balanced_f50,cca_high_balanced_f50}"

echo "=== Sweep (${DATASET}) ==="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TRAIN_SEEDS=${TRAIN_SEEDS}"
echo "COMMITTEE_KEEP_FRACS=${COMMITTEE_KEEP_FRACS}"
echo "FILTER_RUNS=${FILTER_RUNS}"

exec bash scripts/run_seed_loop.sh
