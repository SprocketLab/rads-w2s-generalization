#!/usr/bin/env bash
set -euo pipefail
#
# The ANLI sweep: run_sweep.sh with DATASET=anli and ANLI defaults. ANLI_ROUND picks the
# round (default r1), and the output root defaults to results/anli_<round>. The script sets
# the pool sizes, the question-level evaluation, two epochs over each subset, the 50%
# budget, the Table 1 arms, and the representations the selection scores read (last layer,
# last token, whole prompt). Each setting is an environment variable that overrides its
# default; everything else comes from run_sweep.sh and the scripts it calls. The script
# exits if DATASET is set to anything other than anli.
#
# Usage:
#   bash scripts/run_sweep_anli.sh                  # ANLI-R1, training seeds 42 123 456
#   ANLI_ROUND=r3 bash scripts/run_sweep_anli.sh    # ANLI-R3

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${DATASET:-}" && "${DATASET}" != "anli" ]]; then
  echo "ERROR: run_sweep_anli.sh is ANLI-pinned; use run_sweep.sh for DATASET=${DATASET}" >&2
  exit 1
fi
export DATASET=anli
export ANLI_ROUND="${ANLI_ROUND:-r1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-results/anli_${ANLI_ROUND}}"
export TRAIN_SEEDS="${TRAIN_SEEDS:-42 123 456}"

# pool sizes
export N_TRAIN="${N_TRAIN:-6000}"
export N_VAL="${N_VAL:-1000}"
export N_TEST="${N_TEST:-1000}"

# question-level evaluation, two epochs over each subset, the 50% budget
export EVAL_3CLASS="${EVAL_3CLASS:-1}"
export N_EVAL_QUESTIONS="${N_EVAL_QUESTIONS:-1000}"
export EPOCHS="${EPOCHS:-2}"
export COMMITTEE_KEEP_FRACS="${COMMITTEE_KEEP_FRACS:-0.5}"

# the representations the selection scores read; the weak probe keeps the defaults
export REPRESENTATION_LAYER="${REPRESENTATION_LAYER:-end}"
export REPRESENTATION_POOLING="${REPRESENTATION_POOLING:-last}"
# REPRESENTATION_SPAN is 'full' (the whole prompt), 'answer' or 'context'; see run_w2s_lora.py --help
export REPRESENTATION_SPAN="${REPRESENTATION_SPAN:-full}"

# The Table 1 arms at the 50% budget.
export FILTER_RUNS="${FILTER_RUNS:-mlpstep1_high_balanced_f50,rp_high_balanced_f50,elloss_high_balanced_f50,ccarobust_high_balanced_f50,random_balanced_f50,overlap_high_balanced_f50,confidence_high_balanced_f50,toklen_high_balanced_f50,knn_high_balanced_f50,cca_high_balanced_f50}"

echo "=== ANLI sweep (${ANLI_ROUND}) ==="
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "TRAIN_SEEDS=${TRAIN_SEEDS}  EPOCHS=${EPOCHS}  COMMITTEE_KEEP_FRACS=${COMMITTEE_KEEP_FRACS}"
echo "REPRESENTATION_LAYER=${REPRESENTATION_LAYER}  REPRESENTATION_POOLING=${REPRESENTATION_POOLING}  REPRESENTATION_SPAN=${REPRESENTATION_SPAN}"

exec bash scripts/run_sweep.sh
