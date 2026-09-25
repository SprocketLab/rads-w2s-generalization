#!/usr/bin/env bash
# Runs the sweeps behind one result of the paper, with the settings of Appendix B.1 over three
# training seeds, writes them to results/<result>/<dataset>, and saves the PGR of each in pgr.txt.
#
# usage: bash scripts/reproduce.sh <result> [dataset]
#
#   table1 <dataset>     Table 1: the ten selection methods at the 50% budget
#   table6 <dataset>     Table 6: the low bands of the same methods
#   random <dataset>     Random Selection: five random halves, then their mean PGR
#   table7 <dataset>     Table 7 and the weight-decay columns of Table 11: linstep1 and mlpstep2
#   table9 <dataset>     Table 9: Qwen2.5-1.5B as the weak model (anli-r1, wanli, fevernli)
#   table10 <dataset>    Tables 10 and 11: RADS and mlpstep2 without weight decay (anli-r1, wanli, fevernli)
#   figure1b             Figure 1b: ground-truth labels on the random half of ANLI-R1
#   figure2 <dataset>    Figures 2 and 8: seven data budgets (anli-r1, sst2)
#   figure3 <dataset>    Figure 3 and the CKA and RADS columns of Table 8
#   figure4              Figure 4: noised labels on ANLI-R1
#
# datasets: wanli fevernli hellaswag sciq sst2 imdb agnews ethicsjustice anli-r1 tweetsent wikitoxic
#           qnli scitail boolq cosmosqa csqa imppres scinli quail aquarat riddlesense blimp anli-r3
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

usage() {
  sed -n '/^# usage:/,/^set /{/^set /d;s/^# \{0,1\}//;p;}' scripts/reproduce.sh >&2
  exit 1
}

# Exports DATASET and the settings in which the dataset departs from the sweep defaults.
use_dataset() {
  case "$1" in
    wanli|fevernli|blimp) export DATASET="$1" ;;
    hellaswag|sciq|sst2) export DATASET="$1" N_TEST=2000 ;;
    imdb) export DATASET=imdb N_TEST=2000 MAX_LENGTH=512 ;;
    agnews) export DATASET=agnews N_TEST=5000 ;;
    ethicsjustice|tweetsent|wikitoxic|qnli|scitail|boolq|cosmosqa|csqa|imppres|scinli)
      export DATASET="$1" N_VAL=800 N_TEST=800 ;;
    quail) export DATASET=quail N_VAL=800 N_TEST=800 MAX_LENGTH=1024 ;;
    aquarat) export DATASET=aquarat N_VAL=500 N_TEST=500 ;;
    riddlesense) export DATASET=riddlesense N_TRAIN=3000 N_VAL=400 N_TEST=800 ;;
    anli-r1|anli-r3) export DATASET=anli ANLI_ROUND="${1#anli-}" ;;
    *) usage ;;
  esac
}

# sweep <output root> [VAR=value ...]: one sweep of the current dataset; the settings apply to it alone.
# It then saves the PGR of every method in pgr.txt; a random half without its own ground-truth model
# is scored with the others at the end of "random".
sweep() {
  local driver=scripts/run_sweep.sh
  if [[ "$DATASET" == anli ]]; then driver=scripts/run_sweep_anli.sh; fi
  env OUTPUT_ROOT="$1" "${@:2}" bash "$driver"
  if [[ " ${*:2} " != *" BASELINE_RUNS=base "* ]]; then
    python scripts/compute_pgr.py "$1" | tee "$1/pgr.txt"
  fi
}

result="${1:-}"
data="${2:-}"
case "$result" in
  figure1b|figure4) [[ $# -eq 1 ]] || usage; data=anli-r1 ;;
  table9|table10) [[ $# -eq 2 && "$data" =~ ^(anli-r1|wanli|fevernli)$ ]] || usage ;;
  figure2) [[ $# -eq 2 && "$data" =~ ^(anli-r1|sst2)$ ]] || usage ;;
  table1|table6|random|table7|figure3) [[ $# -eq 2 ]] || usage ;;
  *) usage ;;
esac

# Clear the settings this script varies or derives, so none left in the calling shell leaks in.
unset DATASET OUTPUT_ROOT BASELINE_OUTPUT_DIR FILTER_OUTPUT_ROOT BASELINE_RUNS FILTER_RUNS \
  COMMITTEE_KEEP_FRACS N_TRAIN N_VAL N_TEST N_EVAL_QUESTIONS MAX_LENGTH ANLI_ROUND DUMP_ACTS \
  CUSTOM_SCORES_NPZ MLPSTEP1_WD WEAK_MODEL PURITY_MODE PURITY_GRID PURITY_DRAWS RANDOM_CONTROL_SIZE
# Batches of two fit the excess-loss pass of the longest prompts on a 24 GB GPU.
export EL_BATCH=2
use_dataset "$data"
root="results/$result/$data"

# The python steps below run after hours of training, so a missing environment stops the run here.
python -c "import torch" || { echo "activate the environment of the README's Installation first" >&2; exit 1; }

TABLE1=mlpstep1_high_balanced_f50,rp_high_balanced_f50,elloss_high_balanced_f50,ccarobust_high_balanced_f50,random_balanced_f50,overlap_high_balanced_f50,confidence_high_balanced_f50,toklen_high_balanced_f50,knn_high_balanced_f50,cca_high_balanced_f50

case "$result" in
  table1) sweep "$root" FILTER_RUNS="$TABLE1" ;;
  table6)
    # Random Selection has no low band; its column comes from the five halves of "random".
    low="${TABLE1//_high_/_low_}"
    sweep "$root" FILTER_RUNS="${low/random_balanced_f50,/}" ;;
  random)
    # A half is drawn by the position of 0.5 in COMMITTEE_KEEP_FRACS: positions 0, 1, 2, 4 and 5.
    # Only the first half trains the ground-truth model; compute_pgr --anchors lends it to the rest.
    i=0
    for fracs in 0.5 0.1,0.5 0.1,0.2,0.5 0.1,0.2,0.4,0.6,0.5 0.1,0.2,0.4,0.6,0.8,0.5; do
      i=$((i + 1))
      baseline=base,ground_truth
      if [[ $i -gt 1 ]]; then baseline=base; fi
      sweep "$root/half$i" BASELINE_RUNS="$baseline" FILTER_RUNS=random_balanced_f50 COMMITTEE_KEEP_FRACS="$fracs"
    done
    python scripts/compute_pgr.py --anchors "$root/half1" "$root"/half{1..5} | tee "$root/pgr.txt" ;;
  table7)
    # mlpstep2 is scored offline from the representations the linstep1 sweep dumps.
    sweep "$root/linstep1" FILTER_RUNS=linstep1_high_balanced_f50,linstep1_low_balanced_f50 DUMP_ACTS="$root/acts.npz"
    python scripts/score_mlpstep2.py --acts "$root/acts.npz" --out "$root/mlpstep2_scores.npz"
    sweep "$root/mlpstep2" FILTER_RUNS=cs1_high_balanced_f50,cs1_low_balanced_f50 CUSTOM_SCORES_NPZ="$root/mlpstep2_scores.npz" ;;
  table9)
    sweep "$root" WEAK_MODEL=Qwen/Qwen2.5-1.5B \
      FILTER_RUNS=mlpstep1_high_balanced_f50,mlpstep1_low_balanced_f50,rp_high_balanced_f50,random_balanced_f50 ;;
  table10)
    sweep "$root/rads" MLPSTEP1_WD=0 FILTER_RUNS=mlpstep1_high_balanced_f50,mlpstep1_low_balanced_f50 DUMP_ACTS="$root/acts.npz"
    python scripts/score_mlpstep2.py --acts "$root/acts.npz" --out "$root/mlpstep2_scores.npz" --wd 0
    sweep "$root/mlpstep2" FILTER_RUNS=cs1_high_balanced_f50,cs1_low_balanced_f50 CUSTOM_SCORES_NPZ="$root/mlpstep2_scores.npz" ;;
  figure1b) sweep "$root" FILTER_RUNS=random_gt_balanced_f50 ;;
  figure2)
    methods="mlpstep1_high random confidence_low confidence_high elloss_high"
    if [[ "$data" == anli-r1 ]]; then methods="$methods weak_correct_only"; fi
    runs=""
    for m in $methods; do
      for pct in 10 20 40 50 60 80 100; do runs="${runs:+$runs,}${m}_balanced_f$pct"; done
    done
    sweep "$root" COMMITTEE_KEEP_FRACS=0.1,0.2,0.4,0.5,0.6,0.8,1.0 FILTER_RUNS="$runs"
    # Figure 8 has the correct-only oracle at the 50% budget alone.
    if [[ "$data" == sst2 ]]; then
      sweep "$root/oracle" COMMITTEE_KEEP_FRACS=0.5 FILTER_RUNS=weak_correct_only_balanced_f50
    fi ;;
  figure3)
    sweep "$root" DUMP_ACTS="$root/acts.npz" FILTER_RUNS=mlpstep1_high_balanced_f50,mlpstep1_low_balanced_f50
    python scripts/compute_cka.py "$root/acts.npz" | tee "$root/cka.txt" ;;
  figure4)
    sweep "$root/relabel" FILTER_RUNS=purity_grid PURITY_MODE=relabel PURITY_GRID=50,60,80,90,100 RANDOM_CONTROL_SIZE=1500
    sweep "$root/iid" FILTER_RUNS=purity_grid,purity_natural PURITY_MODE=iid PURITY_GRID=50,60,nat,80,90 PURITY_DRAWS=3 RANDOM_CONTROL_SIZE=1500 ;;
esac
