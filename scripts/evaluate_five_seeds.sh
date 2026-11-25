#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: bash scripts/evaluate_five_seeds.sh DATASET DATA_DIR RUN_ROOT [DEVICE]" >&2
  exit 2
fi
tarc_dataset="$1"
tarc_data="$2"
tarc_runs="$3"
tarc_device="${4:-cpu}"
case "$tarc_dataset" in mosi|mosei|iemocap) ;; *) echo "Unknown dataset" >&2; exit 2;; esac
for tarc_seed in 1 2 3 4 5; do
  test -f "$tarc_runs/seed_$tarc_seed/best.pt"
  test ! -e "$tarc_runs/seed_$tarc_seed/test_predictions.json"
  test ! -e "$tarc_runs/seed_$tarc_seed/test_metrics.json"
done
test -f "$tarc_data/test.npz"
test ! -e "$tarc_runs/test_summary.json"
for tarc_seed in 1 2 3 4 5; do
  python -m tarc.predict --dataset "$tarc_dataset" \
    --checkpoint "$tarc_runs/seed_$tarc_seed/best.pt" --features "$tarc_data/test.npz" \
    --device "$tarc_device" --output "$tarc_runs/seed_$tarc_seed/test_predictions.json"
  python -m tarc.evaluate --dataset "$tarc_dataset" --seed "$tarc_seed" --split test \
    --predictions "$tarc_runs/seed_$tarc_seed/test_predictions.json" \
    --output "$tarc_runs/seed_$tarc_seed/test_metrics.json"
done
python -m tarc.aggregate --inputs \
  "$tarc_runs/seed_1/test_metrics.json" "$tarc_runs/seed_2/test_metrics.json" \
  "$tarc_runs/seed_3/test_metrics.json" "$tarc_runs/seed_4/test_metrics.json" \
  "$tarc_runs/seed_5/test_metrics.json" --output "$tarc_runs/test_summary.json"
