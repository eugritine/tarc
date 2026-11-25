#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
  echo "Usage: bash scripts/train_five_seeds.sh DATASET DATA_DIR ENCODER_DIR INITIALIZATION OUTPUT_ROOT [DEVICE]" >&2
  exit 2
fi
tarc_dataset="$1"
tarc_data="$2"
tarc_encoder="$3"
tarc_initialization="$4"
tarc_output="$5"
tarc_device="${6:-cuda}"
case "$tarc_dataset" in mosi|mosei|iemocap) ;; *) echo "Unknown dataset" >&2; exit 2;; esac
for tarc_seed in 1 2 3 4 5; do
  python -m tarc.train --dataset "$tarc_dataset" --data "$tarc_data" \
    --encoder "$tarc_encoder" --initialization "$tarc_initialization" \
    --output "$tarc_output/seed_$tarc_seed" --seed "$tarc_seed" --device "$tarc_device"
done
