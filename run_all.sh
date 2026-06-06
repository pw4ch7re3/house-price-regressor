#!/usr/bin/env bash
# Complete run: preprocess the data once (materializing the split, encoding and
# scaling for both location-encoding variants), then train every model on each
# variant. 2 variants x 4 models = 8 invocations.
set -euo pipefail

# Run from the repo root so relative paths (results/metrics.csv,
# models/best_models, data/processed) resolve correctly.
cd "$(dirname "$(readlink -f "$0")")"

# 0. Remove stale metrics (record.py appends, so a fresh run needs a clean file).
rm -f results/metrics.csv

# 1. Preprocess the raw data into cat/tgt train/test variant files + scalers.
python data/preprocess.py

# 2. Train every model on each variant. The 'cat' variant (ordinal codes, no
#    coords) is the categorical-encoding reference; 'tgt' uses target-encoded
#    city/zipcode plus cartesian coordinates.
variants=(cat tgt)
models=(mlp dt rf mgbdt)

for variant in "${variants[@]}"; do
    for model in "${models[@]}"; do
        echo "=== Training ${model} | variant=${variant} ==="
        python training/train.py --variant "${variant}" --model "${model}"
    done
done

echo "All runs complete. Metrics written to results/metrics.csv"
