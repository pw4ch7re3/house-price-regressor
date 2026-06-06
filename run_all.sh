#!/usr/bin/env bash
# Complete run: preprocess the data once (materializing the split, encoding and
# scaling for both location-encoding variants), then train every model on each
# variant. 2 variants x 4 models = 8 invocations.
set -euo pipefail

# Run from the repo root so relative paths (results/metrics.csv,
# models/best_models, data/processed) resolve correctly.
cd "$(dirname "$(readlink -f "$0")")"

# Remove stale metrics (record.py appends, so a fresh run needs a clean file).
rm -f results/metrics.csv

python data/preprocess.py

variants=(cat coord_only tgt_only tgt)
models=(mlp dt rf mgbdt)

# Early stopping for the MLP cuts wasted epochs once the held-out loss plateaus
# (other models ignore these flags).
for variant in "${variants[@]}"; do
    for model in "${models[@]}"; do
        echo "=== Training ${model} | variant=${variant} ==="
        if [[ "${model}" == "mlp" ]]; then
            python training/train.py --variant "${variant}" --model "${model}" --patience 5
        else
            python training/train.py --variant "${variant}" --model "${model}"
        fi
    done
done

echo "All runs complete. Metrics written to results/metrics.csv"
