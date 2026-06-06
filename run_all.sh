#!/usr/bin/env bash
# Complete run: preprocess the data once (materializing the raw split + encoding
# for each location variant), then train every model on each variant, both as a
# single model (training/train.py) and as the price/price_per_sqft ensemble
# (training/train_ensemble.py). 4 variants x 4 models x 2 = 32 invocations.
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

# Ensemble: same variant x model grid, blending a direct price model with a
# price_per_sqft model (weight optimized on train, evaluated on the test split).
for variant in "${variants[@]}"; do
    for model in "${models[@]}"; do
        echo "=== Ensemble ${model} | variant=${variant} ==="
        if [[ "${model}" == "mlp" ]]; then
            python training/train_ensemble.py --variant "${variant}" --model "${model}" --patience 5
        else
            python training/train_ensemble.py --variant "${variant}" --model "${model}"
        fi
    done
done

echo "All runs complete. Metrics written to results/metrics.csv"
