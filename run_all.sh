#!/usr/bin/env bash
# Complete run: preprocess data, then train all model/drop-case combinations.
# Each train.py invocation internally loops over both targets (price and
# price_per_sqft), so 3 models x 4 drop cases = 12 invocations.
set -euo pipefail

# Run from the repo root so relative paths (results/metrics.csv,
# models/best_models, data/processed) resolve correctly.
cd "$(dirname "$(readlink -f "$0")")"

# 0. Remove stale metrics (record.py appends, so a fresh run needs a clean file).
rm -f results/metrics.csv

# 1. Preprocess the raw data.
python data/preprocess.py

# 2. Train every model across every drop case.
models=(dt mlp mgbdt)
declare -a drop_flags=(
    ""                          # drop nothing
    "--drop_address"            # drop address
    "--drop_coord"              # drop coords
    "--drop_address --drop_coord"  # drop both
)

for model in "${models[@]}"; do
    for flags in "${drop_flags[@]}"; do
        echo "=== Training ${model} ${flags:-(no drops)} ==="
        python training/train.py --model "${model}" ${flags}
    done
done

echo "All runs complete. Metrics written to results/metrics.csv"
