# House Price Regressor

Predicting residential sale prices on the [USA House Prices](https://www.kaggle.com/datasets/fratzcan/usa-house-prices/data)
dataset (Washington State). The project compares four regression models across
four location-encoding variants, each both as a single model and as a
price / price-per-sqft **stacking ensemble**, all under shared 5-fold
cross-validation.

The study asks a single question: _can richer feature encoding or model structure
beat a naive Random Forest on ordinal-encoded categories?_ The answer is yes —
domain-aware preprocessing (coordinate encoding + price-per-sqft stacking) matters
more than model depth on this tabular task.

## Key results

All numbers are 5-fold CV means of **Adjusted R²** on the validation split
(`price` target, USD). Full results are in [`results/metrics.csv`](results/metrics.csv)
and the writeup in [`.reports/`](.reports/).

| Configuration                    | Mode             | RMSE (k\$) | Adjusted R² |
| :------------------------------- | :--------------- | ---------: | ----------: |
| `rf` / `cat`                     | baseline         |      197.1 |       0.667 |
| `mGBDT(XGBoost)` / `tgt`         | best single      |      164.9 |       0.765 |
| **ensemble `rf` / `coord_only`** | **best overall** |  **154.1** |   **0.794** |

The best configuration cuts RMSE by **22%** and lifts Adjusted R² from 0.667 to
**0.794** over the baseline. Note that the reported "mGBDT" uses a single
target-propagation layer, which reduces to plain **XGBoost**; a two-layer variant
was tried but degraded accuracy.

## Models and variants

**Models** (`--model`): `mlp`, `dt` (CART), `rf` (Random Forest), `mgbdt`
(single-layer mGBDT ≡ XGBoost).

**Location-encoding variants** (`--variant`):

| Variant      | Encoding                                                    |
| :----------- | :---------------------------------------------------------- |
| `cat`        | ordinal city/zipcode codes, no coordinates (naive baseline) |
| `tgt`        | target-encoded city/zipcode + cartesian `x,y,z`             |
| `coord_only` | cartesian `x,y,z` only                                      |
| `tgt_only`   | target-encoded address only                                 |

## Repository structure

```
data/          Preprocessing + per-fold data loading
  preprocess.py    Clean raw CSV -> data/processed/
  dataload.py      K-fold splits, scaling, target/coordinate encoding (leakage-free)
  raw/, processed/ Input and cleaned datasets
models/        Model implementations
  mlp.py, decision_tree.py, random_forest.py
  mgbdt_ours.py    mGBDT wrapper over the vendored mgbdt/ library
  best_models/     Saved fold checkpoints (.pth)
metrics/       RMSE, MAE, MAPE, R^2, Adjusted R^2 (+ timer)
training/      train.py (single), train_ensemble.py (stacking), record.py (CSV logging)
results/       metrics.csv, figures, visualize_results.ipynb
run_all.sh     Full pipeline: preprocess + 32 runs (4 variants x 4 models x 2 modes)
.reports/      LaTeX report (main.tex + sections) and compiled PDF
```

## Installation

Python 3.12 is recommended.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Reproduce everything

```bash
./run_all.sh
```

This removes stale metrics, runs preprocessing once, then trains all 32
configurations and writes across-fold mean ± std metrics to
`results/metrics.csv`.

### Run individual steps

```bash
# 1. Preprocess raw data -> data/processed/
python data/preprocess.py

# 2. Train a single model (5-fold CV)
python training/train.py --variant coord_only --model rf --k 5

# 3. Train the price / price-per-sqft stacking ensemble
python training/train_ensemble.py --variant coord_only --model rf --k 5
```

Useful flags (defaults match the reported configs): `--seed 42`, `--k 5`,
`--model {mlp,dt,rf,mgbdt}`, `--variant {cat,tgt,coord_only,tgt_only}`,
`--patience 5` (MLP early stopping). See `python training/train.py --help` for the
full list of hyperparameters.

### Visualize results

Open [`results/visualize_results.ipynb`](results/visualize_results.ipynb) to
regenerate the ranking figures from `metrics.csv`.

## Method summary

- **Preprocessing:** drop non-positive prices and the top 0.1% price outliers
  (4,140 -> 4,086 records); `lat/long` -> cartesian `x,y,z` (coordinate encoding);
  `log_sqft_lot`, `age`, `was_renovated` engineering; MinMax/Z-score scaling fit
  per fold.
- **Target encoding:** smoothed (factor 10) category means, fit on training rows
  only — leakage-free under cross-validation.
- **Stacking ensemble:** blend a direct `price` model with a `price_per_sqft`
  model (converted back via `× sqft_living`); blend weight grid-searched on each
  fold's training split.
- **Evaluation:** 5-fold CV (seed 42); metrics reported as across-fold mean ± std;
  ranked primarily by validation Adjusted R².

## Team

|     Name      |             GitHub             |
| :-----------: | :----------------------------: |
|  Seungho Han  |   https://github.com/hsh5405   |
|  Sungho Kim   |  https://github.com/pw4ch7re3  |
| Moonseong Son | https://github.com/hmoon-seong |
|  Jaeyun Lim   |  https://github.com/JaeyunLim  |

## Dataset

[USA House Prices (Kaggle)](https://www.kaggle.com/datasets/fratzcan/usa-house-prices/data)

## License

See [LICENSE](LICENSE).
