"""Hyperparameter search for the mGBDT model (single- and two-layer).

Reuses the exact preprocessing pipeline from training/train.py, then grid-searches
mGBDT hyperparameters -- including `layer_configs` for the 2-layer architecture --
and reports the best config (by test RMSE) for each target.

Usage:
    python experiments/mgbdt_hyperparameter_tuning.py            # both targets, full grid
    python experiments/mgbdt_hyperparameter_tuning.py --quick    # small grid (smoke test)
    python experiments/mgbdt_hyperparameter_tuning.py --target price
"""

import os
import sys
import argparse
import itertools
import random
import time

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from data.dataload import (
    load_df,
    target_encode,
    split_X_y,
    split_train_test,
    PRICE_PATH,
    PRICE_PER_SQFT_PATH,
)
from metrics.mse import rmse
from metrics.r2_score import r2_score
from metrics.mae import mae
from metrics.mape import mape

from models.mgbdt_ours import mGBDTConfig, MGBDTModel

MINMAX_COLS = ["x", "y", "z", "condition", "age", "bedrooms", "bathrooms", "floors", "view"]
ZSCORE_COLS = ["sqft_living", "sqft_above", "sqft_basement", "log_sqft_lot", "city", "zipcode"]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def preprocess(data_path: str):
    """Replicates training/train.py preprocessing for one target file."""
    df = load_df(data_path)
    X, y = split_X_y(df, "price")
    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)

    # raw (original-scale) targets for reporting
    y_train_raw = y_train.copy()
    y_test_raw = y_test.copy()

    target_scaler = None
    if data_path == PRICE_PATH:
        target_scaler = MinMaxScaler()
        y_train = pd.Series(
            target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
            index=y_train.index,
        )

    for col in ["city", "zipcode"]:
        X_train[col], X_test[col] = target_encode(X_train, y_train, X_test, col)

    scaler_mm = MinMaxScaler()
    X_train[MINMAX_COLS] = scaler_mm.fit_transform(X_train[MINMAX_COLS])
    X_test[MINMAX_COLS] = scaler_mm.transform(X_test[MINMAX_COLS])

    scaler = StandardScaler()
    X_train[ZSCORE_COLS] = scaler.fit_transform(X_train[ZSCORE_COLS])
    X_test[ZSCORE_COLS] = scaler.transform(X_test[ZSCORE_COLS])

    return X_train, X_test, y_train, y_train_raw, y_test_raw, target_scaler


def build_layer_configs(input_size: int):
    """Return a list of (name, layer_configs) candidates: 1-layer and 2-layer."""
    configs = [("1L", [("tp_layer", "xgb")])]
    # 2-layer: an intermediate hidden representation of size `h`, then map to output.
    for h in (input_size, input_size * 2, max(2, input_size // 2)):
        configs.append((f"2L-h{h}", [("tp_layer", "xgb", h), ("tp_layer", "xgb")]))
    return configs


def evaluate(X_train, X_test, y_train, y_train_raw, y_test_raw, target_scaler,
             layer_configs, learning_rate, max_depth, num_boost_round,
             target_lr, epochs, seed=42):
    set_seed(seed)
    config = mGBDTConfig(
        model="mgbdt",
        input_size=X_train.shape[1],
        output_size=1,
        task="regression",
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_boost_round=num_boost_round,
        target_lr=target_lr,
    )
    model = MGBDTModel(config, layer_configs=layer_configs, verbose=False)
    model.init(X_train, n_rounds=1)
    model.fit_arrays(X_train, y_train, n_epochs=epochs)

    y_test_pred = np.asarray(model.predict(X_test)).reshape(-1, 1)
    y_train_pred = np.asarray(model.predict(X_train)).reshape(-1, 1)
    if target_scaler is not None:
        y_test_pred = target_scaler.inverse_transform(y_test_pred)
        y_train_pred = target_scaler.inverse_transform(y_train_pred)
    y_test_pred = y_test_pred.ravel()
    y_train_pred = y_train_pred.ravel()

    return {
        "test_rmse": rmse(y_test_raw, y_test_pred),
        "test_mae": mae(y_test_raw, y_test_pred),
        "test_mape": mape(y_test_raw, y_test_pred),
        "test_r2": r2_score(y_test_raw, y_test_pred),
        "train_rmse": rmse(y_train_raw, y_train_pred),
        "train_r2": r2_score(y_train_raw, y_train_pred),
    }


def search(target_name, data_path, grid, quick=False):
    print(f"\n{'='*70}\nTarget: {target_name}  ({data_path})\n{'='*70}")
    X_train, X_test, y_train, y_train_raw, y_test_raw, target_scaler = preprocess(data_path)
    input_size = X_train.shape[1]

    layer_candidates = build_layer_configs(input_size)
    if quick:
        layer_candidates = layer_candidates[:2]

    combos = list(itertools.product(
        layer_candidates,
        grid["learning_rate"],
        grid["max_depth"],
        grid["num_boost_round"],
        grid["target_lr"],
        grid["epochs"],
    ))
    print(f"input_size={input_size}, {len(combos)} configs to evaluate\n")

    results = []
    for i, (lc, lr, md, nbr, tlr, ep) in enumerate(combos, 1):
        lc_name, lc_cfg = lc
        t0 = time.time()
        try:
            m = evaluate(X_train, X_test, y_train, y_train_raw, y_test_raw,
                         target_scaler, lc_cfg, lr, md, nbr, tlr, ep)
        except Exception as e:
            print(f"[{i}/{len(combos)}] {lc_name} lr={lr} md={md} nbr={nbr} "
                  f"tlr={tlr} ep={ep} -> FAILED: {e}")
            continue
        dt = time.time() - t0
        row = dict(layers=lc_name, layer_configs=lc_cfg, learning_rate=lr,
                   max_depth=md, num_boost_round=nbr, target_lr=tlr, epochs=ep,
                   secs=dt, **m)
        results.append(row)
        print(f"[{i}/{len(combos)}] {lc_name:8s} lr={lr:<5} md={md} nbr={nbr:<3} "
              f"tlr={tlr} ep={ep:<3} | test_rmse={m['test_rmse']:>12.2f} "
              f"test_r2={m['test_r2']:.4f} train_r2={m['train_r2']:.4f} ({dt:.1f}s)")

    if not results:
        print("No successful runs.")
        return None

    results.sort(key=lambda r: r["test_rmse"])
    best = results[0]
    print(f"\n--- TOP 5 for {target_name} (by test RMSE) ---")
    for r in results[:5]:
        print(f"  {r['layers']:8s} lr={r['learning_rate']:<5} md={r['max_depth']} "
              f"nbr={r['num_boost_round']:<3} tlr={r['target_lr']} ep={r['epochs']:<3} "
              f"| test_rmse={r['test_rmse']:.2f} test_r2={r['test_r2']:.4f}")
    print(f"\n>>> BEST {target_name}: {best['layers']}  "
          f"layer_configs={best['layer_configs']}")
    print(f"    learning_rate={best['learning_rate']}, max_depth={best['max_depth']}, "
          f"num_boost_round={best['num_boost_round']}, target_lr={best['target_lr']}, "
          f"epochs={best['epochs']}")
    print(f"    test_rmse={best['test_rmse']:.2f}  test_mae={best['test_mae']:.2f}  "
          f"test_mape={best['test_mape']:.4f}  test_r2={best['test_r2']:.4f}")
    return best


def main():
    parser = argparse.ArgumentParser(description="mGBDT hyperparameter search")
    parser.add_argument("--target", choices=["price", "price_per_sqft", "both"],
                        default="both")
    parser.add_argument("--quick", action="store_true", help="small smoke-test grid")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    if args.quick:
        grid = dict(learning_rate=[0.1], max_depth=[5], num_boost_round=[5],
                    target_lr=[0.5], epochs=[5])
    else:
        grid = dict(
            learning_rate=[0.05, 0.1, 0.3],
            max_depth=[3, 5, 7],
            num_boost_round=[5, 10],
            target_lr=[0.5],
            epochs=[10, 20],
        )

    targets = []
    if args.target in ("price", "both"):
        targets.append(("price", PRICE_PATH))
    if args.target in ("price_per_sqft", "both"):
        targets.append(("price_per_sqft", PRICE_PER_SQFT_PATH))

    summary = {}
    for name, path in targets:
        summary[name] = search(name, path, grid, quick=args.quick)

    print(f"\n{'#'*70}\nFINAL BEST CONFIGS\n{'#'*70}")
    for name, best in summary.items():
        if best is None:
            continue
        print(f"\n{name}:")
        print(f"  layer_configs   = {best['layer_configs']}")
        print(f"  learning_rate   = {best['learning_rate']}")
        print(f"  max_depth       = {best['max_depth']}")
        print(f"  num_boost_round = {best['num_boost_round']}")
        print(f"  target_lr       = {best['target_lr']}")
        print(f"  epochs          = {best['epochs']}")
        print(f"  -> test RMSE={best['test_rmse']:.2f}, R2={best['test_r2']:.4f}")


if __name__ == "__main__":
    main()
