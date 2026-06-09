import torch
import os
import sys
import random
import argparse
import pandas as pd
import numpy as np

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if path not in sys.path:
    sys.path.insert(0, path)

from data.dataload import (
    load_df,
    split_X_y,
    kfold_splits,
    apply_variant,
    scale_features,
    scale_target,
    invert_target,
    PRICE_PATH,
    TARGET,
)
from models import TrainConfig

from metrics.mse import rmse
from training.train import build_model, make_model_config
from training.record import (
    compute_metrics,
    mean_metrics,
    std_metrics,
    save_split_metrics,
    METRIC_LABELS,
)


def fit_target(model_config, X_train, y_train_scaled, train_kwargs):
    """Fit one base model on a scaled target (no persistence during CV)."""
    return build_model(
        model_config, TrainConfig(X=X_train, y=y_train_scaled, **train_kwargs)
    )


def optimize_weight(y_true, price_pred, price_from_pps_pred, steps: int = 101):
    """Grid-search w in [0, 1] minimizing RMSE of w*price + (1-w)*price_from_pps."""
    best_w, best_rmse = 0.0, float("inf")
    for w in np.linspace(0.0, 1.0, steps):
        blended = w * price_pred + (1.0 - w) * price_from_pps_pred
        score = rmse(y_true, blended)
        if score < best_rmse:
            best_w, best_rmse = float(w), score
    return best_w, best_rmse


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)

    # Reproducible k-fold CV over the raw engineered frame. Split + per-fold
    # location encoding (apply_variant) happen in memory so target encoding is
    # re-fit per fold (no leakage). The blend weight is optimized on each fold's
    # own train split; metrics are the across-fold mean.
    df = load_df(PRICE_PATH)
    X, y = split_X_y(df, TARGET)

    epochs = args.epochs_mgbdt if args.model == "mgbdt" else args.epochs_mlp
    train_kwargs = dict(
        epochs=epochs,
        lr=args.lr_mlp,
        batch_size=args.batch_size,
        verbose=args.verbose,
        patience=args.patience,
        val_split=args.val_split,
    )

    model_name = f"ensemble_{args.model}"
    fold_train_metrics = []
    fold_test_metrics = []
    weights = []

    for fold, ((X_train, y_train), (X_test, y_test)) in enumerate(
        kfold_splits(X, y, args.k, args.seed)
    ):
        X_train, X_test = apply_variant(X_train, X_test, y_train, args.variant)

        # Raw living area + price, captured before feature scaling z-scores them —
        # used to form the pps target and convert pps predictions back to price.
        sqft_train = X_train["sqft_living"].to_numpy()
        sqft_test = X_test["sqft_living"].to_numpy()
        price_train = y_train.to_numpy(dtype=float)
        price_test = y_test.to_numpy(dtype=float)

        # Model 2 target: price per square foot of living area (raw $/sqft).
        pps_train = price_train / sqft_train
        pps_test = price_test / sqft_test

        # Feature scaling (fit on train) shared by both base models.
        X_train, X_test = scale_features(X_train.copy(), X_test.copy())

        # Each target is MinMax-scaled for training, then predictions are inverted
        # back to raw units for blending/metrics.
        yp_tr_s, _, price_scaler = scale_target(price_train, price_test)
        pps_tr_s, _, pps_scaler = scale_target(pps_train, pps_test)

        model_config_price = make_model_config(args.model, X_train.shape[1], args)
        model_config_pps = make_model_config(args.model, X_train.shape[1], args)
        model1 = fit_target(model_config_price, X_train, yp_tr_s, train_kwargs)
        model2 = fit_target(model_config_pps, X_train, pps_tr_s, train_kwargs)

        # Predictions, inverted to raw units.
        price1_tr = invert_target(model1.predict(X_train), price_scaler)
        price1_te = invert_target(model1.predict(X_test), price_scaler)
        pps2_tr = invert_target(model2.predict(X_train), pps_scaler)
        pps2_te = invert_target(model2.predict(X_test), pps_scaler)
        price2_tr = pps2_tr * sqft_train
        price2_te = pps2_te * sqft_test

        # Blend weight optimized on this fold's train split (in dollar space).
        best_w, _ = optimize_weight(price_train, price1_tr, price2_tr)
        weights.append(best_w)
        ens_tr = best_w * price1_tr + (1.0 - best_w) * price2_tr
        ens_te = best_w * price1_te + (1.0 - best_w) * price2_te

        n_features = X_train.shape[1]
        fold_train_metrics.append(compute_metrics(price_train, ens_tr, n_features))
        fold_test_metrics.append(compute_metrics(price_test, ens_te, n_features))
        print(
            f"[{model_name}|{args.variant}] fold {fold + 1}/{args.k} w={best_w:.3f} "
            f"test Adjusted R²: {fold_test_metrics[-1]['Adjusted_R2']:.4f}"
        )

    train_mean = mean_metrics(fold_train_metrics)
    test_mean = mean_metrics(fold_test_metrics)
    train_std = std_metrics(fold_train_metrics)
    test_std = std_metrics(fold_test_metrics)

    print(
        f"\n=== {model_name} | price (variant={args.variant}) | {args.k}-fold CV mean ==="
    )
    print(f"mean blend weight w = {sum(weights) / len(weights):.4f}")
    for metric, value in train_mean.items():
        print(f"Train {METRIC_LABELS[metric] + ':':<14} {value:.4f}")
    for metric, value in test_mean.items():
        print(f"Test  {METRIC_LABELS[metric] + ':':<14} {value:.4f} ± {test_std[metric]:.4f}")

    # Record the across-fold mean (value) and std under split=train/test, one row
    # per metric, so results/visualize_results.ipynb can draw error bars.
    save_split_metrics("price", model_name, args.variant, "train", train_mean, train_std)
    save_split_metrics("price", model_name, args.variant, "test", test_mean, test_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a blended ensemble: a direct price model plus a "
        "price_per_sqft model (weight optimized per fold on its train split), "
        "evaluated with k-fold cross-validation (reported as the across-fold mean)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="global random seed for reproducibility"
    )
    parser.add_argument(
        "--k", type=int, default=5, help="number of cross-validation folds"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="rf",
        choices=["mlp", "dt", "rf", "mgbdt"],
        help="base model used for both ensemble members",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="tgt",
        choices=["cat", "tgt", "coord_only", "tgt_only"],
        help="location-encoding variant (same as train.py)",
    )

    parser.add_argument(
        "--epochs_mlp", type=int, default=50, help="training epochs for mlp"
    )
    parser.add_argument(
        "--epochs_mgbdt", type=int, default=20, help="training epochs for mgbdt"
    )
    parser.add_argument(
        "--lr_mlp", type=float, default=1e-2, help="learning rate for mlp (Adam)"
    )
    parser.add_argument(
        "--lr_mgbdt", type=float, default=0.1, help="learning rate for mgbdt (xgb)"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="early-stopping patience for mlp (0 disables early stopping)",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="fraction of training data held out for early-stopping validation",
    )

    parser.add_argument(
        "--max_depth_dt", type=int, default=6, help="max depth for decision tree"
    )
    parser.add_argument(
        "--max_depth_rf", type=int, default=None, help="max depth for random forest"
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100,
        help="number of trees for random forest",
    )
    parser.add_argument(
        "--max_depth_mgbdt", type=int, default=3, help="max depth for mgbdt (xgb)"
    )
    parser.add_argument(
        "--min_samples_split",
        type=int,
        default=2,
        help="min samples to split for decision tree",
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=2,
        help="min samples per leaf for decision tree",
    )

    parser.add_argument(
        "--num_boost_round",
        type=int,
        default=5,
        help="num boost rounds per layer for mgbdt",
    )
    parser.add_argument(
        "--target_lr",
        type=float,
        default=0.5,
        help="target-propagation step size for mgbdt",
    )

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
