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
    load_variant,
    scale_features,
    scale_target,
    invert_target,
)
from models import TrainConfig
from models.mlp import MLPConfig
from models.decision_tree import DecisionTreeConfig
from models.random_forest import RandomForestConfig
from models.mgbdt_ours import mGBDTConfig

from metrics.mse import rmse
from training.train import build_model
from training.record import print_metrics

output_path = "models/best_models"

# Targets reported by the ensemble: the direct price model, the price_per_sqft
# model, and the blended ensemble.
TARGETS = ["price", "price_per_sqft", "price_ensemble"]


def make_model_config(model: str, input_dim: int, args):
    """Build the ModelConfig for ``model`` from CLI args (mirrors train.py)."""
    if model == "mlp":
        return MLPConfig(
            model="mlp",
            input_dim=input_dim,
            hidden_dims=[32, 32],
            output_dim=1,
        )
    if model == "dt":
        return DecisionTreeConfig(
            model="dt",
            max_depth=args.max_depth_dt,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
        )
    if model == "rf":
        return RandomForestConfig(
            model="rf",
            n_estimators=args.n_estimators,
            max_depth=args.max_depth_rf,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
        )
    if model == "mgbdt":
        return mGBDTConfig(
            model="mgbdt",
            input_size=input_dim,
            output_size=1,
            task="regression",
            learning_rate=args.lr_mgbdt,
            max_depth=args.max_depth_mgbdt,
            num_boost_round=args.num_boost_round,
            target_lr=args.target_lr,
        )
    raise ValueError(f"Unknown model: {model}")


def fit_target(model_config, X_train, y_train_scaled, train_kwargs, save_name):
    """Fit one base model on a scaled target and persist its state dict."""
    model = build_model(
        model_config, TrainConfig(X=X_train, y=y_train_scaled, **train_kwargs)
    )
    os.makedirs(output_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_path, f"{save_name}.pth"))
    return model


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

    # Raw variant data: features and price are unscaled, so price_per_sqft can be
    # formed from raw price / raw sqft_living before any scaling is applied.
    (X_train, y_train), (X_test, y_test) = load_variant(args.variant)

    # Raw living area (per row) — used to build the pps target and to convert pps
    # predictions back to price. Captured before feature scaling z-scores it.
    sqft_train = X_train["sqft_living"].to_numpy()
    sqft_test = X_test["sqft_living"].to_numpy()

    price_train = y_train.to_numpy(dtype=float)
    price_test = y_test.to_numpy(dtype=float)

    # Model 2 target: price per square foot of living area (raw dollars / sqft).
    pps_train = price_train / sqft_train
    pps_test = price_test / sqft_test

    # Feature scaling (fit on train) shared by both base models.
    X_train, X_test = scale_features(X_train.copy(), X_test.copy())

    # Each target is MinMax-scaled for training (so mlp/mgbdt behave), then
    # predictions are inverted back to their raw units for blending/metrics.
    yp_tr_s, _, price_scaler = scale_target(price_train, price_test)
    pps_tr_s, _, pps_scaler = scale_target(pps_train, pps_test)

    epochs = args.epochs_mgbdt if args.model == "mgbdt" else args.epochs_mlp
    train_kwargs = dict(
        epochs=epochs,
        lr=args.lr_mlp,
        batch_size=args.batch_size,
        verbose=args.verbose,
        patience=args.patience,
        val_split=args.val_split,
    )

    model_config_price = make_model_config(args.model, X_train.shape[1], args)
    model_config_pps = make_model_config(args.model, X_train.shape[1], args)

    name = f"ensemble_{args.model}_{args.variant}"
    model1 = fit_target(
        model_config_price, X_train, yp_tr_s, train_kwargs, f"{name}_price"
    )
    model2 = fit_target(
        model_config_pps, X_train, pps_tr_s, train_kwargs, f"{name}_price_per_sqft"
    )

    # Predictions, inverted to raw units.
    price1_tr = invert_target(model1.predict(X_train), price_scaler)
    price1_te = invert_target(model1.predict(X_test), price_scaler)

    pps2_tr = invert_target(model2.predict(X_train), pps_scaler)
    pps2_te = invert_target(model2.predict(X_test), pps_scaler)
    price2_tr = pps2_tr * sqft_train
    price2_te = pps2_te * sqft_test

    # Blend weight optimized on the train split (in dollar space), held-out test
    # evaluated once (no cross-validation).
    best_w, _ = optimize_weight(price_train, price1_tr, price2_tr)
    ens_tr = best_w * price1_tr + (1.0 - best_w) * price2_tr
    ens_te = best_w * price1_te + (1.0 - best_w) * price2_te
    print(
        f"\n=== ensemble weight (optimized on train) ===\n"
        f"w={best_w:.4f}  -> price = {best_w:.4f}*price_1 + "
        f"{1 - best_w:.4f}*(price_per_sqft_2 * sqft_living)"
    )

    n_features = X_train.shape[1]
    preds = {
        "price": (price_train, price1_tr, price_test, price1_te),
        "price_per_sqft": (pps_train, pps2_tr, pps_test, pps2_te),
        "price_ensemble": (price_train, ens_tr, price_test, ens_te),
    }

    model_name = f"ensemble_{args.model}"
    for target in TARGETS:
        if target == "price_ensemble":
            y_tr_true, y_tr_pred, y_te_true, y_te_pred = preds[target]
            print_metrics(
                "price",
                model_name,
                args.variant,
                y_tr_true,
                y_tr_pred,
                y_te_true,
                y_te_pred,
                n_features,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a blended ensemble: a direct price model plus a "
        "price_per_sqft model (weight optimized on the train split), evaluated "
        "on the held-out test set"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="global random seed for reproducibility"
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
