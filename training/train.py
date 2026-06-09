import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import random
import argparse
import pandas as pd
import numpy as np
from typing import cast

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
from models import ModelConfig, TrainConfig
from models.mlp import MLPConfig, MLP
from models.decision_tree import DecisionTreeConfig, DecisionTree
from models.random_forest import RandomForestConfig, RandomForest

from models.mgbdt_ours import mGBDTConfig, MGBDTModel

from training.record import (
    compute_metrics,
    mean_metrics,
    std_metrics,
    save_split_metrics,
    METRIC_LABELS,
)

def build_model(model_config: ModelConfig, train_config: TrainConfig):
    """Instantiate the model for ``model_config.model`` and fit it. Shared by
    train.py and training/train_ensemble.py so both dispatch identically."""
    model_name = model_config.model.lower()
    if model_name == "mlp":
        model = MLP(cast(MLPConfig, model_config))
        model.fit(train_config)
    elif model_name in ("decision_tree", "decision tree", "dt"):
        model = DecisionTree(cast(DecisionTreeConfig, model_config))
        model.fit(train_config)
    elif model_name in ("random_forest", "random forest", "rf"):
        model = RandomForest(cast(RandomForestConfig, model_config))
        model.fit(train_config)
    elif model_name == "mgbdt":
        model = MGBDTModel(
            cast(mGBDTConfig, model_config),
            layer_configs=[("tp_layer", "xgb")],
            verbose=train_config.verbose,
        )
        model.fit(train_config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model_config(model: str, input_dim: int, args):
    """Build the ModelConfig for ``model`` from CLI args. Shared by train.py and
    training/train_ensemble.py so both dispatch identically. ``args.seed`` is
    plumbed into dt/rf for reproducibility."""
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
            random_state=args.seed,
        )
    if model == "rf":
        return RandomForestConfig(
            model="rf",
            n_estimators=args.n_estimators,
            max_depth=args.max_depth_rf,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.seed,
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


def print_cv_summary(model_name, variant, k, train_mean, test_mean, test_std):
    """Print the across-fold mean (± std on test) per metric."""
    print(f"\n=== {model_name} | price (variant={variant}) | {k}-fold CV mean ===")
    for metric, value in train_mean.items():
        print(f"Train {METRIC_LABELS[metric] + ':':<14} {value:.4f}")
    for metric, value in test_mean.items():
        print(f"Test  {METRIC_LABELS[metric] + ':':<14} {value:.4f} ± {test_std[metric]:.4f}")


def main(args):
    set_seed(args.seed)

    # Reproducible k-fold CV over the raw engineered frame. The split AND the
    # per-fold location encoding (apply_variant) are done in memory so target
    # encoding is re-fit on each fold's train rows (no leakage). Feature scaling
    # (fit on train) and target MinMax scaling are applied per fold at train time.
    df = load_df(PRICE_PATH)
    X, y = split_X_y(df, TARGET)

    # MLP and mGBDT need different epoch budgets; dt/rf ignore epochs entirely.
    epochs = args.epochs_mgbdt if args.model == "mgbdt" else args.epochs_mlp

    fold_train_metrics = []
    fold_test_metrics = []

    for fold, ((X_train, y_train), (X_test, y_test)) in enumerate(
        kfold_splits(X, y, args.k, args.seed)
    ):
        X_train, X_test = apply_variant(X_train, X_test, y_train, args.variant)
        X_train, X_test = scale_features(X_train.copy(), X_test.copy())
        # target_scaler holds the train {min, max} so predictions can be inverted
        # back to dollars below.
        y_train_scaled, _, target_scaler = scale_target(y_train, y_test)

        train_config = TrainConfig(
            X=X_train,
            y=y_train_scaled,
            epochs=epochs,
            lr=args.lr_mlp,
            batch_size=args.batch_size,
            verbose=args.verbose,
            patience=args.patience,
            val_split=args.val_split,
        )
        model_config = make_model_config(args.model, X_train.shape[1], args)
        model = build_model(model_config, train_config)

        # Invert the MinMax target scaling so metrics are reported in dollars. The
        # true targets are already raw dollars; only predictions need inverting.
        y_train_pred = invert_target(model.predict(X_train), target_scaler)
        y_test_pred = invert_target(model.predict(X_test), target_scaler)
        y_train_true = np.asarray(y_train.values, dtype=float).ravel()
        y_test_true = np.asarray(y_test.values, dtype=float).ravel()

        n_features = X_train.shape[1]
        fold_train_metrics.append(compute_metrics(y_train_true, y_train_pred, n_features))
        fold_test_metrics.append(compute_metrics(y_test_true, y_test_pred, n_features))
        print(
            f"[{args.model}|{args.variant}] fold {fold + 1}/{args.k} "
            f"test Adjusted R²: {fold_test_metrics[-1]['Adjusted_R2']:.4f}"
        )

    train_mean = mean_metrics(fold_train_metrics)
    test_mean = mean_metrics(fold_test_metrics)
    train_std = std_metrics(fold_train_metrics)
    test_std = std_metrics(fold_test_metrics)
    print_cv_summary(args.model, args.variant, args.k, train_mean, test_mean, test_std)

    # Record the across-fold mean (value) and std under split=train/test, one row
    # per metric, so results/visualize_results.ipynb can draw error bars.
    save_split_metrics("price", args.model, args.variant, "train", train_mean, train_std)
    save_split_metrics("price", args.model, args.variant, "test", test_mean, test_std)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a house price regression model")
    parser.add_argument(
        "--seed", type=int, default=42, help="global random seed for reproducibility"
    )
    parser.add_argument(
        "--k", type=int, default=5, help="number of cross-validation folds"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "dt", "rf", "mgbdt"],
        help="model to train (mlp, dt, rf or mgbdt)",
    )

    parser.add_argument(
        "--variant",
        type=str,
        default="tgt",
        choices=["cat", "tgt", "coord_only", "tgt_only"],
        help="location-encoding variant: 'cat' (ordinal codes, no coords), "
        "'tgt' (target-encoded city/zipcode + cartesian x/y/z), "
        "'coord_only' (x/y/z only) or 'tgt_only' (target-encoded address only)",
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
        help="early-stopping patience for mlp (epochs without val improvement; "
        "0 disables early stopping)",
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
