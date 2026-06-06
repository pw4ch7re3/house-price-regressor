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

from data.dataload import load_variant, scale_features, scale_target, invert_target
from models import ModelConfig, TrainConfig
from models.mlp import MLPConfig, MLP
from models.decision_tree import DecisionTreeConfig, DecisionTree
from models.random_forest import RandomForestConfig, RandomForest

from models.mgbdt_ours import mGBDTConfig, MGBDTModel

from training.record import print_metrics

output_path = "models/best_models"


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


def train(model_config: ModelConfig, train_config: TrainConfig, save_name: str):
    model = build_model(model_config, train_config)

    os.makedirs(output_path, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(output_path, f"best_{model_config.model.lower()}_{save_name}.pth"),
    )

    return model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)

    # Split and location-encoding are baked into the materialized variant files
    # by data/preprocess.py; the files are kept on the RAW scale. Feature scaling
    # (fit on train) and target MinMax scaling are applied here at train time.
    (X_train, y_train), (X_test, y_test) = load_variant(args.variant)
    X_train, X_test = scale_features(X_train.copy(), X_test.copy())
    # target_scaler holds the train {min, max} so predictions and true targets
    # can be inverted back to dollars below.
    y_train_scaled, y_test_scaled, target_scaler = scale_target(y_train, y_test)

    # MLP and mGBDT need different epoch budgets; dt/rf ignore epochs entirely.
    epochs = args.epochs_mgbdt if args.model == "mgbdt" else args.epochs_mlp

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

    if args.model == "mlp":
        model_config = MLPConfig(
            model="mlp",
            input_dim=X_train.shape[1],
            hidden_dims=[32, 32],
            output_dim=1,
        )
    elif args.model == "dt":
        model_config = DecisionTreeConfig(
            model="dt",
            max_depth=args.max_depth_dt,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
        )
    elif args.model == "rf":
        model_config = RandomForestConfig(
            model="rf",
            n_estimators=args.n_estimators,
            max_depth=args.max_depth_rf,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
        )
    elif args.model == "mgbdt":
        model_config = mGBDTConfig(
            model="mgbdt",
            input_size=X_train.shape[1],
            output_size=1,
            task="regression",
            learning_rate=args.lr_mgbdt,
            max_depth=args.max_depth_mgbdt,
            num_boost_round=args.num_boost_round,
            target_lr=args.target_lr,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = train(model_config, train_config, args.variant)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Invert the MinMax target scaling so metrics are reported in dollars. The
    # true targets are already raw dollars; only predictions need inverting.
    y_train_pred = invert_target(y_train_pred, target_scaler)
    y_test_pred = invert_target(y_test_pred, target_scaler)
    y_train_true = np.asarray(y_train.values, dtype=float).ravel()
    y_test_true = np.asarray(y_test.values, dtype=float).ravel()

    n_features = X_train.shape[1]

    print_metrics(
        "price",
        args.model,
        args.variant,
        y_train_true,
        y_train_pred,
        y_test_true,
        y_test_pred,
        n_features,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a house price regression model")
    parser.add_argument(
        "--seed", type=int, default=42, help="global random seed for reproducibility"
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
