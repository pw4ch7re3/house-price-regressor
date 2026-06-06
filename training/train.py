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

from data.dataload import load_variant
from models import ModelConfig, TrainConfig
from models.mlp import MLPConfig, MLP
from models.decision_tree import DecisionTreeConfig, DecisionTree
from models.random_forest import RandomForestConfig, RandomForest

from models.mgbdt_ours import mGBDTConfig, MGBDTModel

from training.record import print_metrics

output_path = "models/best_models"


def train(model_config: ModelConfig, train_config: TrainConfig, save_name: str):
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
            layer_configs=[("tp_layer", "xgb", 8), ("tp_layer", "xgb")],
            verbose=train_config.verbose,
        )
        model.fit(train_config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    os.makedirs(output_path, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(output_path, f"best_{model_name}_{save_name}.pth"),
    )

    return model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)

    # Split, encoding (target/categorical), and feature + target scaling are
    # all baked into the materialized variant files by data/preprocess.py. The
    # target column is MinMax-scaled; target_scaler holds the train {min, max}
    # so predictions and true targets can be inverted back to dollars below.
    (X_train, y_train), (X_test, y_test), target_scaler = load_variant(args.variant)

    # DT regularization: coarse age bins (computed on the already-scaled age).
    if args.model == "dt":
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train["age_bin"] = pd.cut(
            X_train["age"], bins=5, labels=[0, 1, 2, 3, 4]
        ).astype(float)
        X_test["age_bin"] = pd.cut(
            X_test["age"], bins=5, labels=[0, 1, 2, 3, 4]
        ).astype(float)

    train_config = TrainConfig(
        X=X_train,
        y=y_train,
        epochs=args.epochs,
        lr=args.lr_mlp,
        batch_size=args.batch_size,
        verbose=args.verbose,
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

    # Invert the MinMax target scaling (baked in by preprocessing) so metrics
    # are reported in dollars. Both predictions and the scaled true targets are
    # mapped back: x = scaled * (max - min) + min.
    span = target_scaler["max"] - target_scaler["min"]

    def to_dollars(values):
        return np.asarray(values, dtype=float).ravel() * span + target_scaler["min"]

    y_train_pred = to_dollars(y_train_pred)
    y_test_pred = to_dollars(y_test_pred)
    y_train_true = to_dollars(y_train.values)
    y_test_true = to_dollars(y_test.values)

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
        choices=["cat", "tgt"],
        help="location-encoding variant: 'cat' (ordinal codes, no coords) or "
        "'tgt' (target-encoded city/zipcode + cartesian x/y/z)",
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--lr_mlp", type=float, default=1e-2, help="learning rate for mlp (Adam)"
    )
    parser.add_argument(
        "--lr_mgbdt", type=float, default=0.3, help="learning rate for mgbdt (xgb)"
    )
    parser.add_argument("--batch_size", type=int, default=16)

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
        default=10,
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
