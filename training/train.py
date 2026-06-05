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

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from data.dataload import (
    load_df,
    drop_addr,
    drop_coord,
    target_encode,
    split_X_y,
    split_train_test,
    PRICE_PATH,
    PRICE_PER_SQFT_PATH,
)
from models import ModelConfig, TrainConfig
from models.mlp import MLPConfig, MLP
from models.decision_tree import DecisionTreeConfig, DecisionTree
from models.random_forest import RandomForestConfig, RandomForest

from models.mgbdt_ours import mGBDTConfig, MGBDTModel

from training.record import print_metrics

# input_path = "data/processed"
output_path = "models/best_models"

MINMAX_COLS = [
    "x",
    "y",
    "z",
    "condition",
    "age",
    "bedrooms",
    "bathrooms",
    "floors",
    "view",
]

ZSCORE_COLS = [
    "sqft_living",
    "sqft_above",
    "sqft_basement",
    "log_sqft_lot",
    "city",
    "zipcode",
]


def train(model_config: ModelConfig, train_config: TrainConfig, target_name: str):
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
        os.path.join(output_path, f"best_{model_name}_{target_name}.pth"),
    )

    return model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)

    for data_path in [PRICE_PATH, PRICE_PER_SQFT_PATH]:
        df = load_df(data_path)

        X, y = split_X_y(df, "price")

        (X_train, y_train), (X_test, y_test) = split_train_test(X, y)

        # Target transformation (minmax). Always scale the target: the linear
        # mGBDT path takes SGD steps whose gradients scale with the target
        # magnitude, so an unnormalized target (e.g. price_per_sqft ~ hundreds)
        # makes the loss diverge to inf/nan. XGB is scale-invariant and hides
        # this, but linear is not.
        y_train_raw = y_train
        target_scaler = MinMaxScaler()
        y_train = pd.Series(
            target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel(),
            index=y_train.index,
        )

        # Target Encoding
        for col in ["city", "zipcode"]:
            X_train[col], X_test[col] = target_encode(X_train, y_train, X_test, col)

        # Minmax Regularization
        scaler_mm = MinMaxScaler()
        X_train[MINMAX_COLS] = scaler_mm.fit_transform(X_train[MINMAX_COLS])
        X_test[MINMAX_COLS] = scaler_mm.transform(X_test[MINMAX_COLS])

        # Z-score Regularization
        scaler = StandardScaler()
        X_train[ZSCORE_COLS] = scaler.fit_transform(X_train[ZSCORE_COLS])
        X_test[ZSCORE_COLS] = scaler.transform(X_test[ZSCORE_COLS])

        # DT regularization
        if args.model == "dt":
            X_train["age_bin"] = pd.cut(
                X_train["age"], bins=5, labels=[0, 1, 2, 3, 4]
            ).astype(float)
            X_test["age_bin"] = pd.cut(
                X_test["age"], bins=5, labels=[0, 1, 2, 3, 4]
            ).astype(float)

        if args.drop_address:
            X_train = drop_addr(X_train)
            X_test = drop_addr(X_test)
        if args.drop_coord:
            X_train = drop_coord(X_train)
            X_test = drop_coord(X_test)

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

        if data_path == PRICE_PATH:
            model = train(model_config, train_config, "price")
        else:
            model = train(model_config, train_config, "price_per_sqft")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Inverse-transform predictions back to the original target scale
        y_train_pred = target_scaler.inverse_transform(
            np.asarray(y_train_pred).reshape(-1, 1)
        ).ravel()
        y_test_pred = target_scaler.inverse_transform(
            np.asarray(y_test_pred).reshape(-1, 1)
        ).ravel()

        n_features = X_train.shape[1]
        target_name = "price" if data_path == PRICE_PATH else "price_per_sqft"

        print_metrics(
            target_name,
            args.model,
            args.drop_address,
            args.drop_coord,
            y_train_raw,
            y_train_pred,
            y_test,
            y_test_pred,
            n_features,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a house price regression model")
    # parser.add_argument("--target", type=str, default="price")
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
        "--drop_address",
        action="store_true",
        help="drop address (city, street, statezip)",
    )
    parser.add_argument(
        "--drop_coord",
        action="store_true",
        help="drop coord (x, y, z)",
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--lr_mlp", type=float, default=1e-2, help="learning rate for mlp (Adam)"
    )
    parser.add_argument(
        "--lr_mgbdt", type=float, default=0.1, help="learning rate for mgbdt (xgb)"
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
        "--max_depth_mgbdt", type=int, default=5, help="max depth for mgbdt (xgb)"
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
