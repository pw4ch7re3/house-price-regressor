import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import argparse
from typing import cast

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if path not in sys.path:
    sys.path.insert(0, path)

from sklearn.preprocessing import StandardScaler

from data.dataload import load_df, drop_addr, drop_coord, split_X_y, split_train_test
from data.preprocess import FILL_PATH, DROP_PATH, FILL_THEN_DROP_PATH
from metrics.mse import rmse
from metrics.r2_score import r2_score

from models import ModelConfig, TrainConfig
from models.mlp import MLPConfig, MLP
from models.decision_tree import DecisionTreeConfig, DecisionTree

# input_path = "data/processed"
output_path = "models/best_models"


def train(model_config: ModelConfig, train_config: TrainConfig):
    model_name = model_config.model.lower()
    if model_name == "mlp":
        model = MLP(cast(MLPConfig, model_config))
    elif model_name in ("decision_tree", "decision tree", "dt"):
        model = DecisionTree(cast(DecisionTreeConfig, model_config))
    elif model_name == "mgbdt":
        raise ValueError(f"TODO. implement mgbdt")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(train_config)

    os.makedirs(output_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_path, f"best_{model_name}.pth"))

    return model


def main(args):
    if args.null == "fill":
        data_path = FILL_PATH
    elif args.null == "drop":
        data_path = DROP_PATH
    elif args.null == "fill_then_drop":
        data_path = FILL_THEN_DROP_PATH
    else:
        raise ValueError(f"Unknown null preprocess method: {args.null}")

    df = load_df(data_path)

    if args.drop_address:
        df = drop_addr(df)
    if args.drop_coord:
        df = drop_coord(df)

    X, y = split_X_y(df, args.target)

    # X = X[["bedrooms", "bathrooms", "floors"]]

    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_config = TrainConfig(
        X=X_train,
        y=y_train,
        epochs=args.epochs,
        lr=args.lr,
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
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = train(model_config, train_config)

    y_train_pred = model.predict(X_train)
    print(f"Train  RMSE: {rmse(y_train, y_train_pred):.4f}")
    print(f"Train  R²:   {r2_score(y_train, y_train_pred):.4f}")

    y_test_pred = model.predict(X_test)
    print(f"Test  RMSE: {rmse(y_test, y_test_pred):.4f}")
    print(f"Test  R²:   {r2_score(y_test, y_test_pred):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a house price regression model")
    parser.add_argument("--target", type=str, default="price")
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "dt"],
        help="model to train (mlp or dt)",
    )

    parser.add_argument(
        "--null",
        type=str,
        default="fill",
        choices=["fill", "drop", "fill_then_drop"],
        help="how to process null coordinates?",
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

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument(
        "--max_depth", type=int, default=6, help="max depth for decision tree"
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

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
