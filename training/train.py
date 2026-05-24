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

from data.dataload import load_df, split_X_y, split_train_test
from metrics.mse import rmse
from metrics.r2_score import r2_score
from models import ModelConfig, TrainConfig
from models.mlp import MLPConfig, MLP
from models.decision_tree import DecisionTreeConfig, DecisionTree


input_path = "data/raw"
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
    data_path = input_path + "/usa_housing_dataset_processed.csv"
    df = load_df(data_path)
    X, y = split_X_y(df, args.target)

    # X = X[["bedrooms", "bathrooms", "floors"]]

    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)

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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "dt"],
                        help="model to train (mlp or dt)")
    parser.add_argument("--max_depth", type=int, default=None,
                        help="max depth for decision tree")
    parser.add_argument("--min_samples_split", type=int, default=2,
                        help="min samples to split for decision tree")
    parser.add_argument("--min_samples_leaf", type=int, default=1,
                        help="min samples per leaf for decision tree")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
