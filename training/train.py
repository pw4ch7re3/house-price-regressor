import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import sys

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if path not in sys.path:
    sys.path.insert(0, path)

from data.dataload import load_df, split_X_y, split_train_test
from models import ModelConfig, TrainConfig
from models.mlp import MLPConfig, MLP


input_path = "data/raw"
output_path = "models/best_models"


def train(model_config: ModelConfig, train_config: TrainConfig):
    model_name = model_config.model.lower()
    if model_name == "mlp":
        model = MLP(model_config)
        model.fit(train_config)

        os.makedirs(output_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(output_path, f"best_{model_name}.pth"))


def main():
    df = load_df(input_path + "/" + "usa_housing_dataset.csv")
    X, y = split_X_y(df, "price")

    X = X[["bedrooms", "bathrooms", "floors"]]

    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)

    train_config = TrainConfig(
        X=X_train,
        y=y_train,
        epochs=10,
        lr=1e-3,
        batch_size=None,
        verbose=True,
    )

    mlp_config = MLPConfig(
        model="mlp",
        input_dim=X_train.shape[1],
        hidden_dims=[32, 32],
        output_dim=1,
    )

    train(mlp_config, train_config)


if __name__ == "__main__":
    main()