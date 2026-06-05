import torch
import os
import sys
import random
import argparse
import numpy as np
from typing import cast

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if path not in sys.path:
    sys.path.insert(0, path)

from data.dataload import (
    load_df,
    drop_addr,
    drop_coord,
    split_X_y,
    split_train_test,
    PRICE_PATH,
    PRICE_PER_SQFT_PATH,
)
from models import ModelConfig, TrainConfig
from models.random_forest import RandomForestConfig, RandomForest

from training.record import print_metrics

output_path = "models/best_models"


def train(model_config: ModelConfig, train_config: TrainConfig, target_name: str):
    model = RandomForest(cast(RandomForestConfig, model_config))
    model.fit(train_config)

    os.makedirs(output_path, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(output_path, f"baseline_rf_{target_name}.pth"),
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

        # Random forest is scale-invariant, so the target is left unscaled.
        y_train_raw = y_train

        # Baseline: no target encoding and no feature regularization. city/
        # zipcode stay as their raw integer codes; all features are left
        # unscaled (minmax & z-score regularization excluded).

        if args.drop_address:
            X_train = drop_addr(X_train)
            X_test = drop_addr(X_test)
        # Baseline always drops coordinates.
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

        model_config = RandomForestConfig(
            model="rf",
            n_estimators=args.n_estimators,
            max_depth=args.max_depth_rf,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
        )

        if data_path == PRICE_PATH:
            model = train(model_config, train_config, "price")
        else:
            model = train(model_config, train_config, "price_per_sqft")

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        n_features = X_train.shape[1]
        target_name = "price" if data_path == PRICE_PATH else "price_per_sqft"

        print_metrics(
            target_name,
            "rf",
            args.drop_address,
            True,
            y_train_raw,
            y_train_pred,
            y_test,
            y_test_pred,
            n_features,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the random forest baseline (no target encoding, drops coords)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="global random seed for reproducibility"
    )

    parser.add_argument(
        "--drop_address",
        action="store_true",
        help="drop address (city, zipcode)",
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--lr_mlp", type=float, default=1e-2, help="learning rate for mlp (Adam)"
    )
    parser.add_argument("--batch_size", type=int, default=16)

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
        "--min_samples_split",
        type=int,
        default=2,
        help="min samples to split",
    )
    parser.add_argument(
        "--min_samples_leaf",
        type=int,
        default=2,
        help="min samples per leaf",
    )

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)
