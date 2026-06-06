import torch
import os
import sys
import random
import argparse
import numpy as np

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if path not in sys.path:
    sys.path.insert(0, path)

from data.dataload import load_variant
from models import TrainConfig
from models.random_forest import RandomForestConfig, RandomForest

from metrics.mse import rmse
from training.record import print_metrics

output_path = "models/best_models"

# Targets reported by the ensemble: the direct price model, the price_per_sqft
# model, and the blended ensemble.
TARGETS = ["price", "price_per_sqft", "price_ensemble"]


def fit_rf(model_config: RandomForestConfig, X_train, y_train, save_name: str):
    """Fit a RandomForest on raw (unscaled) values and persist its state dict.
    RandomForest is invariant to per-feature scaling and to target scaling, so
    everything is trained directly on raw price / price_per_sqft."""
    model = RandomForest(model_config)
    # RandomForest.fit only reads X and y; epochs/lr are unused placeholders.
    model.fit(TrainConfig(X=X_train, y=y_train, epochs=0, lr=0.0))

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

    # Raw variant data: features and price are unscaled. The ensemble trains a
    # RandomForest directly on these raw values (no feature or target scaling).
    (X_train, y_train), (X_test, y_test) = load_variant(args.variant)

    # Raw living area (per row): used to build the pps target and to convert pps
    # predictions back to price.
    sqft_train = X_train["sqft_living"].to_numpy()
    sqft_test = X_test["sqft_living"].to_numpy()

    price_train = y_train.to_numpy(dtype=float)
    price_test = y_test.to_numpy(dtype=float)

    # Model 2 target: price per square foot of living area (raw dollars / sqft).
    pps_train = price_train / sqft_train
    pps_test = price_test / sqft_test

    model_config = RandomForestConfig(
        model="rf",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth_rf,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
    )

    name = f"ensemble_rf_{args.variant}"
    model1 = fit_rf(model_config, X_train, price_train, f"{name}_price")
    model2 = fit_rf(model_config, X_train, pps_train, f"{name}_price_per_sqft")

    # Predictions are already in raw units.
    price1_tr = model1.predict(X_train)
    price1_te = model1.predict(X_test)

    pps2_tr = model2.predict(X_train)
    pps2_te = model2.predict(X_test)
    price2_tr = pps2_tr * sqft_train
    price2_te = pps2_te * sqft_test

    # Blend weight optimized on the train split, held-out test evaluated once.
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

    for target in TARGETS:
        y_tr_true, y_tr_pred, y_te_true, y_te_pred = preds[target]
        print_metrics(
            target,
            "ensemble_rf",
            args.variant,
            y_tr_true,
            y_tr_pred,
            y_te_true,
            y_te_pred,
            n_features,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a RandomForest ensemble on raw values: blend a direct "
        "price model with a price_per_sqft model (weight optimized on the train "
        "split), evaluated on the held-out test set"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="global random seed for reproducibility"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="tgt",
        choices=["cat", "tgt", "coord_only", "tgt_only"],
        help="location-encoding variant (same as train.py)",
    )

    parser.add_argument(
        "--max_depth_rf", type=int, default=None, help="max depth for random forest"
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        default=25,
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

    args = parser.parse_args()
    main(args)
