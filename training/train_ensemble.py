import torch
import os
import sys
import random
import argparse
import numpy as np
from typing import cast

from sklearn.model_selection import KFold

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
)
from models import ModelConfig, TrainConfig
from models.random_forest import RandomForestConfig, RandomForest

from metrics.mse import rmse
from training.record import (
    compute_metrics,
    print_metrics,
    save_split_metrics,
    METRIC_LABELS,
)

output_path = "models/best_models"

# Targets reported by the ensemble: the direct price model, the
# price_per_sqft model, and the blended ensemble.
TARGETS = ["price", "price_per_sqft", "price_ensemble"]


def fit_model(model_config: ModelConfig, train_config: TrainConfig):
    model = RandomForest(cast(RandomForestConfig, model_config))
    model.fit(train_config)
    return model


def train(model_config: ModelConfig, train_config: TrainConfig, target_name: str):
    model = fit_model(model_config, train_config)

    os.makedirs(output_path, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(output_path, f"ensemble_rf_{target_name}.pth"),
    )

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


def train_and_predict(
    X_tr, y_tr, X_ev, y_ev, drop_address, model_config, base_train_kwargs, save_models
):
    """Train both models on (X_tr, y_tr), blend, and return predictions for the
    train and eval splits keyed by target. y_tr/y_ev are the raw price."""
    # Model 2 target: price per square foot of living area.
    y_pps_tr = y_tr / X_tr["sqft_living"]
    y_pps_ev = y_ev / X_ev["sqft_living"]

    # Keep sqft_living to convert price_per_sqft predictions back to price.
    sqft_tr = X_tr["sqft_living"].to_numpy()
    sqft_ev = X_ev["sqft_living"].to_numpy()

    if drop_address:
        X_tr = drop_addr(X_tr)
        X_ev = drop_addr(X_ev)
    # Baseline always drops coordinates.
    X_tr = drop_coord(X_tr)
    X_ev = drop_coord(X_ev)

    builder = train if save_models else (
        lambda mc, tc, name: fit_model(mc, tc)
    )

    model1 = builder(
        model_config, TrainConfig(X=X_tr, y=y_tr, **base_train_kwargs), "price"
    )
    model2 = builder(
        model_config,
        TrainConfig(X=X_tr, y=y_pps_tr, **base_train_kwargs),
        "price_per_sqft",
    )

    price1_tr = model1.predict(X_tr)
    price1_ev = model1.predict(X_ev)

    pps2_tr = model2.predict(X_tr)
    pps2_ev = model2.predict(X_ev)
    price2_tr = pps2_tr * sqft_tr
    price2_ev = pps2_ev * sqft_ev

    best_w, _ = optimize_weight(y_tr, price1_tr, price2_tr)
    ens_tr = best_w * price1_tr + (1.0 - best_w) * price2_tr
    ens_ev = best_w * price1_ev + (1.0 - best_w) * price2_ev

    preds = {
        "price": (y_tr, price1_tr, y_ev, price1_ev),
        "price_per_sqft": (y_pps_tr, pps2_tr, y_pps_ev, pps2_ev),
        "price_ensemble": (y_tr, ens_tr, y_ev, ens_ev),
    }
    return preds, best_w, X_tr.shape[1]


def cross_validate(
    X_train, y_train, drop_address, model_config, base_train_kwargs, k_folds, seed
):
    """Run k-fold CV on the training set, returning per-target lists of the
    validation-fold metric dicts and the per-fold weights."""
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    fold_metrics = {t: [] for t in TARGETS}
    weights = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]

        preds, w, n_features = train_and_predict(
            X_tr, y_tr, X_val, y_val, drop_address, model_config, base_train_kwargs,
            save_models=False,
        )
        weights.append(w)
        for target in TARGETS:
            _, _, y_val_true, y_val_pred = preds[target]
            fold_metrics[target].append(
                compute_metrics(y_val_true, y_val_pred, n_features)
            )
        print(f"  fold {fold}/{k_folds}: w={w:.4f}")

    return fold_metrics, weights


def report_cv(fold_metrics, weights, drop_address, k_folds):
    """Print mean +/- std of validation metrics across folds and save the mean."""
    print(
        f"\n=== {k_folds}-fold CV on train set | weight mean: "
        f"{np.mean(weights):.4f} +/- {np.std(weights):.4f} ==="
    )
    for target in TARGETS:
        metric_keys = fold_metrics[target][0].keys()
        means = {
            m: float(np.mean([fm[m] for fm in fold_metrics[target]]))
            for m in metric_keys
        }
        stds = {
            m: float(np.std([fm[m] for fm in fold_metrics[target]]))
            for m in metric_keys
        }
        print(f"--- ensemble | {target} (CV val) ---")
        for m in metric_keys:
            print(f"CV {METRIC_LABELS[m] + ':':<14} {means[m]:.4f} +/- {stds[m]:.4f}")
        save_split_metrics(target, "ensemble", f"ens_da{int(drop_address)}", "cv", means)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)

    # Both models train on the same price dataset. price_per_sqft is derived on
    # the fly by dividing price by sqft_living (still a feature here).
    df = load_df(PRICE_PATH)

    X, y = split_X_y(df, "price")

    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)

    model_config = RandomForestConfig(
        model="rf",
        n_estimators=args.n_estimators,
        max_depth=args.max_depth_rf,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
    )

    base_train_kwargs = dict(
        epochs=args.epochs,
        lr=args.lr_mlp,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    # --- K-fold cross-validation on the train set (held-out test untouched) ---
    fold_metrics, weights = cross_validate(
        X_train, y_train, args.drop_address, model_config, base_train_kwargs,
        args.k_folds, args.seed,
    )
    report_cv(fold_metrics, weights, args.drop_address, args.k_folds)

    # --- Final fit on the full train set, evaluated on the held-out test set ---
    preds, best_w, n_features = train_and_predict(
        X_train, y_train, X_test, y_test, args.drop_address, model_config,
        base_train_kwargs, save_models=True,
    )
    print(
        f"\n=== final holdout | ensemble weight (optimized on full train) ===\n"
        f"w={best_w:.4f}  -> price = {best_w:.4f}*price_1 + "
        f"{1 - best_w:.4f}*(price_per_sqft_2 * sqft_living)"
    )

    for target in TARGETS:
        y_tr_true, y_tr_pred, y_te_true, y_te_pred = preds[target]
        print_metrics(
            target,
            "ensemble",
            f"ens_da{int(args.drop_address)}",
            y_tr_true,
            y_tr_pred,
            y_te_true,
            y_te_pred,
            n_features,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an RF ensemble: blend a direct price model with a "
        "price_per_sqft model (weight optimized on train), evaluated with k-fold CV"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="global random seed for reproducibility"
    )

    parser.add_argument(
        "--drop_address",
        action="store_true",
        help="drop address (city, zipcode)",
    )

    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="number of folds for cross-validation on the train set",
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
