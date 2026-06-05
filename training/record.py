# saving results
import os
import csv
from datetime import datetime

from metrics.mse import rmse
from metrics.r2_score import r2_score
from metrics.mae import mae
from metrics.mape import mape
from metrics.adjusted_r2 import adjusted_r2

METRICS_PATH = "results/metrics.csv"

# Metric key -> display label used when printing
METRIC_LABELS = {
    "RMSE": "RMSE",
    "MAE": "MAE",
    "MAPE": "MAPE",
    "R2": "R²",
    "Adjusted_R2": "Adjusted R²",
}


def compute_metrics(y_true, y_pred, n_features):
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "Adjusted_R2": adjusted_r2(y_true, y_pred, n_features),
    }


def save_metrics(
    target_name, model_name, drop_address, drop_coord, train_metrics, test_metrics
):
    """Append metrics to a long-format CSV for later visualization."""
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    file_exists = os.path.exists(METRICS_PATH)
    timestamp = datetime.now().isoformat(timespec="seconds")

    with open(METRICS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "model",
                    "target",
                    "drop_address",
                    "drop_coord",
                    "split",
                    "metric",
                    "value",
                ]
            )
        for split, metrics in (("train", train_metrics), ("test", test_metrics)):
            for metric, value in metrics.items():
                writer.writerow(
                    [
                        timestamp,
                        model_name,
                        target_name,
                        int(drop_address),
                        int(drop_coord),
                        split,
                        metric,
                        value,
                    ]
                )


def save_split_metrics(
    target_name, model_name, drop_address, drop_coord, split, metrics
):
    """Append a single split's metrics (e.g. cross-validation mean) to the CSV."""
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    file_exists = os.path.exists(METRICS_PATH)
    timestamp = datetime.now().isoformat(timespec="seconds")

    with open(METRICS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "model",
                    "target",
                    "drop_address",
                    "drop_coord",
                    "split",
                    "metric",
                    "value",
                ]
            )
        for metric, value in metrics.items():
            writer.writerow(
                [
                    timestamp,
                    model_name,
                    target_name,
                    int(drop_address),
                    int(drop_coord),
                    split,
                    metric,
                    value,
                ]
            )


def print_metrics(
    target_name,
    model_name,
    drop_address,
    drop_coord,
    y_train_true,
    y_train_pred,
    y_test_true,
    y_test_pred,
    n_features,
):
    train_metrics = compute_metrics(y_train_true, y_train_pred, n_features)
    test_metrics = compute_metrics(y_test_true, y_test_pred, n_features)

    print(
        f"=== {model_name} | {target_name} "
        f"(drop_address={drop_address}, drop_coord={drop_coord}) ==="
    )
    for metric, value in train_metrics.items():
        print(f"Train {METRIC_LABELS[metric] + ':':<14} {value:.4f}")
    for metric, value in test_metrics.items():
        print(f"Test  {METRIC_LABELS[metric] + ':':<14} {value:.4f}")

    save_metrics(
        target_name, model_name, drop_address, drop_coord, train_metrics, test_metrics
    )
