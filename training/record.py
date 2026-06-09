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


def mean_metrics(fold_metrics):
    """Average a list of per-fold metric dicts into a single mean-per-metric dict.

    Keys are taken from the first dict (every fold shares the same metric set).
    """
    keys = fold_metrics[0].keys()
    n = len(fold_metrics)
    return {k: sum(m[k] for m in fold_metrics) / n for k in keys}


def std_metrics(fold_metrics):
    """Population std of each metric across folds (for printing mean ± std)."""
    keys = fold_metrics[0].keys()
    means = mean_metrics(fold_metrics)
    n = len(fold_metrics)
    return {
        k: (sum((m[k] - means[k]) ** 2 for m in fold_metrics) / n) ** 0.5 for k in keys
    }


# Long-format schema. ``std`` holds the across-fold standard deviation for
# cross-validation rows, and is left blank for single-split rows.
METRICS_HEADER = [
    "timestamp",
    "model",
    "target",
    "variant",
    "split",
    "metric",
    "value",
    "std",
]


def save_metrics(target_name, model_name, variant, train_metrics, test_metrics):
    """Append metrics to a long-format CSV for later visualization."""
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    file_exists = os.path.exists(METRICS_PATH)
    timestamp = datetime.now().isoformat(timespec="seconds")

    with open(METRICS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(METRICS_HEADER)
        for split, metrics in (("train", train_metrics), ("test", test_metrics)):
            for metric, value in metrics.items():
                # Single-split metrics carry no across-fold std.
                writer.writerow(
                    [timestamp, model_name, target_name, variant, split, metric, value, ""]
                )


def save_split_metrics(target_name, model_name, variant, split, metrics, stds=None):
    """Append a single split's metrics (e.g. cross-validation mean) to the CSV.

    ``stds`` is an optional ``{metric: std}`` dict (e.g. from ``std_metrics``); when
    given, each row's ``std`` column holds the across-fold standard deviation.
    """
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    file_exists = os.path.exists(METRICS_PATH)
    timestamp = datetime.now().isoformat(timespec="seconds")

    with open(METRICS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(METRICS_HEADER)
        for metric, value in metrics.items():
            std_val = "" if stds is None else stds.get(metric, "")
            writer.writerow(
                [timestamp, model_name, target_name, variant, split, metric, value, std_val]
            )


def print_metrics(
    target_name,
    model_name,
    variant,
    y_train_true,
    y_train_pred,
    y_test_true,
    y_test_pred,
    n_features,
):
    train_metrics = compute_metrics(y_train_true, y_train_pred, n_features)
    test_metrics = compute_metrics(y_test_true, y_test_pred, n_features)

    print(f"=== {model_name} | {target_name} (variant={variant}) ===")
    for metric, value in train_metrics.items():
        print(f"Train {METRIC_LABELS[metric] + ':':<14} {value:.4f}")
    for metric, value in test_metrics.items():
        print(f"Test  {METRIC_LABELS[metric] + ':':<14} {value:.4f}")

    save_metrics(target_name, model_name, variant, train_metrics, test_metrics)
