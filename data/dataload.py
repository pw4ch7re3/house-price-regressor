import json

import pandas as pd
from sklearn.model_selection import train_test_split

HOUSING_PATH = "data/raw/latlong_added.csv"
PRICE_PATH = "data/processed/usa_housing_dataset_price.csv"
PRICE_PER_SQFT_PATH = "data/processed/usa_housing_dataset_price_per_sqft.csv"

PRICE_TEST_PATH = "data/processed/usa_housing_dataset_price_test.csv"

# Materialized, model-ready variant datasets (split + encoded + scaled by
# data/preprocess.py). Each CSV holds X and the scaled target together.
VARIANTS = ("cat", "tgt")
VARIANT_PATHS = {
    "cat": {
        "train": "data/processed/cat_train.csv",
        "test": "data/processed/cat_test.csv",
    },
    "tgt": {
        "train": "data/processed/tgt_train.csv",
        "test": "data/processed/tgt_test.csv",
    },
}
TARGET = "price"


def target_scaler_path(variant: str) -> str:
    return f"data/processed/target_scaler_{variant}.json"


def load_df(pathname: str):
    return pd.read_csv(pathname)


def drop_addr(df: pd.DataFrame):
    return df.drop(columns=["city", "zipcode"])


def drop_coord(df: pd.DataFrame):
    return df.drop(columns=["x", "y", "z"])


def target_encode(train_X, train_y, test_X, col, smoothing=10):
    global_mean = train_y.mean()
    tmp = train_X[[col]].copy()
    tmp["_target"] = train_y.values
    stats = tmp.groupby(col)["_target"].agg(["mean", "count"])
    smooth = (stats["count"] * stats["mean"] + smoothing * global_mean) / (
        stats["count"] + smoothing
    )
    train_enc = train_X[col].map(smooth).fillna(global_mean)
    test_enc = test_X[col].map(smooth).fillna(global_mean)
    return train_enc, test_enc


def split_X_y(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def split_train_test(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (X_train, y_train), (X_test, y_test)


def load_variant(variant: str):
    """Load a materialized variant dataset.

    Returns ``((X_train, y_train), (X_test, y_test), target_scaler)`` where
    ``target_scaler`` is the ``{"min": ..., "max": ...}`` dict used by
    preprocessing to MinMax-scale the target, so callers can inverse-transform
    predictions back to the original (dollar) scale.
    """
    if variant not in VARIANT_PATHS:
        raise ValueError(f"Unknown variant: {variant} (expected one of {VARIANTS})")

    train_df = load_df(VARIANT_PATHS[variant]["train"])
    test_df = load_df(VARIANT_PATHS[variant]["test"])

    X_train, y_train = split_X_y(train_df, TARGET)
    X_test, y_test = split_X_y(test_df, TARGET)

    with open(target_scaler_path(variant)) as f:
        target_scaler = json.load(f)

    return (X_train, y_train), (X_test, y_test), target_scaler
