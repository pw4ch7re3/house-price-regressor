import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler

HOUSING_PATH = "data/raw/latlong_added.csv"
PRICE_PATH = "data/processed/usa_housing_dataset_price.csv"
PRICE_PER_SQFT_PATH = "data/processed/usa_housing_dataset_price_per_sqft.csv"

PRICE_TEST_PATH = "data/processed/usa_housing_dataset_price_test.csv"

# Materialized variant datasets (split + location-encoded by data/preprocess.py).
# Features and the price target are kept on their RAW scales here; feature and
# target scaling are applied at train time (see scale_features / scale_target).
# Each CSV holds X and the raw target together.
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

# Derived variants that isolate a single location signal from the 'tgt' files
# (which carry both target-encoded city/zipcode and cartesian x/y/z). They are
# built by dropping columns at load time, so they share the 'tgt' split,
# encoding, scaling and target scaler.
#   coord_only -> location via x/y/z only      (drop city/zipcode)
#   tgt_only   -> location via target-encoding  (drop x/y/z)
DERIVED_VARIANTS = {
    "coord_only": ("tgt", "drop_addr"),
    "tgt_only": ("tgt", "drop_coord"),
}

VARIANTS = tuple(VARIANT_PATHS) + tuple(DERIVED_VARIANTS)
TARGET = "price"

# Feature scaling column lists, applied at train time (fit on train, applied to
# both splits). Only columns actually present are scaled (e.g. the 'cat' variant
# has no x/y/z; derived variants drop one location signal).
MINMAX_COLS = ["x", "y", "z", "condition", "age", "bedrooms", "bathrooms", "floors", "view"]
ZSCORE_COLS = ["sqft_living", "sqft_above", "sqft_basement", "log_sqft_lot", "city", "zipcode"]


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


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """MinMax + Z-score feature scaling, fit on train, applied to both. Only
    columns actually present are scaled (some variants drop x/y/z or city/zip)."""
    minmax_cols = [c for c in MINMAX_COLS if c in X_train.columns]
    zscore_cols = [c for c in ZSCORE_COLS if c in X_train.columns]

    mm = MinMaxScaler()
    X_train[minmax_cols] = mm.fit_transform(X_train[minmax_cols])
    X_test[minmax_cols] = mm.transform(X_test[minmax_cols])

    zs = StandardScaler()
    X_train[zscore_cols] = zs.fit_transform(X_train[zscore_cols])
    X_test[zscore_cols] = zs.transform(X_test[zscore_cols])

    return X_train, X_test


def scale_target(y_train, y_test):
    """MinMax-scale a target, fit on train. Returns ``(y_train_scaled,
    y_test_scaled, scaler)`` where ``scaler`` is a ``{"min", "max"}`` dict that
    ``invert_target`` uses to map predictions back to the original scale."""
    y_train = np.asarray(y_train, dtype=float).reshape(-1, 1)
    y_test = np.asarray(y_test, dtype=float).reshape(-1, 1)

    mm = MinMaxScaler()
    y_train_scaled = mm.fit_transform(y_train).ravel()
    y_test_scaled = mm.transform(y_test).ravel()
    scaler = {"min": float(mm.data_min_[0]), "max": float(mm.data_max_[0])}
    return y_train_scaled, y_test_scaled, scaler


def invert_target(values, scaler):
    """Inverse of ``scale_target``: map scaled values back to the original scale
    via ``values * (max - min) + min``."""
    span = scaler["max"] - scaler["min"]
    return np.asarray(values, dtype=float).ravel() * span + scaler["min"]


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


def kfold_splits(X: pd.DataFrame, y: pd.Series, n_splits: int = 5, seed: int = 42):
    """Yield reproducible k-fold train/test splits of the raw engineered frame.

    Each yielded item mirrors ``split_train_test``'s shape:
    ``((X_train, y_train), (X_test, y_test))``. The fold assignment is fixed by
    ``KFold(shuffle=True, random_state=seed)`` so runs are reproducible.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        yield (X_train, y_train), (X_test, y_test)


def apply_variant(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, variant: str
):
    """Build one location-encoding variant from a raw split, at train time.

    Single source of truth for variant construction (shared by training and by
    data/preprocess.py). Target encoding is fit on ``y_train`` only, so it is
    safe to call per cross-validation fold without leakage.

    - ``cat``: ordinal city/zipcode codes, drop cartesian x/y/z.
    - ``tgt``: target-encoded city/zipcode (fit on train) plus x/y/z.
    - ``coord_only``: ``tgt`` then drop city/zipcode (x/y/z only).
    - ``tgt_only``: ``tgt`` then drop x/y/z (target-encoded address only).

    Returns the RAW ``(X_train, X_test)``; callers apply ``scale_features`` /
    ``scale_target`` afterwards.
    """
    X_train, X_test = X_train.copy(), X_test.copy()

    if variant == "cat":
        return drop_coord(X_train), drop_coord(X_test)

    if variant in ("tgt", "coord_only", "tgt_only"):
        for col in ["city", "zipcode"]:
            X_train[col], X_test[col] = target_encode(X_train, y_train, X_test, col)
        if variant == "coord_only":
            return drop_addr(X_train), drop_addr(X_test)
        if variant == "tgt_only":
            return drop_coord(X_train), drop_coord(X_test)
        return X_train, X_test

    raise ValueError(f"Unknown variant: {variant} (expected one of {VARIANTS})")


def load_variant(variant: str):
    """Load a materialized variant dataset on its RAW scale.

    Returns ``((X_train, y_train), (X_test, y_test))``. Features and the price
    target are unscaled; callers apply ``scale_features`` / ``scale_target`` at
    train time.

    Derived variants (``coord_only``, ``tgt_only``) are built by dropping
    location columns from the base ``tgt`` files at load time.
    """
    if variant in DERIVED_VARIANTS:
        base, dropper = DERIVED_VARIANTS[variant]
        (X_train, y_train), (X_test, y_test) = load_variant(base)
        drop = {"drop_addr": drop_addr, "drop_coord": drop_coord}[dropper]
        return (drop(X_train), y_train), (drop(X_test), y_test)

    if variant not in VARIANT_PATHS:
        raise ValueError(f"Unknown variant: {variant} (expected one of {VARIANTS})")

    train_df = load_df(VARIANT_PATHS[variant]["train"])
    test_df = load_df(VARIANT_PATHS[variant]["test"])

    X_train, y_train = split_X_y(train_df, TARGET)
    X_test, y_test = split_X_y(test_df, TARGET)

    return (X_train, y_train), (X_test, y_test)
