import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

HOUSING_PATH = "data/raw/latlong_added.csv"

PRICE_PATH = "data/processed/usa_housing_dataset_price.csv"
PRICE_PER_SQFT_PATH = "data/processed/usa_housing_dataset_price_per_sqft.csv"

MINMAX_COLS = [
    "x",
    "y",
    "z",
    "condition",
    "age",
    "bedrooms",
    "bathrooms",
    "floors",
    "view",
]

ZSCORE_COLS = [
    "sqft_living",
    "sqft_above",
    "sqft_basement",
    "log_sqft_lot",
    "city",
    "zipcode",
]


def load_df(pathname: str):
    return pd.read_csv(pathname)


def drop_addr(df: pd.DataFrame):
    return df.drop(columns=["street", "city", "statezip"])


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
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    seed: int = 42,
    is_dt: bool = False,
    log: bool = True,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    if log:
        y_train = np.log1p(y_train)

    for col in ["city", "zipcode"]:
        X_train[col], X_test[col] = target_encode(X_train, y_train, X_test, col)

    # Minmax Regularization
    scaler_mm = MinMaxScaler()
    X_train[MINMAX_COLS] = scaler_mm.fit_transform(X_train[MINMAX_COLS])
    X_test[MINMAX_COLS] = scaler_mm.transform(X_test[MINMAX_COLS])

    # Z-score Regularization
    scaler = StandardScaler()
    X_train[ZSCORE_COLS] = scaler.fit_transform(X_train[ZSCORE_COLS])
    X_test[ZSCORE_COLS] = scaler.transform(X_test[ZSCORE_COLS])

    if is_dt:
        # DT regularization
        X_train["age_bin"], bins = pd.cut(
            X_train["age"], bins=5, labels=[0, 1, 2, 3, 4], retbins=True
        ).astype(float)

        X_test["age_bin"] = pd.cut(
            X_test["age"], bins=bins, labels=[0, 1, 2, 3, 4]
        ).astype(float)

    return (X_train, y_train), (X_test, y_test)
