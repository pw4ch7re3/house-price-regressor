import pandas as pd
from sklearn.model_selection import train_test_split

HOUSING_PATH = "data/raw/latlong_added.csv"
PRICE_PATH = "data/processed/usa_housing_dataset_price.csv"
PRICE_PER_SQFT_PATH = "data/processed/usa_housing_dataset_price_per_sqft.csv"


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
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return (X_train, y_train), (X_test, y_test)
