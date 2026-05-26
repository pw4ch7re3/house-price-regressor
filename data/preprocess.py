import re
import pandas as pd
import numpy as np

from data.dataload import load_df
from sklearn.preprocessing import StandardScaler

HOUSING_PATH = "data/raw/latlong_added.csv"

FILL_PATH = "data/processed/usa_housing_dataset_processed.csv"
DROP_PATH = "data/processed/usa_housing_dataset_dropped.csv"
FILL_THEN_DROP_PATH = "data/processed/usa_housing_dataset_fill_dropped.csv"


def latlong2cartesian(lat, long):
    lat_rad = np.radians(lat)
    long_rad = np.radians(long)

    x = np.cos(lat_rad) * np.cos(long_rad)
    y = np.cos(lat_rad) * np.sin(long_rad)
    z = np.sin(lat_rad)
    return x, y, z


def add_cartesian(housing: pd.DataFrame) -> pd.DataFrame:
    x, y, z = latlong2cartesian(housing["lat"], housing["long"])
    housing = housing.assign(x=x, y=y, z=z).drop(columns=["lat", "long"])
    return housing


def normalize_price(housing: pd.DataFrame) -> pd.DataFrame:
    housing = housing[housing["price"] > 0]
    housing["price"] = housing["price"] / housing["sqft_living"]
    housing = housing[housing["price"] < 1000]
    return housing


def fill_missing_coords(housing: pd.DataFrame) -> pd.DataFrame:
    coord_cols = ["x", "y", "z"]
    missing = housing[coord_cols].isna().any(axis=1)
    means = housing.groupby("city")[coord_cols].transform("mean")

    for col in coord_cols:
        housing.loc[missing, col] = means.loc[missing, col]

    # print(f"Filled {missing.sum()} missing coords ")
    return housing


def drop_missing_coords(housing: pd.DataFrame) -> pd.DataFrame:
    return housing.dropna(subset=["x", "y", "z"])


def drop_random_coords(housing: pd.DataFrame) -> pd.DataFrame:
    housing = fill_missing_coords(housing)
    drop_idx = housing.sample(n=101, random_state=42).index
    return housing.drop(drop_idx).reset_index(drop=True)


def misc(housing: pd.DataFrame) -> pd.DataFrame:
    housing = housing.drop(columns=["country"])
    for col in ["street", "city", "statezip"]:
        housing[col] = housing[col].astype("category").cat.codes

    housing["has_basement"] = (housing["sqft_basement"] > 0).astype(int)
    housing = housing.drop(columns=["sqft_basement"])

    housing["was_renovated"] = (housing["yr_renovated"] > 0).astype(int)
    housing = housing.drop(columns=["yr_renovated"])

    housing["date"] = pd.to_datetime(housing["date"])
    housing["age"] = housing["date"].dt.year - housing["yr_built"]
    # housing = housing.drop(columns=["date", "yr_built", "x", "y", "z"])
    housing = housing.drop(columns=["date", "yr_built"])

    return housing


def main() -> None:
    housing = load_df(HOUSING_PATH)
    housing = add_cartesian(housing)
    housing = normalize_price(housing)

    housing_filled = fill_missing_coords(housing.copy())
    housing_dropped = drop_missing_coords(housing.copy())
    housing_filled_then_dropped = drop_random_coords(housing.copy())

    housing_filled = misc(housing_filled)
    housing_dropped = misc(housing_dropped)
    housing_filled_then_dropped = misc(housing_filled_then_dropped)

    housing_filled.to_csv(FILL_PATH, index=False)
    housing_dropped.to_csv(DROP_PATH, index=False)
    housing_filled_then_dropped.to_csv(FILL_THEN_DROP_PATH, index=False)


if __name__ == "__main__":
    main()
