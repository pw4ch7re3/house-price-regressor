import re
import pandas as pd
import numpy as np

from dataload import load_df, HOUSING_PATH, PRICE_PATH, PRICE_PER_SQFT_PATH
from sklearn.preprocessing import StandardScaler


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


def drop_missing_coords(housing: pd.DataFrame) -> pd.DataFrame:
    return housing.dropna(subset=["x", "y", "z"])


def misc(housing: pd.DataFrame) -> pd.DataFrame:
    housing = housing.drop(columns=["country", "street", "city", "statezip"])
    # for col in ["street", "city", "statezip"]:
    #     housing[col] = housing[col].astype("category").cat.codes

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
    housing_sqft = normalize_price(housing.copy())

    housing = drop_missing_coords(housing)
    housing_sqft = drop_missing_coords(housing_sqft)

    housing = misc(housing)
    housing_sqft = misc(housing_sqft)

    housing.to_csv(PRICE_PATH, index=False)
    housing_sqft.to_csv(PRICE_PER_SQFT_PATH, index=False)


if __name__ == "__main__":
    main()
