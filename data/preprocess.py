import re
import pandas as pd
import numpy as np

GEOCODE_PATH = "./raw/GeocodeResults.csv"
HOUSING_PATH = "./raw/usa_housing_dataset.csv"
OUTPUT_PATH = "./processed/usa_housing_dataset_processed.csv"

GEOCODE_COLS = [
    "row_id",
    "input_address",
    "match",
    "match_type",
    "matched_address",
    "coordinate",
    "tigerline_id",
    "side",
]

def latlong2cartesian(lat, long):
    lat_rad = np.radians(lat)
    long_rad = np.radians(long)

    x = np.cos(lat_rad) * np.cos(long_rad)
    y = np.cos(lat_rad) * np.sin(long_rad)
    z = np.sin(lat_rad)
    return x, y, z

def normalize_address(address: str) -> str:
    return re.sub(r"[\s,]+", " ", str(address)).strip().lower()


def geocode_bulk(housing: pd.DataFrame) -> pd.DataFrame:
    geocode = pd.read_csv(GEOCODE_PATH, header=None, names=GEOCODE_COLS, dtype=str)
    geocode = geocode.sort_values("row_id", key=lambda s: s.astype(int))
    geocode = geocode[["input_address", "coordinate"]].reset_index(drop=True)

    housing_address = (
        housing["street"] + ", " + housing["city"] + ", " + housing["statezip"]
    )

    coord_by_address = {
        normalize_address(addr): coord
        for addr, coord in zip(geocode["input_address"], geocode["coordinate"])
        if isinstance(coord, str) and "," in coord
    }

    coords = housing_address.map(lambda a: coord_by_address.get(normalize_address(a)))
    matched = coords.notna()

    lon_lat = coords[matched].str.split(",", expand=True).astype(float)
    x, y, z = latlong2cartesian(lon_lat[1].values, lon_lat[0].values)
    housing["x"] = pd.NA
    housing["y"] = pd.NA
    housing["z"] = pd.NA
    housing.loc[matched, "x"] = x
    housing.loc[matched, "y"] = y
    housing.loc[matched, "z"] = z

    print(f"Geocoded {matched.sum()} / {len(housing)} rows")
    return housing


def fill_missing_coords(housing: pd.DataFrame) -> pd.DataFrame:
    coord_cols = ["x", "y", "z"]
    missing = housing[coord_cols].isna().any(axis=1)
    means = housing.groupby("city")[coord_cols].transform("mean")

    for col in coord_cols:
        housing.loc[missing, col] = means.loc[missing, col]

    still_missing = housing[coord_cols].isna().any(axis=1)
    print(
        f"Filled {missing.sum() - still_missing.sum()} / {missing.sum()} missing coords "
        f"({still_missing.sum()} city groups had no coords)"
    )
    return housing

def drop_missing_coords(housing: pd.DataFrame) -> pd.DataFrame:
    return housing.dropna(subset=["x", "y", "z"])

def drop_random_coords(housing: pd.DataFrame) -> pd.DataFrame:
    housing = fill_missing_coords(housing)
    drop_idx = housing.sample(n=101, random_state=42).index
    return housing.drop(drop_idx).reset_index(drop=True)

def normalize_price(housing: pd.DataFrame) -> pd.DataFrame:
    housing = housing[housing["price"] > 0]
    # housing["price"] = housing["price"] / housing["sqft_living"]
    housing["price"] = np.log1p(housing["price"] / housing["sqft_living"])
    return housing

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
    housing = pd.read_csv(HOUSING_PATH)
    housing = geocode_bulk(housing)
    housing = fill_missing_coords(housing)
    # housing = drop_missing_coords(housing)
    # housing = drop_random_coords(housing)
    housing = normalize_price(housing)
    housing = misc(housing)
    housing.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
