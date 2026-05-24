import re
import pandas as pd
import numpy as np

GEOCODE_PATH = "./raw/GeocodeResults.csv"
HOUSING_PATH = "./raw/usa_housing_dataset.csv"
OUTPUT_PATH = "./raw/usa_housing_dataset_processed.csv"

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
    housing["longitude"] = pd.NA
    housing["latitude"] = pd.NA
    housing.loc[matched, "longitude"] = lon_lat[0].values
    housing.loc[matched, "latitude"] = lon_lat[1].values

    print(f"Geocoded {matched.sum()} / {len(housing)} rows")
    return housing


def fill_missing_coords(housing: pd.DataFrame) -> pd.DataFrame:
    missing = housing["longitude"].isna() | housing["latitude"].isna()
    means = housing.groupby("statezip")[["longitude", "latitude"]].transform("mean")

    housing.loc[missing, "longitude"] = means.loc[missing, "longitude"]
    housing.loc[missing, "latitude"] = means.loc[missing, "latitude"]

    still_missing = housing["longitude"].isna() | housing["latitude"].isna()
    print(
        f"Filled {missing.sum() - still_missing.sum()} / {missing.sum()} missing coords "
        f"({still_missing.sum()} statezip groups had no coords)"
    )
    return housing


def normalize_price(housing: pd.DataFrame) -> pd.DataFrame:
    housing = housing[housing["price"] > 0]
    # housing["price"] = housing["price"] / housing["sqft_living"]
    housing["price"] = np.log1p(housing["price"] / housing["sqft_living"])
    return housing

def misc(housing: pd.DataFrame) -> pd.DataFrame:
    housing = housing.drop(columns=["street", "city", "statezip", "country"])

    housing["has_basement"] = (housing["sqft_basement"] > 0).astype(int)
    housing = housing.drop(columns=["sqft_basement"])

    housing["was_renovated"] = (housing["yr_renovated"] > 0).astype(int)
    housing = housing.drop(columns=["yr_renovated"])

    housing["date"] = pd.to_datetime(housing["date"])
    housing["age"] = housing["date"].dt.year - housing["yr_built"]
    housing = housing.drop(columns=["date", "yr_built"])

    return housing


def main() -> None:
    housing = pd.read_csv(HOUSING_PATH)
    housing = geocode_bulk(housing)
    housing = fill_missing_coords(housing)
    housing = normalize_price(housing)
    housing = misc(housing)
    housing.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
