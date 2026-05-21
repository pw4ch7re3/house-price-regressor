import re
import pandas as pd

GEOCODE_PATH = "./raw/GeocodeResults.csv"
HOUSING_PATH = "./raw/usa_housing_dataset.csv"
OUTPUT_PATH = "./raw/usa_housing_geocoded.csv"

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


def normalize(address: str) -> str:
    return re.sub(r"[\s,]+", " ", str(address)).strip().lower()


def main() -> None:
    geocode = pd.read_csv(GEOCODE_PATH, header=None, names=GEOCODE_COLS, dtype=str)
    geocode = geocode.sort_values("row_id", key=lambda s: s.astype(int))
    geocode = geocode[["input_address", "coordinate"]].reset_index(drop=True)

    housing = pd.read_csv(HOUSING_PATH)
    housing_address = (
        housing["street"] + ", " + housing["city"] + ", " + housing["statezip"]
    )

    coord_by_address = {
        normalize(addr): coord
        for addr, coord in zip(geocode["input_address"], geocode["coordinate"])
        if isinstance(coord, str) and "," in coord
    }

    coords = housing_address.map(lambda a: coord_by_address.get(normalize(a)))
    matched = coords.notna()

    lon_lat = coords[matched].str.split(",", expand=True).astype(float)
    housing["longitude"] = pd.NA
    housing["latitude"] = pd.NA
    housing.loc[matched, "longitude"] = lon_lat[0].values
    housing.loc[matched, "latitude"] = lon_lat[1].values

    housing.to_csv(OUTPUT_PATH, index=False)
    print(f"Matched {matched.sum()} / {len(housing)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
