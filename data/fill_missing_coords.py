import pandas as pd

INPUT_PATH = "./raw/usa_housing_geocoded.csv"
OUTPUT_PATH = "./raw/usa_housing_geocoded.csv"


def main() -> None:
    housing = pd.read_csv(INPUT_PATH)

    missing = housing["longitude"].isna() | housing["latitude"].isna()
    means = housing.groupby("statezip")[["longitude", "latitude"]].transform("mean")

    housing.loc[missing, "longitude"] = means.loc[missing, "longitude"]
    housing.loc[missing, "latitude"] = means.loc[missing, "latitude"]

    still_missing = housing["longitude"].isna() | housing["latitude"].isna()
    housing.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Filled {missing.sum() - still_missing.sum()} / {missing.sum()} missing rows "
        f"({still_missing.sum()} statezip groups had no coords) -> {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
