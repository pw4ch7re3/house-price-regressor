from urllib import request, parse
import json
import time
from concurrent.futures import ThreadPoolExecutor

import os
import sys

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if path not in sys.path:
    sys.path.insert(0, path)

from data.dataload import load_df


def addr2latlong(street, city, statezip, country="USA"):
    url = "https://geocoding.geo.census.gov/geocoder/locations/address"
    params = {
        "street": street,
        "benchmark": "Public_AR_Current",
        "format": "json"
    }

    state, zip_code = statezip.strip().split()
    if city:
        params["city"] = city
    if state:
        params["state"] = state
    if zip_code:
        params["zip"] = zip_code

    r = request.Request(f"{url}?{parse.urlencode(params)}")
    with request.urlopen(r) as response:
        data = json.loads(response.read().decode('utf-8'))
        matches = data.get("result", {}).get("addressMatches", [])

        if matches:
            coords = matches[0].get("coordinates", {})
            if "y" in coords and "x" in coords:
                return (coords["y"], coords["x"]) # (latitude, longitude)

    return None


def _augment_latlong():
    df = load_df("data/raw/usa_housing_dataset.csv")
    records = df[['street', 'city', 'statezip']].to_dict('records')

    def get_coords(row):
        try:
            time.sleep(0.05)
            coords = addr2latlong(row['street'], row['city'], row['statezip'])
            return coords if coords else (None, None)
        except Exception:
            return (None, None)

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(get_coords, records))

    df['lat'] = [r[0] for r in results]
    df['long'] = [r[1] for r in results]

    df.to_csv("data/processed/latlong_added.csv", index=False)


def _null_latlong():
    df = load_df("data/processed/latlong_added.csv")
    df = df[df['lat'].isna() | df['long'].isna()]

    df.to_csv("data/raw/null_latlong.csv", index=False)


def gaussian_normalize(df, column_names):
    scaled_df = df.copy()
    for column in column_names:
        mean = df[column].mean()
        std = df[column].std()
        scaled_df[column] = (df[column] - mean) / std
    return scaled_df


if __name__ == "__main__":
    _null_latlong()