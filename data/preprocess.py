from urllib import request, parse
import json

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


if __name__ == "__main__":
    lat, long = addr2latlong("9245-9249 Fremont Ave N", "Seattle", "WA 98103")
    print(lat, long)

    lat, long = addr2latlong("33001 NE 24th St", "Carnation", "WA 98014")
    print(lat, long)