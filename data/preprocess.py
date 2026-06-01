import re
import pandas as pd
import numpy as np

from dataload import load_df, HOUSING_PATH, PRICE_PATH, PRICE_PER_SQFT_PATH


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
    housing["zipcode"] = housing["statezip"].str.extract(r"(\d{5})")
    housing = housing.drop(columns=["country", "statezip", "street"])

    for col in ["city", "zipcode"]:
        housing[col] = housing[col].astype("category").cat.codes

    housing["log_sqft_lot"] = np.log1p(housing["sqft_lot"])
    housing = housing.drop(columns=["sqft_lot"])

    housing["was_renovated"] = (housing["yr_renovated"] > 0).astype(int)
    housing = housing.drop(columns=["yr_renovated"])

    housing["date"] = pd.to_datetime(housing["date"])
    housing["age"] = housing["date"].dt.year - housing["yr_built"]
    # housing = housing.drop(columns=["date", "yr_built", "x", "y", "z"])
    housing = housing.drop(columns=["date", "yr_built"])

    return housing


COLUMN_DESCRIPTIONS = {
    "price": "Target — 주택 가격(USD) / price_per_sqft 데이터셋에서는 평방피트당 가격(USD/sqft)",
    "bedrooms": "침실 수",
    "bathrooms": "욕실 수 (0.5 단위 포함)",
    "sqft_living": "실내 거주 면적 (평방피트)",
    "floors": "층 수",
    "waterfront": "해안가 접면 여부 (0=아니오, 1=예)",
    "view": "전망 품질 점수 (0~4)",
    "condition": "주택 상태 점수 (1~5)",
    "sqft_above": "지상층 면적 (평방피트)",
    "sqft_basement": "지하실 면적 (평방피트)",
    "city": "도시명 — 카테고리 인코딩 (정수)",
    "x": "위경도 → 3D 직교좌표 변환: cos(lat)·cos(lon)",
    "y": "위경도 → 3D 직교좌표 변환: cos(lat)·sin(lon)",
    "z": "위경도 → 3D 직교좌표 변환: sin(lat)",
    "zipcode": "우편번호 — 카테고리 인코딩 (정수)",
    "log_sqft_lot": "부지 면적의 log1p 변환값 (sqft_lot 대체)",
    "was_renovated": "리모델링 이력 여부 (0=없음, 1=있음)",
    "age": "판매 시점 기준 건축 연수 (판매연도 − 건축연도)",
}


def print_column_descriptions(df: pd.DataFrame) -> None:
    print("\n=== 컬럼 설명 ===")
    for col in df.columns:
        desc = COLUMN_DESCRIPTIONS.get(col, "(설명 없음)")
        print(f"  {col:<18} : {desc}")


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

    print_column_descriptions(housing_sqft)


if __name__ == "__main__":
    main()
