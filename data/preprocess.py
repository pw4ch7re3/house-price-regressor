import pandas as pd
import numpy as np

from dataload import (
    load_df,
    apply_variant,
    split_X_y,
    split_train_test,
    HOUSING_PATH,
    PRICE_PATH,
    VARIANT_PATHS,
    TARGET,
)


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


def fill_missing_coords(housing: pd.DataFrame) -> pd.DataFrame:
    housing = housing.copy()
    for col in ["x", "y", "z"]:
        city_mean = housing.groupby("city")[col].transform("mean")
        housing[col] = housing[col].fillna(city_mean)
        # fall back to global mean for cities with no valid coords
        housing[col] = housing[col].fillna(housing[col].mean())
    return housing


def drop_price_outliers(
    housing: pd.DataFrame, upper_quantile: float = 0.999
) -> pd.DataFrame:
    cap = housing["price"].quantile(upper_quantile)
    n_before = len(housing)
    dropped_idx = housing.index[housing["price"] > cap]
    housing = housing[housing["price"] <= cap]
    print(
        f"drop_price_outliers: q{upper_quantile} cap={cap:.2f} "
        f"removed={n_before - len(housing)} ({n_before} -> {len(housing)}) "
        f"max={housing['price'].max():.2f}"
    )
    print(f"  dropped index: {dropped_idx.tolist()}")
    return housing


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
    housing = housing.drop(columns=["date", "yr_built"])

    return housing


def build_variant(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, variant: str
):
    """Build one location-encoding variant from the shared split.

    Thin wrapper over ``dataload.apply_variant`` (the single source of truth) so
    the materialized variant files match the train-time CV encoding.
    """
    return apply_variant(X_train, X_test, y_train, variant)


def write_variant(variant, X_train, X_test, y_train, y_test):
    train_df = X_train.copy()
    train_df[TARGET] = y_train
    test_df = X_test.copy()
    test_df[TARGET] = y_test

    train_df.to_csv(VARIANT_PATHS[variant]["train"], index=False)
    test_df.to_csv(VARIANT_PATHS[variant]["test"], index=False)
    print(
        f"variant '{variant}': train={train_df.shape} test={test_df.shape} "
        f"cols={list(X_train.columns)}"
    )


COLUMN_DESCRIPTIONS = {
    "price": "Target — 주택 가격(USD)",
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
    # Feature engineering (unchanged), producing the canonical processed frame.
    housing = load_df(HOUSING_PATH)
    housing = add_cartesian(housing)
    housing = housing[housing["price"] > 0]
    housing = drop_price_outliers(housing)
    housing = fill_missing_coords(housing)
    housing = misc(housing)

    # Keep the unsplit engineered dataset for experiments/ensemble scripts.
    housing.to_csv(PRICE_PATH, index=False)

    # One canonical split shared by every variant (reproducible, comparable).
    X, y = split_X_y(housing, TARGET)
    (X_train, y_train), (X_test, y_test) = split_train_test(X, y)

    # Materialize each variant on its raw scale. Feature and target scaling are
    # deferred to train time (training/train.py, training/train_ensemble.py).
    for variant in VARIANT_PATHS:
        Xtr_v, Xte_v = build_variant(X_train, X_test, y_train, variant)
        write_variant(variant, Xtr_v, Xte_v, y_train, y_test)

    print_column_descriptions(housing)


if __name__ == "__main__":
    main()
