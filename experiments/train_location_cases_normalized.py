import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from typing import cast

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score as sk_r2_score

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if path not in sys.path:
    sys.path.insert(0, path)

from data.dataload import target_encode

from models import ModelConfig, TrainConfig
from models.mlp import MLPConfig, MLP
from models.decision_tree import DecisionTreeConfig, DecisionTree


output_path = "models/best_models"


LOW_CARD_ONEHOT_COLS = [
    "condition",
    "view",
    "waterfront",
    "was_renovated",
    "has_basement",
]


MINMAX_COLS = [
    "x",
    "y",
    "z",
    "lat",
    "long",
    "latitude",
    "longitude",
    "lon",
    "condition",
    "age",
    "bedrooms",
    "bathrooms",
    "floors",
    "view",
]


ZSCORE_COLS = [
    "sqft_living",
    "sqft_above",
    "sqft_basement",
    "log_sqft_lot",
    "city",
    "zipcode",
    "statezip",
]


def _existing_cols(df, cols):
    return [col for col in cols if col in df.columns]


def normalized_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_latlong_cols(df):
    lat_col = None
    lon_col = None

    if "lat" in df.columns:
        lat_col = "lat"
    elif "latitude" in df.columns:
        lat_col = "latitude"

    if "long" in df.columns:
        lon_col = "long"
    elif "longitude" in df.columns:
        lon_col = "longitude"
    elif "lon" in df.columns:
        lon_col = "lon"

    if lat_col is None or lon_col is None:
        raise ValueError(f"lat/long columns not found. Available columns: {df.columns.tolist()}")

    return lat_col, lon_col


def get_address_encode_cols(df):
    cols = []

    if "city" in df.columns:
        cols.append("city")

    if "zipcode" in df.columns:
        cols.append("zipcode")
    elif "statezip" in df.columns:
        cols.append("statezip")

    return cols


def add_xyz_from_latlong(df):
    df = df.copy()

    lat_col, lon_col = get_latlong_cols(df)

    lat_rad = np.radians(df[lat_col].astype(float))
    lon_rad = np.radians(df[lon_col].astype(float))

    df["x"] = np.cos(lat_rad) * np.cos(lon_rad)
    df["y"] = np.cos(lat_rad) * np.sin(lon_rad)
    df["z"] = np.sin(lat_rad)

    return df


def load_master_df():
    """
    모든 location case가 같은 raw 데이터에서 출발하도록 하는 master dataframe.
    raw latlong 데이터를 직접 load한 뒤,
    processed 데이터와 최대한 같은 파생 feature를 생성한다.
    """
    raw_path = "data/raw/latlong_added.csv"

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"{raw_path} does not exist.")

    df = pd.read_csv(raw_path)
    df.columns = [c.strip() for c in df.columns]

    # lat/long 확인
    get_latlong_cols(df)

    # age 생성
    if "yr_built" in df.columns and "age" not in df.columns:
        if "date" in df.columns:
            # date에서 판매연도 추출 가능하면 판매연도 - 건축연도 사용
            sale_year = pd.to_datetime(df["date"], errors="coerce").dt.year
            if sale_year.notna().sum() > 0:
                df["age"] = sale_year.fillna(sale_year.median()) - df["yr_built"]
            else:
                df["age"] = 2025 - df["yr_built"]
        else:
            df["age"] = 2025 - df["yr_built"]

    # 리모델링 여부
    if "yr_renovated" in df.columns and "was_renovated" not in df.columns:
        df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)

    # 지하실 여부
    if "sqft_basement" in df.columns and "has_basement" not in df.columns:
        df["has_basement"] = (df["sqft_basement"] > 0).astype(int)

    # lot size log 변환
    if "sqft_lot" in df.columns and "log_sqft_lot" not in df.columns:
        df["log_sqft_lot"] = np.log1p(df["sqft_lot"])

    # zipcode 추출
    if "zipcode" not in df.columns and "statezip" in df.columns:
        zipcode = df["statezip"].astype(str).str.extract(r"(\d+)")
        df["zipcode"] = pd.to_numeric(zipcode[0], errors="coerce")

    # price_per_sqft 생성
    # 단, price를 예측할 때는 아래 run_one_experiment에서 반드시 제거한다.
    if "price_per_sqft" not in df.columns:
        if "price" in df.columns and "sqft_living" in df.columns:
            df["price_per_sqft"] = df["price"] / df["sqft_living"]

    # lat/long에서 xyz 생성
    df = add_xyz_from_latlong(df)

    # 원본에서 모델에 직접 쓰지 않을 column 제거
    drop_cols = [
        "date",
        "yr_built",
        "yr_renovated",
        "sqft_lot",
    ]

    df = df.drop(columns=_existing_cols(df, drop_cols))

    # inf 처리
    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def apply_location_case(X_train, X_test, location_case):
    X_train = X_train.copy()
    X_test = X_test.copy()

    address_cols = ["street", "city", "zipcode", "statezip", "country"]
    xyz_cols = ["x", "y", "z"]
    latlong_cols = ["lat", "long", "latitude", "longitude", "lon"]

    if location_case == "none":
        drop_cols = address_cols + xyz_cols + latlong_cols

    elif location_case == "address_encoded":
        encode_cols = get_address_encode_cols(X_train)
        keep_address = set(encode_cols)

        drop_cols = []

        for col in address_cols:
            if col not in keep_address:
                drop_cols.append(col)

        drop_cols += xyz_cols + latlong_cols

    elif location_case == "xyz":
        drop_cols = address_cols + latlong_cols

    elif location_case == "latlong":
        drop_cols = address_cols + xyz_cols

    else:
        raise ValueError(f"Unknown location_case: {location_case}")

    X_train = X_train.drop(columns=_existing_cols(X_train, drop_cols))
    X_test = X_test.drop(columns=_existing_cols(X_test, drop_cols))

    return X_train, X_test


def add_frequency_encoding(X_train, X_test, cols):
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in _existing_cols(X_train, cols):
        freq = X_train[col].value_counts(normalize=True)

        X_train[f"{col}_freq"] = X_train[col].map(freq).fillna(0.0)
        X_test[f"{col}_freq"] = X_test[col].map(freq).fillna(0.0)

    return X_train, X_test


def onehot_low_cardinality(X_train, X_test, cols):
    X_train = X_train.copy()
    X_test = X_test.copy()

    cols = _existing_cols(X_train, cols)

    if len(cols) == 0:
        return X_train, X_test, []

    X_train_ohe = pd.get_dummies(X_train[cols].astype(str), prefix=cols, dtype=float)
    X_test_ohe = pd.get_dummies(X_test[cols].astype(str), prefix=cols, dtype=float)

    X_test_ohe = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0.0)

    X_train = pd.concat([X_train.drop(columns=cols), X_train_ohe], axis=1)
    X_test = pd.concat([X_test.drop(columns=cols), X_test_ohe], axis=1)

    return X_train, X_test, list(X_train_ohe.columns)


def add_train_based_age_bin(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    if "age" not in X_train.columns:
        return X_train, X_test

    X_train["age_bin"], age_bins = pd.cut(
        X_train["age"],
        bins=5,
        labels=False,
        retbins=True,
        duplicates="drop",
    )

    X_test["age_bin"] = pd.cut(
        X_test["age"],
        bins=age_bins,
        labels=False,
        include_lowest=True,
    )

    X_train["age_bin"] = X_train["age_bin"].astype(float)
    X_test["age_bin"] = X_test["age_bin"].astype(float)

    X_test["age_bin"] = X_test["age_bin"].fillna(X_train["age_bin"].median())

    return X_train, X_test


def clean_numeric(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    obj_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    if len(obj_cols) > 0:
        X_train_ohe = pd.get_dummies(X_train[obj_cols].astype(str), prefix=obj_cols, dtype=float)
        X_test_ohe = pd.get_dummies(X_test[obj_cols].astype(str), prefix=obj_cols, dtype=float)

        X_test_ohe = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0.0)

        X_train = pd.concat([X_train.drop(columns=obj_cols), X_train_ohe], axis=1)
        X_test = pd.concat([X_test.drop(columns=obj_cols), X_test_ohe], axis=1)

    X_train = X_train.fillna(0.0)
    X_test = X_test.fillna(0.0)

    return X_train, X_test


def preprocess_for_mlp(X_train, y_train, X_test, location_case):
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train, X_test = apply_location_case(X_train, X_test, location_case)

    if location_case == "address_encoded":
        address_encode_cols = get_address_encode_cols(X_train)

        X_train, X_test = add_frequency_encoding(X_train, X_test, address_encode_cols)

        for col in _existing_cols(X_train, address_encode_cols):
            X_train[col], X_test[col] = target_encode(X_train, y_train, X_test, col)

    X_train, X_test, ohe_cols = onehot_low_cardinality(
        X_train, X_test, LOW_CARD_ONEHOT_COLS
    )

    minmax_cols = _existing_cols(X_train, MINMAX_COLS)

    freq_cols = [col for col in X_train.columns if col.endswith("_freq")]
    minmax_cols = list(dict.fromkeys(minmax_cols + freq_cols))

    if len(minmax_cols) > 0:
        scaler_mm = MinMaxScaler()
        X_train[minmax_cols] = scaler_mm.fit_transform(X_train[minmax_cols])
        X_test[minmax_cols] = scaler_mm.transform(X_test[minmax_cols])

    zscore_cols = _existing_cols(X_train, ZSCORE_COLS)
    zscore_cols = [col for col in zscore_cols if col not in ohe_cols]

    if len(zscore_cols) > 0:
        scaler = StandardScaler()
        X_train[zscore_cols] = scaler.fit_transform(X_train[zscore_cols])
        X_test[zscore_cols] = scaler.transform(X_test[zscore_cols])

    X_train, X_test = clean_numeric(X_train, X_test)

    return X_train, X_test


def preprocess_for_dt(X_train, y_train, X_test, location_case):
    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train, X_test = apply_location_case(X_train, X_test, location_case)

    if location_case == "address_encoded":
        address_encode_cols = get_address_encode_cols(X_train)

        X_train, X_test = add_frequency_encoding(X_train, X_test, address_encode_cols)

        for col in _existing_cols(X_train, address_encode_cols):
            X_train[col], X_test[col] = target_encode(X_train, y_train, X_test, col)

    X_train, X_test = add_train_based_age_bin(X_train, X_test)

    X_train, X_test = clean_numeric(X_train, X_test)

    return X_train, X_test


def train(model_config: ModelConfig, train_config: TrainConfig, target: str, location_case: str):
    model_name = model_config.model.lower()

    if model_name == "mlp":
        model = MLP(cast(MLPConfig, model_config))
    elif model_name in ("decision_tree", "decision tree", "dt"):
        model = DecisionTree(cast(DecisionTreeConfig, model_config))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(train_config)

    save_dir = os.path.join(output_path, location_case)
    os.makedirs(save_dir, exist_ok=True)

    torch.save(
        model.state_dict(),
        os.path.join(save_dir, f"best_{model_name}_{target}.pth")
    )

    return model


def make_target_scaled(y_train, y_test):
    """
    target을 train 기준으로 0~1 MinMax 정규화.
    이 덕분에 normalized RMSE를 0~1 scale에서 비교할 수 있다.
    """
    scaler = MinMaxScaler()

    y_train_arr = np.asarray(y_train).reshape(-1, 1)
    y_test_arr = np.asarray(y_test).reshape(-1, 1)

    y_train_scaled = scaler.fit_transform(y_train_arr).reshape(-1)
    y_test_scaled = scaler.transform(y_test_arr).reshape(-1)

    y_train_scaled = pd.Series(y_train_scaled, index=y_train.index)
    y_test_scaled = pd.Series(y_test_scaled, index=y_test.index)

    return y_train_scaled, y_test_scaled, scaler


def inverse_target(y_scaled, scaler):
    y_arr = np.asarray(y_scaled).reshape(-1, 1)
    return scaler.inverse_transform(y_arr).reshape(-1)


def run_one_experiment(args, target_col):
    df = load_master_df()

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' does not exist.")

    # target leakage 방지
    # price 예측 시 price_per_sqft는 절대 feature에 들어가면 안 됨.
    # price_per_sqft 예측 시 price도 feature에 들어가면 안 됨.
    if target_col == "price":
        df = df.drop(columns=_existing_cols(df, ["price_per_sqft"]))
    elif target_col == "price_per_sqft":
        df = df.drop(columns=_existing_cols(df, ["price"]))

    y = df[target_col]
    X = df.drop(columns=[target_col])

    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )

    # target 0~1 정규화
    y_train, y_test, y_scaler = make_target_scaled(y_train_raw, y_test_raw)

    if args.model == "mlp":
        X_train, X_test = preprocess_for_mlp(X_train, y_train, X_test, args.location_case)

        model_config = MLPConfig(
            model="mlp",
            input_dim=X_train.shape[1],
            hidden_dims=[32, 32],
            output_dim=1,
        )

    elif args.model == "dt":
        X_train, X_test = preprocess_for_dt(X_train, y_train, X_test, args.location_case)

        model_config = DecisionTreeConfig(
            model="dt",
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")

    print("=" * 100)
    print(f"model: {args.model}")
    print(f"target: {target_col}")
    print(f"location_case: {args.location_case}")
    print(f"data_path: data/raw/latlong_added.csv")
    print(f"target_transform: minmax_0_1")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print("used columns:")
    print(X_train.columns.tolist())
    print("=" * 100)

    train_config = TrainConfig(
        X=X_train,
        y=y_train,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    model = train(model_config, train_config, target_col, args.location_case)

    # normalized scale prediction
    y_train_pred_norm = model.predict(X_train)
    y_test_pred_norm = model.predict(X_test)

    y_train_pred_norm = np.asarray(y_train_pred_norm).reshape(-1)
    y_test_pred_norm = np.asarray(y_test_pred_norm).reshape(-1)

    # normalized metric
    train_rmse_norm = normalized_rmse(y_train, y_train_pred_norm)
    test_rmse_norm = normalized_rmse(y_test, y_test_pred_norm)

    train_r2_norm = sk_r2_score(y_train, y_train_pred_norm)
    test_r2_norm = sk_r2_score(y_test, y_test_pred_norm)

    # original scale metric
    y_train_pred_original = inverse_target(y_train_pred_norm, y_scaler)
    y_test_pred_original = inverse_target(y_test_pred_norm, y_scaler)

    y_train_original = np.asarray(y_train_raw).reshape(-1)
    y_test_original = np.asarray(y_test_raw).reshape(-1)

    train_rmse_original = normalized_rmse(y_train_original, y_train_pred_original)
    test_rmse_original = normalized_rmse(y_test_original, y_test_pred_original)

    train_r2_original = sk_r2_score(y_train_original, y_train_pred_original)
    test_r2_original = sk_r2_score(y_test_original, y_test_pred_original)

    print(f"Train Normalized RMSE: {train_rmse_norm:.6f}")
    print(f"Train Normalized R2:   {train_r2_norm:.6f}")
    print(f"Test  Normalized RMSE: {test_rmse_norm:.6f}")
    print(f"Test  Normalized R2:   {test_r2_norm:.6f}")

    print()
    print(f"Train Original RMSE:   {train_rmse_original:.4f}")
    print(f"Train Original R2:     {train_r2_original:.6f}")
    print(f"Test  Original RMSE:   {test_rmse_original:.4f}")
    print(f"Test  Original R2:     {test_r2_original:.6f}")

    result = {
        "model": args.model,
        "target": target_col,
        "location_case": args.location_case,
        "data_path": "data/raw/latlong_added.csv",
        "target_transform": "minmax_0_1",
        "n_features": X_train.shape[1],
        "train_rmse_norm": train_rmse_norm,
        "train_r2_norm": train_r2_norm,
        "test_rmse_norm": test_rmse_norm,
        "test_r2_norm": test_r2_norm,
        "train_rmse_original": train_rmse_original,
        "train_r2_original": train_r2_original,
        "test_rmse_original": test_rmse_original,
        "test_r2_original": test_r2_original,
        "used_columns": "|".join(X_train.columns.tolist()),
    }

    os.makedirs("experiments/location_cases_normalized", exist_ok=True)
    result_path = "experiments/location_cases_normalized/results.csv"

    result_df = pd.DataFrame([result])

    if os.path.exists(result_path):
        old_df = pd.read_csv(result_path)
        result_df = pd.concat([old_df, result_df], ignore_index=True)

    result_df.to_csv(result_path, index=False)


def main(args):
    if args.target == "both":
        run_one_experiment(args, "price")
        run_one_experiment(args, "price_per_sqft")
    else:
        run_one_experiment(args, args.target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Location case experiment with normalized target RMSE")

    parser.add_argument(
        "--target",
        type=str,
        default="price",
        choices=["price", "price_per_sqft", "both"],
    )

    parser.add_argument(
        "--location_case",
        type=str,
        default="none",
        choices=["none", "address_encoded", "xyz", "latlong"],
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "dt"],
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=2)

    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    main(args)