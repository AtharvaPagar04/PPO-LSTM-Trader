from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config.assets import normalize_asset_name
from src.config.paths import ensure_dir, PROCESSED_DIR, resolve_existing_raw_data_path


FEATURE_COLUMNS = [
    "log_return",
    "volatility_10",
    "volatility_20",
    "momentum_5",
    "momentum_10",
    "trend",
    "rsi",
    "body_ratio",
    "range_pct",
    "vol_z",
]

REGIME_FEATURE_COLUMNS = [
    "return_24",
    "return_72",
    "volatility_24",
    "volatility_72",
    "ma_ratio_24_72",
    "ma_slope_24",
    "ma_slope_72",
    "trend_strength_24",
    "volatility_regime",
    "rsi_24",
    "drawdown_from_rolling_high_72",
]

ALL_FEATURE_COLUMNS = FEATURE_COLUMNS + REGIME_FEATURE_COLUMNS

PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class ProcessedDataset:
    asset: str
    train_windows: np.ndarray
    test_windows: np.ndarray
    train_price_windows: np.ndarray
    test_price_windows: np.ndarray
    scaler: StandardScaler
    metadata: dict


def resolve_selected_features(selected_features: list[str] | None = None) -> list[str]:
    if selected_features is None:
        return list(FEATURE_COLUMNS)
    if not selected_features:
        raise ValueError("Selected feature list must be non-empty.")
    unknown = [feature for feature in selected_features if feature not in ALL_FEATURE_COLUMNS]
    if unknown:
        raise ValueError(
            "Unknown selected features: " + ", ".join(unknown)
        )
    return list(selected_features)


def load_raw_dataframe(asset: str) -> pd.DataFrame:
    df = pd.read_csv(resolve_existing_raw_data_path(asset))
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("timestamp").dropna().reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    features["log_return"] = np.log(features["close"] / features["close"].shift(1))
    features["volatility_10"] = features["log_return"].rolling(10).std()
    features["volatility_20"] = features["log_return"].rolling(20).std()
    features["volatility_24"] = features["log_return"].rolling(24).std()
    features["volatility_72"] = features["log_return"].rolling(72).std()
    features["momentum_5"] = features["close"].pct_change(5)
    features["momentum_10"] = features["close"].pct_change(10)
    features["return_24"] = np.log(features["close"] / features["close"].shift(24))
    features["return_72"] = np.log(features["close"] / features["close"].shift(72))
    features["ma_10"] = features["close"].rolling(10).mean()
    features["ma_24"] = features["close"].rolling(24).mean()
    features["ma_30"] = features["close"].rolling(30).mean()
    features["ma_72"] = features["close"].rolling(72).mean()
    features["trend"] = features["ma_10"] - features["ma_30"]
    features["ma_ratio_24_72"] = (
        features["ma_24"] / (features["ma_72"] + 1e-8)
    ) - 1.0
    features["ma_slope_24"] = np.log(
        (features["ma_24"] + 1e-8) / (features["ma_24"].shift(6) + 1e-8)
    )
    features["ma_slope_72"] = np.log(
        (features["ma_72"] + 1e-8) / (features["ma_72"].shift(6) + 1e-8)
    )

    delta = features["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    features["rsi"] = 100 - (100 / (1 + rs))
    gain_24 = delta.where(delta > 0, 0.0).rolling(24).mean()
    loss_24 = (-delta.where(delta < 0, 0.0)).rolling(24).mean()
    rs_24 = gain_24 / (loss_24 + 1e-8)
    features["rsi_24"] = 100 - (100 / (1 + rs_24))

    features["body"] = features["close"] - features["open"]
    features["range"] = features["high"] - features["low"]
    features["body_ratio"] = features["body"] / (features["range"] + 1e-8)
    features["range_pct"] = features["range"] / (features["close"] + 1e-8)
    features["vol_z"] = (
        (features["volume"] - features["volume"].rolling(20).mean())
        / (features["volume"].rolling(20).std() + 1e-8)
    )
    features["trend_strength_24"] = (
        features["return_24"].abs() / (features["volatility_24"] + 1e-8)
    )
    features["volatility_regime"] = (
        features["volatility_24"] / (features["volatility_72"] + 1e-8)
    )
    rolling_high_72 = features["high"].rolling(72).max()
    features["drawdown_from_rolling_high_72"] = (
        features["close"] / (rolling_high_72 + 1e-8)
    ) - 1.0

    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna().reset_index(drop=True)


def create_windows(data: np.ndarray, window_size: int) -> np.ndarray:
    if len(data) < window_size:
        raise ValueError(
            f"Cannot create windows: dataset length {len(data)} < window size {window_size}."
        )
    return np.array(
        [data[i : i + window_size] for i in range(len(data) - window_size + 1)]
    )


def split_windows(
    windows: np.ndarray, price_windows: np.ndarray, train_split: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split_idx = int(len(windows) * train_split)
    return (
        windows[:split_idx],
        windows[split_idx:],
        price_windows[:split_idx],
        price_windows[split_idx:],
    )


def build_processed_dataset(
    asset: str, window_size: int, train_split: float, selected_features: list[str] | None = None
) -> ProcessedDataset:
    asset = normalize_asset_name(asset)
    df = engineer_features(load_raw_dataframe(asset))
    selected_features = resolve_selected_features(selected_features)

    price_data = df[PRICE_COLUMNS].to_numpy(dtype=np.float32)
    feature_data = df[selected_features].to_numpy(dtype=np.float32)

    price_windows = create_windows(price_data, window_size)
    windows = create_windows(feature_data, window_size)

    train_windows, test_windows, train_price_windows, test_price_windows = split_windows(
        windows, price_windows, train_split
    )

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(
        train_windows.reshape(-1, train_windows.shape[-1])
    ).reshape(train_windows.shape)
    test_scaled = scaler.transform(
        test_windows.reshape(-1, test_windows.shape[-1])
    ).reshape(test_windows.shape)

    metadata = {
        "asset": asset,
        "window_size": window_size,
        "features": selected_features,
        "price_columns": PRICE_COLUMNS,
        "split_ratio": train_split,
        "train_rows": int(len(train_scaled)),
        "test_rows": int(len(test_scaled)),
        "start_timestamp": df["timestamp"].min().isoformat(),
        "end_timestamp": df["timestamp"].max().isoformat(),
        "train_end_timestamp": df.iloc[len(train_scaled) + window_size - 2][
            "timestamp"
        ].isoformat(),
        "test_start_timestamp": df.iloc[len(train_scaled)]["timestamp"].isoformat(),
    }

    return ProcessedDataset(
        asset=asset,
        train_windows=train_scaled.astype(np.float32),
        test_windows=test_scaled.astype(np.float32),
        train_price_windows=train_price_windows.astype(np.float32),
        test_price_windows=test_price_windows.astype(np.float32),
        scaler=scaler,
        metadata=metadata,
    )


def save_processed_dataset(dataset: ProcessedDataset) -> None:
    ensure_dir(PROCESSED_DIR)
    asset = dataset.asset
    np.save(PROCESSED_DIR / f"{asset}_train_windows.npy", dataset.train_windows)
    np.save(PROCESSED_DIR / f"{asset}_test_windows.npy", dataset.test_windows)
    np.save(
        PROCESSED_DIR / f"{asset}_train_price_windows.npy",
        dataset.train_price_windows,
    )
    np.save(
        PROCESSED_DIR / f"{asset}_test_price_windows.npy",
        dataset.test_price_windows,
    )
    joblib.dump(dataset.scaler, PROCESSED_DIR / f"{asset}_scaler.pkl")
    with (PROCESSED_DIR / f"{asset}_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset.metadata, handle, indent=2)
