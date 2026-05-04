import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

DATA_PATH = "data/raw/btc_usdt_1h.csv"
OUTPUT_DIR = "data"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def create_features(df):
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = df["log_return"].rolling(10).std()
    df["body"] = df["close"] - df["open"]
    df["wick"] = df["high"] - df["low"]

    df["momentum_5"] = df["close"].pct_change(5)
    df["range"] = (df["high"] - df["low"]) / df["close"]

    df["vol_z"] = (
        (df["volume"] - df["volume"].rolling(20).mean()) /
        (df["volume"].rolling(20).std() + 1e-8)
    )

    df["body_ratio"] = (
        (df["close"] - df["open"]) /
        (df["high"] - df["low"] + 1e-8)
    )

    return df.dropna().reset_index(drop=True)


def create_windows(data, window_size=20):
    windows = []
    for i in range(len(data) - window_size + 1):
        windows.append(data[i:i + window_size])
    return np.array(windows)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_data()
    df = create_features(df)

    WINDOW_SIZE = 20

    # -------------------------
    # Price windows (RAW)
    # -------------------------
    price_cols = ["open", "high", "low", "close", "volume"]
    price_data = df[price_cols].values
    price_windows = create_windows(price_data, WINDOW_SIZE)

    # -------------------------
    # Feature windows
    # -------------------------
    features = [
        "log_return",
        "volatility",
        "body",
        "wick",
        "momentum_5",
        "range",
        "vol_z",
        "body_ratio"
    ]

    feature_data = df[features].values
    windows = create_windows(feature_data, WINDOW_SIZE)

    # -------------------------
    # Train/Test split
    # -------------------------
    split_idx = int(len(windows) * 0.8)

    train_windows = windows[:split_idx]
    test_windows = windows[split_idx:]

    train_price = price_windows[:split_idx]
    test_price = price_windows[split_idx:]

    # -------------------------
    # Scaling (ONLY FEATURES)
    # -------------------------
    scaler = StandardScaler()

    train_reshaped = train_windows.reshape(-1, train_windows.shape[-1])
    test_reshaped = test_windows.reshape(-1, test_windows.shape[-1])

    train_scaled = scaler.fit_transform(train_reshaped)
    test_scaled = scaler.transform(test_reshaped)

    train_windows = train_scaled.reshape(train_windows.shape)
    test_windows = test_scaled.reshape(test_windows.shape)

    # -------------------------
    # Save
    # -------------------------
    np.save(f"{OUTPUT_DIR}/train_windows.npy", train_windows)
    np.save(f"{OUTPUT_DIR}/test_windows.npy", test_windows)

    np.save(f"{OUTPUT_DIR}/train_price_windows.npy", train_price)
    np.save(f"{OUTPUT_DIR}/test_price_windows.npy", test_price)

    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")

    meta = {
        "window_size": WINDOW_SIZE,
        "features": features
    }

    with open(f"{OUTPUT_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    print("✅ Data preprocessing complete")
    print("Train shape:", train_windows.shape)
    print("Test shape:", test_windows.shape)


if __name__ == "__main__":
    main()