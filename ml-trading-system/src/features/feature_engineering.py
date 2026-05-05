import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

# -------------------------------
# PATH SETUP (FIXED)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# 🔥 dynamic dataset selection
DATA_PREFIX = os.environ.get("DATA_PREFIX", "btc_usdt")
CSV_PATH = os.path.join(RAW_DIR, f"{DATA_PREFIX}_1h.csv")

print(f"📊 Processing: {DATA_PREFIX}")
print("Loading from:", CSV_PATH)

# -------------------------------
# Step 1: Load Data
# -------------------------------
df = pd.read_csv(CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("Raw data preview:")
print(df.head())

# -------------------------------
# Step 2: Feature Engineering
# -------------------------------
df["log_return"] = np.log(df["close"] / df["close"].shift(1))

df["volatility_10"] = df["log_return"].rolling(10).std()
df["volatility_20"] = df["log_return"].rolling(20).std()

df["momentum_5"] = df["close"].pct_change(5)
df["momentum_10"] = df["close"].pct_change(10)

df["ma_10"] = df["close"].rolling(10).mean()
df["ma_30"] = df["close"].rolling(30).mean()
df["trend"] = df["ma_10"] - df["ma_30"]

# RSI
delta = df["close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 1e-8)
df["rsi"] = 100 - (100 / (1 + rs))

df["body"] = df["close"] - df["open"]
df["range"] = df["high"] - df["low"]
df["body_ratio"] = df["body"] / (df["range"] + 1e-8)

df["range_pct"] = df["range"] / (df["close"] + 1e-8)

df["vol_z"] = (
    (df["volume"] - df["volume"].rolling(20).mean()) /
    (df["volume"].rolling(20).std() + 1e-8)
)

df = df.dropna().reset_index(drop=True)

# -------------------------------
# Step 3: Windowing
# -------------------------------
WINDOW_SIZE = 20

price_cols = ["open", "high", "low", "close", "volume"]
price_data = df[price_cols].values

price_windows = np.array([
    price_data[i:i + WINDOW_SIZE]
    for i in range(len(price_data) - WINDOW_SIZE + 1)
])

features = [
    "log_return",
    "volatility_10",
    "volatility_20",
    "momentum_5",
    "momentum_10",
    "trend",
    "rsi",
    "body_ratio",
    "range_pct",
    "vol_z"
]

data = df[features].values

windows = np.array([
    data[i:i + WINDOW_SIZE]
    for i in range(len(data) - WINDOW_SIZE + 1)
])

# -------------------------------
# Step 4: Train/Test Split
# -------------------------------
split_ratio = 0.8
split_idx = int(len(windows) * split_ratio)

train_windows = windows[:split_idx]
test_windows = windows[split_idx:]

train_price_windows = price_windows[:split_idx]
test_price_windows = price_windows[split_idx:]

print("\nTrain/Test Split:")
print("Train features:", train_windows.shape)
print("Train prices:", train_price_windows.shape)

# -------------------------------
# Step 5: Normalization
# -------------------------------
scaler = StandardScaler()

train_reshaped = train_windows.reshape(-1, train_windows.shape[-1])
test_reshaped = test_windows.reshape(-1, test_windows.shape[-1])

train_scaled = scaler.fit_transform(train_reshaped)
test_scaled = scaler.transform(test_reshaped)

train_windows = train_scaled.reshape(train_windows.shape)
test_windows = test_scaled.reshape(test_windows.shape)

# -------------------------------
# Step 6: Sanity Check
# -------------------------------
assert not np.isnan(train_windows).any()
assert not np.isnan(test_windows).any()

print("\nFeature stats (train):")
print("Mean:", train_windows.mean(axis=(0, 1)))
print("Std:", train_windows.std(axis=(0, 1)))

# -------------------------------
# Step 7: Save (FIXED)
# -------------------------------
meta = {
    "window_size": WINDOW_SIZE,
    "features": features,
    "split_ratio": split_ratio
}

# 🔥 CRITICAL FIX — dataset-specific files
np.save(os.path.join(PROCESSED_DIR, f"{DATA_PREFIX}_train_windows.npy"), train_windows)
np.save(os.path.join(PROCESSED_DIR, f"{DATA_PREFIX}_test_windows.npy"), test_windows)

np.save(os.path.join(PROCESSED_DIR, f"{DATA_PREFIX}_train_price_windows.npy"), train_price_windows)
np.save(os.path.join(PROCESSED_DIR, f"{DATA_PREFIX}_test_price_windows.npy"), test_price_windows)

joblib.dump(scaler, os.path.join(PROCESSED_DIR, f"{DATA_PREFIX}_scaler.pkl"))

with open(os.path.join(PROCESSED_DIR, f"{DATA_PREFIX}_meta.json"), "w") as f:
    json.dump(meta, f, indent=4)

print(f"\n✅ Saved successfully for: {DATA_PREFIX}")
print("Location:", PROCESSED_DIR)