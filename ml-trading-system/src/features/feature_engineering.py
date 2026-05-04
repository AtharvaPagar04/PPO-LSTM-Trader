import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import json

# -------------------------------
# Step 1: Load and Prepare Data
# -------------------------------
df = pd.read_csv("../data/btc_usdt_1h.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

print("Raw data preview:")
print(df.head())

# -------------------------------
# Step 2: Feature Engineering
# -------------------------------

# Core features
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df["volatility"] = df["log_return"].rolling(10).std()
df["body"] = df["close"] - df["open"]
df["wick"] = df["high"] - df["low"]

# Additional features
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

# Drop NaNs
df = df.dropna().reset_index(drop=True)

# -------------------------------
# Step 3: RAW PRICE WINDOWS (IMPORTANT)
# -------------------------------
WINDOW_SIZE = 20

price_cols = ["open", "high", "low", "close", "volume"]
price_data = df[price_cols].values

price_windows = []
for i in range(len(price_data) - WINDOW_SIZE + 1):
    price_windows.append(price_data[i:i + WINDOW_SIZE])

price_windows = np.array(price_windows)

# -------------------------------
# Step 4: FEATURE WINDOWS (ML)
# -------------------------------
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

data = df[features].values

windows = []
for i in range(len(data) - WINDOW_SIZE + 1):
    windows.append(data[i:i + WINDOW_SIZE])

windows = np.array(windows)

# -------------------------------
# Step 5: Train/Test Split
# -------------------------------
split_ratio = 0.8
split_idx = int(len(windows) * split_ratio)

# Feature windows
train_windows = windows[:split_idx]
test_windows = windows[split_idx:]

# Price windows (aligned)
train_price_windows = price_windows[:split_idx]
test_price_windows = price_windows[split_idx:]

print("\nTrain/Test Split:")
print("Train features:", train_windows.shape)
print("Train prices:", train_price_windows.shape)

# -------------------------------
# Step 6: Normalize FEATURES ONLY
# -------------------------------
scaler = StandardScaler()

train_reshaped = train_windows.reshape(-1, train_windows.shape[-1])
test_reshaped = test_windows.reshape(-1, test_windows.shape[-1])

train_scaled = scaler.fit_transform(train_reshaped)
test_scaled = scaler.transform(test_reshaped)

train_windows = train_scaled.reshape(train_windows.shape)
test_windows = test_scaled.reshape(test_windows.shape)

# -------------------------------
# Step 7: Sanity Checks
# -------------------------------
assert not np.isnan(train_windows).any(), "NaNs in train windows!"
assert not np.isnan(test_windows).any(), "NaNs in test windows!"

print("\nFinal Output:")
print("Train windows shape:", train_windows.shape)
print("Example feature window:\n", train_windows[0])

print("\nFeature stats (train only):")
print("Mean:", train_windows.mean(axis=(0, 1)))
print("Std:", train_windows.std(axis=(0, 1)))

# -------------------------------
# Step 8: Save Outputs
# -------------------------------

# Feature windows (for ML)
np.save("../data/train_windows.npy", train_windows)
np.save("../data/test_windows.npy", test_windows)

# RAW price windows (for returns)
np.save("../data/train_price_windows.npy", train_price_windows)
np.save("../data/test_price_windows.npy", test_price_windows)

# Optional
np.save("../data/features.npy", data)

# Save scaler
joblib.dump(scaler, "../data/scaler.pkl")

# Metadata
meta = {
    "window_size": WINDOW_SIZE,
    "features": features,
    "split_ratio": split_ratio
}

with open("../data/meta.json", "w") as f:
    json.dump(meta, f, indent=4)

print("\nSaved:")
print("- train_windows.npy (features)")
print("- train_price_windows.npy (RAW prices)")
print("- scaler.pkl")
print("- meta.json")