import os
import subprocess

# 🔥 datasets to train on
SYMBOLS = ["btc_usdt", "ethusdt"]

# paths
DATA_DIR = "data/raw"
LOG_DIR = "logs"
MODEL_DIR = "models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

for symbol in SYMBOLS:
    print(f"\n🚀 Training on {symbol.upper()}...\n")

    data_path = f"{DATA_DIR}/{symbol}_1h.csv"
    log_path = f"{LOG_DIR}/{symbol}.log"

    # pass env variable so train.py can pick correct dataset
    env = os.environ.copy()
    env["DATA_PREFIX"] = symbol
    env["MODEL_PATH"] = f"{MODEL_DIR}/{symbol}_model.pth"

    with open(log_path, "w") as f:
        process = subprocess.Popen(
            ["python", "-m", "src.train"],
            stdout=f,
            stderr=f,
            env=env
        )
        process.wait()

    print(f"✅ Done {symbol.upper()} | Log saved: {log_path}")