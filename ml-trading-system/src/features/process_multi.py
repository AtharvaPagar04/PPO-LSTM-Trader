import os
import subprocess

SYMBOLS = ["btc_usdt", "ethusdt"]

for symbol in SYMBOLS:
    print(f"\n⚙️ Processing {symbol.upper()}...\n")

    env = os.environ.copy()
    env["DATA_PREFIX"] = symbol

    process = subprocess.Popen(
        ["python", "src/features/feature_engineering.py"],
        env=env
    )
    process.wait()

    if process.returncode != 0:
        print(f"❌ Failed: {symbol}")
    else:
        print(f"✅ Done: {symbol}")