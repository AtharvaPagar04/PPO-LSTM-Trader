import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.assets import SUPPORTED_ASSETS


for asset in SUPPORTED_ASSETS:
    print(f"\n⚙️ Processing {asset.upper()}...\n")
    env = os.environ.copy()
    env["DATA_PREFIX"] = asset
    process = subprocess.Popen(
        ["python", "src/features/feature_engineering.py", "--asset", asset],
        env=env,
    )
    process.wait()

    if process.returncode != 0:
        print(f"❌ Failed: {asset}")
    else:
        print(f"✅ Done: {asset}")
