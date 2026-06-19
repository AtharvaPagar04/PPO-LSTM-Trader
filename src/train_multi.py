import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.assets import SUPPORTED_ASSETS
from src.config.paths import LOGS_DIR, ensure_dir


ensure_dir(LOGS_DIR)
for asset in SUPPORTED_ASSETS:
    print(f"\n🚀 Training on {asset.upper()}...\n")
    log_path = LOGS_DIR / f"{asset}.log"
    env = os.environ.copy()
    env["DATA_PREFIX"] = asset

    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            ["python", "-m", "src.train", "--asset", asset],
            stdout=handle,
            stderr=handle,
            env=env,
        )
        process.wait()

    print(f"✅ Done {asset.upper()} | Log saved: {log_path}")
