import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.assets import normalize_asset_name
from src.config.settings import load_config
from src.features.pipeline import build_processed_dataset, save_processed_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--asset",
        default=os.environ.get("DATA_PREFIX", "btc_usdt"),
        help="Asset name such as btc_usdt, BTCUSDT, or ethusdt.",
    )
    args = parser.parse_args()

    config = load_config()
    asset = normalize_asset_name(args.asset)
    dataset = build_processed_dataset(
        asset=asset,
        window_size=config["data"]["window_size"],
        train_split=config["data"]["train_split"],
    )
    save_processed_dataset(dataset)

    print(f"📊 Processed asset: {asset}")
    print("Train features:", dataset.train_windows.shape)
    print("Test features:", dataset.test_windows.shape)
    print("Feature columns:", ", ".join(dataset.metadata["features"]))


if __name__ == "__main__":
    main()
