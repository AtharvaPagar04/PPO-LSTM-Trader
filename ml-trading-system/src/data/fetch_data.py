import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from binance.client import Client
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.assets import asset_to_symbol, normalize_asset_name, symbol_to_asset
from src.config.paths import RAW_DIR, ensure_dir


client = Client()


def fetch_ohlcv(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1HOUR, start_str="1 Jan 2022"):
    print(f"Fetching {symbol} data with progress...")

    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp.now().timestamp() * 1000)

    all_klines = []
    step = 1000
    total_steps = (end_ts - start_ts) // (60 * 60 * 1000)
    pbar = tqdm(total=total_steps)

    while start_ts < end_ts:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=step,
            startTime=start_ts,
        )

        if not klines:
            break

        all_klines.extend(klines)
        start_ts = klines[-1][0] + 1
        pbar.update(len(klines))
        time.sleep(0.2)

    pbar.close()

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "num_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]

    df = pd.DataFrame(all_klines, columns=columns)[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ]
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.sort_values("timestamp").dropna().reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--asset", type=str, default=None)
    args = parser.parse_args()

    if args.asset:
        asset = normalize_asset_name(args.asset)
        symbol = asset_to_symbol(asset)
    else:
        symbol = args.symbol.upper() if args.symbol else "BTCUSDT"
        asset = symbol_to_asset(symbol)

    print("Starting data fetch...")
    print(f"Asset: {asset}")
    print(f"Symbol: {symbol}")

    df = fetch_ohlcv(symbol=symbol)
    print("Data fetched:", df.shape)
    print(df.head())

    ensure_dir(RAW_DIR)
    output_path = RAW_DIR / f"{asset}_1h.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Saved to {output_path}")


if __name__ == "__main__":
    main()
