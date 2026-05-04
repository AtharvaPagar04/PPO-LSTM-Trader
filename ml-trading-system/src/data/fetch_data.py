from binance.client import Client
import pandas as pd
import time
from tqdm import tqdm

client = Client()

def fetch_ohlcv(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1HOUR, start_str="1 Jan 2022"):
    print(f"Fetching {symbol} data with progress...")

    start_ts = int(pd.Timestamp(start_str).timestamp() * 1000)
    end_ts = int(pd.Timestamp.now().timestamp() * 1000)

    all_klines = []

    step = 1000  # Binance limit per request

    pbar = tqdm(total=(end_ts - start_ts) // (60 * 60 * 1000))  # rough estimate (hours)

    while start_ts < end_ts:
        klines = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=step,
            startTime=start_ts
        )

        if not klines:
            break

        all_klines.extend(klines)

        # Move start forward
        start_ts = klines[-1][0] + 1

        pbar.update(len(klines))

        time.sleep(0.2)  # avoid rate limit

    pbar.close()

    # Convert to DataFrame
    columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ]

    df = pd.DataFrame(all_klines, columns=columns)[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ]

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df.sort_values("timestamp").dropna().reset_index(drop=True)

    return df

def main():
    print("Starting data fetch...")

    df = fetch_ohlcv()

    print("Data fetched:", df.shape)
    print(df.head())

    output_path = "data/raw/btc_usdt_1h.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()