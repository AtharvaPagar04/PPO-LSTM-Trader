import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.assets import SUPPORTED_ASSETS, normalize_asset_name
from src.config.settings import load_config
from src.evaluation.benchmark import evaluate_asset, write_summary


def print_result(result):
    rl = result["rl_policy"]
    print(
        f"{result['asset']:>8} | equity={rl['final_equity']:.4f} | return={rl['total_return']:.4f} | "
        f"sharpe={rl['sharpe']:.2f} | mdd={rl['max_drawdown']:.2%} | steps={rl['number_of_steps']}"
    )


def evaluate_assets(assets, config, checkpoint=None):
    results = []
    for asset in assets:
        result, _, _ = evaluate_asset(
            asset=asset,
            config=config,
            checkpoint=checkpoint if len(assets) == 1 else None,
        )
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    config = load_config(*( [args.config] if args.config else [] ))
    assets = SUPPORTED_ASSETS if args.all else [normalize_asset_name(args.asset or "btc_usdt")]
    try:
        results = evaluate_assets(
            assets=assets,
            config=config,
            checkpoint=args.checkpoint if not args.all else None,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        raise SystemExit(1) from exc

    for result in results:
        print_result(result)

    if args.all and results:
        write_summary(results)
        print(f"Saved evaluation summary for {len(results)} assets.")


if __name__ == "__main__":
    main()
