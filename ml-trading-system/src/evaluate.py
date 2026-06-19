import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.assets import SUPPORTED_ASSETS, normalize_asset_name
from src.config.settings import load_config
from src.evaluation.benchmark import evaluate_asset, write_summary
from src.evaluation.walk_forward import (
    evaluate_walk_forward_all,
    evaluate_walk_forward_asset,
)


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


def print_walk_forward_asset(result):
    print(f"Walk-forward evaluation: {result['asset']}\n")
    print("Fold  Steps  Return   Sharpe   Max DD   Start                End")
    for row in result["fold_rows"]:
        print(
            f"{row['fold_index']:<5} {row['num_steps']:<5} "
            f"{row['total_return']:<8.4f} {row['sharpe']:<8.2f} "
            f"{row['max_drawdown']:<8.2%} {row['start_timestamp'][:16]}    {row['end_timestamp'][:16]}"
        )
    aggregate = result["aggregate"]
    print("\nAggregate:")
    print(f"Mean return: {aggregate['mean_total_return']:.4f}")
    print(f"Mean Sharpe: {aggregate['mean_sharpe']:.2f}")
    print(f"Worst drawdown: {aggregate['worst_max_drawdown']:.2%}")
    print(
        f"Positive folds: {aggregate['positive_fold_count']}/{aggregate['total_folds']}"
    )
    print(f"Robustness score: {aggregate['robustness_score']:.2f}")
    print(f"Output: {result['output_dir']}")


def print_walk_forward_asset_baselines(result):
    print(f"Walk-forward baseline comparison: {result['asset']}\n")
    print("Fold  RL Ret   Long Ret  Short Ret  Flat Ret  Random Ret  Best Return   RL Rank")
    for row in result["fold_rows"]:
        print(
            f"{row['fold_index']:<5} {row['rl_total_return']:<8.4f} "
            f"{row['always_long_total_return']:<8.4f} {row['always_short_total_return']:<9.4f} "
            f"{row['always_flat_total_return']:<8.4f} {row['random_total_return']:<10.4f} "
            f"{row['best_strategy_by_return']:<13} {row['rl_rank_by_return']}/5"
        )
    aggregate = result["baseline_aggregate"]
    print("\nAggregate:")
    print(f"RL beat always_long: {aggregate['rl_beat_always_long_count']}/{aggregate['total_folds']}")
    print(f"RL beat always_short: {aggregate['rl_beat_always_short_count']}/{aggregate['total_folds']}")
    print(f"RL beat always_flat: {aggregate['rl_beat_always_flat_count']}/{aggregate['total_folds']}")
    print(f"RL beat random: {aggregate['rl_beat_random_count']}/{aggregate['total_folds']}")
    print(f"RL best by return: {aggregate['rl_best_return_fold_count']}/{aggregate['total_folds']}")
    print(f"Best mean-return strategy: {aggregate['best_overall_strategy_by_mean_return']}")
    print(f"Output: {result['output_dir']}")


def print_walk_forward_all(summary_rows, output_dir):
    print("Asset     Folds  Mean Return  Mean Sharpe  Worst DD  Positive Folds  Robustness")
    for row in summary_rows:
        print(
            f"{row['asset']:<8}  {row['folds']:<5}  {row['mean_return']:<11.4f}  "
            f"{row['mean_sharpe']:<11.2f}  {row['worst_drawdown']:<8.2%}  "
            f"{row['positive_folds']:<14}  {row['robustness_score']:.2f}"
        )
    print(
        "\nNote: Walk-forward v1 evaluates existing checkpoints across chronological test folds. It does not retrain per fold."
    )
    print(f"Output: {output_dir}")


def print_walk_forward_all_baselines(summary_rows, output_dir):
    print(
        "Asset     Folds  RL Mean Ret  Long Mean Ret  Short Mean Ret  Flat Mean Ret  RL Best  RL > Long  RL > Flat"
    )
    for row in summary_rows:
        print(
            f"{row['asset']:<8}  {row['folds']:<5}  {row['rl_mean_return']:<11.4f}  "
            f"{row['always_long_mean_return']:<13.4f}  {row['always_short_mean_return']:<14.4f}  "
            f"{row['always_flat_mean_return']:<13.4f}  {row['rl_best_return_folds']:<7}  "
            f"{row['rl_beat_always_long']:<9}  {row['rl_beat_always_flat']}"
        )
    print(
        "\nNote: Baseline walk-forward v1 evaluates existing checkpoints only. It does not retrain per fold and does not imply live trading profitability."
    )
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(*( [args.config] if args.config else [] ))
    assets = SUPPORTED_ASSETS if args.all else [normalize_asset_name(args.asset or "btc_usdt")]
    if args.walk_forward:
        if args.folds <= 0:
            print("[ERROR] folds must be a positive integer.")
            raise SystemExit(1)
        if args.fold_size is not None and args.fold_size <= 1:
            print("[ERROR] fold-size must be greater than 1.")
            raise SystemExit(1)
        if args.all:
            result = evaluate_walk_forward_all(
                assets=assets,
                config=config,
                checkpoint=args.checkpoint,
                folds=args.folds,
                fold_size=args.fold_size,
                output_dir=args.output_dir,
                include_baselines=args.baselines,
            )
            if args.baselines:
                print_walk_forward_all_baselines(
                    result["baseline_summary_rows"], result["output_dir"]
                )
            else:
                print_walk_forward_all(result["summary_rows"], result["output_dir"])
            raise SystemExit(0)
        result = evaluate_walk_forward_asset(
            assets[0],
            config=config,
            checkpoint=args.checkpoint,
            folds=args.folds,
            fold_size=args.fold_size,
            output_dir=args.output_dir,
            include_baselines=args.baselines,
        )
        if args.baselines:
            print_walk_forward_asset_baselines(result)
        else:
            print_walk_forward_asset(result)
        raise SystemExit(0)
    if args.baselines:
        print("[ERROR] --baselines is currently supported only with --walk-forward")
        raise SystemExit(1)

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
