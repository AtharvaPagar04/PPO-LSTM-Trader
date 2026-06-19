from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import torch

from src.config.assets import SUPPORTED_ASSETS, asset_to_symbol, normalize_asset_name
from src.config.paths import WALK_FORWARD_DIR, ensure_dir, resolve_checkpoint_path
from src.data.dataset import load_metadata, load_processed_data
from src.evaluation.backtest import run_policy_backtest
from src.evaluation.baselines import (
    BASELINE_STRATEGIES,
    EXPOSURE_EQUIVALENT_BASELINES,
    run_baselines,
)
from src.evaluation.benchmark import build_eval_env, load_policy_from_checkpoint
from src.evaluation.metrics import compute_performance_metrics
from src.evaluation.plot import plot_equity_curves
from src.features.pipeline import ProcessedDataset, engineer_features, load_raw_dataframe
from src.inference import display_path
from src.utils.logger import utc_timestamp_slug, write_json
from src.utils.seed import set_global_seed


@dataclass
class WalkForwardFold:
    fold_index: int
    start_index: int
    end_index: int


def create_walk_forward_folds(
    total_steps: int,
    *,
    folds: int = 5,
    fold_size: int | None = None,
) -> list[WalkForwardFold]:
    if total_steps < 2:
        raise ValueError("Walk-forward evaluation requires at least 2 test windows.")
    if folds is not None and folds <= 0:
        raise ValueError("folds must be a positive integer.")
    if fold_size is not None and fold_size <= 1:
        raise ValueError("fold_size must be greater than 1.")

    num_windows = total_steps
    if fold_size is not None:
        generated = []
        start = 0
        index = 1
        while start < num_windows:
            end = min(start + fold_size, num_windows)
            if end - start < 2:
                break
            generated.append(WalkForwardFold(index, start, end))
            start = end
            index += 1
        if not generated:
            raise ValueError("fold_size is too large for the available test data.")
        return generated

    effective_folds = min(folds, num_windows)
    if effective_folds < 1:
        raise ValueError("Unable to create walk-forward folds from test data.")

    base = num_windows // effective_folds
    remainder = num_windows % effective_folds
    if base < 2:
        raise ValueError(
            f"Test dataset is too small for {effective_folds} chronological folds."
        )

    generated = []
    start = 0
    for fold_index in range(1, effective_folds + 1):
        fold_len = base + (1 if fold_index <= remainder else 0)
        end = start + fold_len
        generated.append(WalkForwardFold(fold_index, start, end))
        start = end
    return generated


def _window_end_timestamps(asset: str, metadata: dict) -> list[pd.Timestamp]:
    raw_df = load_raw_dataframe(asset)
    feature_df = engineer_features(raw_df)
    window_size = int(metadata["window_size"])
    timestamps = [
        pd.Timestamp(feature_df["timestamp"].iloc[idx + window_size - 1])
        for idx in range(len(feature_df) - window_size + 1)
    ]
    split_ratio = float(metadata["split_ratio"])
    split_idx = int(len(timestamps) * split_ratio)
    return timestamps[split_idx:]


def _safe_metric_value(value: float | int | None) -> float:
    if value is None:
        return float("-inf")
    value = float(value)
    if np.isnan(value):
        return float("-inf")
    return value


def _rank_strategies(strategy_metrics: dict[str, dict], metric_name: str) -> list[str]:
    return [
        strategy
        for strategy, _ in sorted(
            strategy_metrics.items(),
            key=lambda item: (-_safe_metric_value(item[1].get(metric_name)), item[0]),
        )
    ]


def evaluate_baselines_on_fold(
    price_windows: np.ndarray,
    *,
    transaction_cost: float,
    seed: int,
    reference_actions: np.ndarray | None = None,
) -> tuple[dict[str, dict], dict[str, dict]]:
    traces = run_baselines(
        price_windows,
        transaction_cost=transaction_cost,
        seed=seed,
        reference_actions=reference_actions,
    )
    metrics = {name: compute_performance_metrics(trace) for name, trace in traces.items()}
    return traces, metrics


def compare_fold_strategies(
    *,
    asset: str,
    fold: WalkForwardFold,
    start_timestamp: pd.Timestamp,
    end_timestamp: pd.Timestamp,
    rl_metrics: dict,
    baseline_metrics: dict[str, dict],
) -> dict:
    strategy_metrics = {
        "rl_policy": rl_metrics,
        **{
            name: baseline_metrics[name]
            for name in (*BASELINE_STRATEGIES, *EXPOSURE_EQUIVALENT_BASELINES)
            if name in baseline_metrics
        },
    }
    ranked_by_return = _rank_strategies(strategy_metrics, "total_return")
    ranked_by_sharpe = _rank_strategies(strategy_metrics, "sharpe")
    strategy_count = len(strategy_metrics)

    return {
        "asset": asset,
        "fold_index": fold.fold_index,
        "start_index": fold.start_index,
        "end_index": fold.end_index - 1,
        "num_steps": rl_metrics["number_of_steps"],
        "start_timestamp": start_timestamp.isoformat(sep=" "),
        "end_timestamp": end_timestamp.isoformat(sep=" "),
        "rl_final_equity": rl_metrics["final_equity"],
        "rl_total_return": rl_metrics["total_return"],
        "rl_sharpe": rl_metrics["sharpe"],
        "rl_max_drawdown": rl_metrics["max_drawdown"],
        "always_long_final_equity": baseline_metrics["always_long"]["final_equity"],
        "always_long_total_return": baseline_metrics["always_long"]["total_return"],
        "always_long_sharpe": baseline_metrics["always_long"]["sharpe"],
        "always_long_max_drawdown": baseline_metrics["always_long"]["max_drawdown"],
        "always_short_final_equity": baseline_metrics["always_short"]["final_equity"],
        "always_short_total_return": baseline_metrics["always_short"]["total_return"],
        "always_short_sharpe": baseline_metrics["always_short"]["sharpe"],
        "always_short_max_drawdown": baseline_metrics["always_short"]["max_drawdown"],
        "always_flat_final_equity": baseline_metrics["always_flat"]["final_equity"],
        "always_flat_total_return": baseline_metrics["always_flat"]["total_return"],
        "always_flat_sharpe": baseline_metrics["always_flat"]["sharpe"],
        "always_flat_max_drawdown": baseline_metrics["always_flat"]["max_drawdown"],
        "random_final_equity": baseline_metrics["random"]["final_equity"],
        "random_total_return": baseline_metrics["random"]["total_return"],
        "random_sharpe": baseline_metrics["random"]["sharpe"],
        "random_max_drawdown": baseline_metrics["random"]["max_drawdown"],
        "constant_signed_mean_action_final_equity": baseline_metrics["constant_signed_mean_action"]["final_equity"],
        "constant_signed_mean_action_total_return": baseline_metrics["constant_signed_mean_action"]["total_return"],
        "constant_signed_mean_action_sharpe": baseline_metrics["constant_signed_mean_action"]["sharpe"],
        "constant_signed_mean_action_max_drawdown": baseline_metrics["constant_signed_mean_action"]["max_drawdown"],
        "constant_abs_mean_long_final_equity": baseline_metrics["constant_abs_mean_long"]["final_equity"],
        "constant_abs_mean_long_total_return": baseline_metrics["constant_abs_mean_long"]["total_return"],
        "constant_abs_mean_long_sharpe": baseline_metrics["constant_abs_mean_long"]["sharpe"],
        "constant_abs_mean_long_max_drawdown": baseline_metrics["constant_abs_mean_long"]["max_drawdown"],
        "constant_abs_mean_short_final_equity": baseline_metrics["constant_abs_mean_short"]["final_equity"],
        "constant_abs_mean_short_total_return": baseline_metrics["constant_abs_mean_short"]["total_return"],
        "constant_abs_mean_short_sharpe": baseline_metrics["constant_abs_mean_short"]["sharpe"],
        "constant_abs_mean_short_max_drawdown": baseline_metrics["constant_abs_mean_short"]["max_drawdown"],
        "best_strategy_by_return": ranked_by_return[0],
        "best_strategy_by_sharpe": ranked_by_sharpe[0],
        "rl_rank_by_return": ranked_by_return.index("rl_policy") + 1,
        "rl_rank_by_sharpe": ranked_by_sharpe.index("rl_policy") + 1,
        "strategy_count": strategy_count,
        "rl_beat_always_long": rl_metrics["total_return"] > baseline_metrics["always_long"]["total_return"],
        "rl_beat_always_short": rl_metrics["total_return"] > baseline_metrics["always_short"]["total_return"],
        "rl_beat_always_flat": rl_metrics["total_return"] > baseline_metrics["always_flat"]["total_return"],
        "rl_beat_random": rl_metrics["total_return"] > baseline_metrics["random"]["total_return"],
        "rl_beat_constant_signed_mean_action": rl_metrics["total_return"]
        > baseline_metrics["constant_signed_mean_action"]["total_return"],
        "rl_beat_constant_abs_mean_long": rl_metrics["total_return"]
        > baseline_metrics["constant_abs_mean_long"]["total_return"],
        "rl_beat_constant_abs_mean_short": rl_metrics["total_return"]
        > baseline_metrics["constant_abs_mean_short"]["total_return"],
    }


def aggregate_walk_forward_metrics(fold_rows: list[dict]) -> dict:
    if not fold_rows:
        raise ValueError("Cannot aggregate empty walk-forward results.")

    final_equities = [row["final_equity"] for row in fold_rows]
    total_returns = [row["total_return"] for row in fold_rows]
    sharpes = [row["sharpe"] for row in fold_rows]
    max_drawdowns = [row["max_drawdown"] for row in fold_rows]
    positive_fold_count = sum(1 for row in fold_rows if row["total_return"] > 0)
    negative_fold_count = sum(1 for row in fold_rows if row["total_return"] <= 0)
    total_folds = len(fold_rows)

    return {
        "mean_final_equity": float(np.mean(final_equities)),
        "median_final_equity": float(median(final_equities)),
        "mean_total_return": float(np.mean(total_returns)),
        "median_total_return": float(median(total_returns)),
        "mean_sharpe": float(np.mean(sharpes)),
        "median_sharpe": float(median(sharpes)),
        "mean_max_drawdown": float(np.mean(max_drawdowns)),
        "worst_max_drawdown": float(np.max(max_drawdowns)),
        "positive_fold_count": positive_fold_count,
        "negative_fold_count": negative_fold_count,
        "total_folds": total_folds,
        "robustness_score": float(positive_fold_count / total_folds),
    }


def aggregate_baseline_comparisons(fold_rows: list[dict]) -> dict:
    if not fold_rows:
        raise ValueError("Cannot aggregate empty baseline comparison results.")

    total_folds = len(fold_rows)
    aggregate = {
        "asset": fold_rows[0]["asset"],
        "total_folds": total_folds,
        "rl_positive_folds": sum(1 for row in fold_rows if row["rl_total_return"] > 0),
        "rl_best_return_fold_count": sum(
            1 for row in fold_rows if row["best_strategy_by_return"] == "rl_policy"
        ),
        "rl_best_sharpe_fold_count": sum(
            1 for row in fold_rows if row["best_strategy_by_sharpe"] == "rl_policy"
        ),
        "rl_beat_always_long_count": sum(
            1 for row in fold_rows if row["rl_beat_always_long"]
        ),
        "rl_beat_always_short_count": sum(
            1 for row in fold_rows if row["rl_beat_always_short"]
        ),
        "rl_beat_always_flat_count": sum(
            1 for row in fold_rows if row["rl_beat_always_flat"]
        ),
        "rl_beat_random_count": sum(1 for row in fold_rows if row["rl_beat_random"]),
        "rl_beat_constant_signed_mean_action_count": sum(
            1 for row in fold_rows if row["rl_beat_constant_signed_mean_action"]
        ),
        "rl_beat_constant_abs_mean_long_count": sum(
            1 for row in fold_rows if row["rl_beat_constant_abs_mean_long"]
        ),
        "rl_beat_constant_abs_mean_short_count": sum(
            1 for row in fold_rows if row["rl_beat_constant_abs_mean_short"]
        ),
    }

    metric_prefixes = (
        "rl",
        "always_long",
        "always_short",
        "always_flat",
        "random",
        "constant_signed_mean_action",
        "constant_abs_mean_long",
        "constant_abs_mean_short",
    )
    for prefix in metric_prefixes:
        aggregate[f"{prefix}_mean_return"] = float(
            np.mean([row[f"{prefix}_total_return"] for row in fold_rows])
        )
        aggregate[f"{prefix}_mean_sharpe"] = float(
            np.mean([row[f"{prefix}_sharpe"] for row in fold_rows])
        )

    aggregate["rl_worst_drawdown"] = float(
        np.max([row["rl_max_drawdown"] for row in fold_rows])
    )
    aggregate["rl_beat_always_long_ratio"] = float(
        aggregate["rl_beat_always_long_count"] / total_folds
    )
    aggregate["rl_beat_always_short_ratio"] = float(
        aggregate["rl_beat_always_short_count"] / total_folds
    )
    aggregate["rl_beat_always_flat_ratio"] = float(
        aggregate["rl_beat_always_flat_count"] / total_folds
    )
    aggregate["rl_beat_random_ratio"] = float(
        aggregate["rl_beat_random_count"] / total_folds
    )
    aggregate["rl_beat_constant_signed_mean_action_ratio"] = float(
        aggregate["rl_beat_constant_signed_mean_action_count"] / total_folds
    )
    aggregate["rl_beat_constant_abs_mean_long_ratio"] = float(
        aggregate["rl_beat_constant_abs_mean_long_count"] / total_folds
    )
    aggregate["rl_beat_constant_abs_mean_short_ratio"] = float(
        aggregate["rl_beat_constant_abs_mean_short_count"] / total_folds
    )

    aggregate["best_overall_strategy_by_mean_return"] = _rank_strategies(
        {
            "rl_policy": {"total_return": aggregate["rl_mean_return"]},
            "always_long": {"total_return": aggregate["always_long_mean_return"]},
            "always_short": {"total_return": aggregate["always_short_mean_return"]},
            "always_flat": {"total_return": aggregate["always_flat_mean_return"]},
            "random": {"total_return": aggregate["random_mean_return"]},
            "constant_signed_mean_action": {
                "total_return": aggregate["constant_signed_mean_action_mean_return"]
            },
            "constant_abs_mean_long": {
                "total_return": aggregate["constant_abs_mean_long_mean_return"]
            },
            "constant_abs_mean_short": {
                "total_return": aggregate["constant_abs_mean_short_mean_return"]
            },
        },
        "total_return",
    )[0]
    aggregate["best_overall_strategy_by_mean_sharpe"] = _rank_strategies(
        {
            "rl_policy": {"sharpe": aggregate["rl_mean_sharpe"]},
            "always_long": {"sharpe": aggregate["always_long_mean_sharpe"]},
            "always_short": {"sharpe": aggregate["always_short_mean_sharpe"]},
            "always_flat": {"sharpe": aggregate["always_flat_mean_sharpe"]},
            "random": {"sharpe": aggregate["random_mean_sharpe"]},
            "constant_signed_mean_action": {
                "sharpe": aggregate["constant_signed_mean_action_mean_sharpe"]
            },
            "constant_abs_mean_long": {
                "sharpe": aggregate["constant_abs_mean_long_mean_sharpe"]
            },
            "constant_abs_mean_short": {
                "sharpe": aggregate["constant_abs_mean_short_mean_sharpe"]
            },
        },
        "sharpe",
    )[0]
    aggregate["buy_and_hold_note"] = (
        "Buy and hold is equivalent to always_long in the current spot-style setup and is excluded from rankings."
    )
    return aggregate


def _write_csv(output_path: Path, rows: list[dict]) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_summary_txt(
    output_dir: Path,
    asset: str,
    aggregate: dict,
    fold_rows: list[dict],
    *,
    include_baselines: bool = False,
    baseline_aggregate: dict | None = None,
) -> None:
    if include_baselines:
        lines = [f"Walk-forward baseline comparison: {asset}", ""]
        lines.append(
            "Fold  RL Ret   Long Ret  Short Ret  Flat Ret  Mean Ret   Best Return   RL Rank"
        )
        for row in fold_rows:
            lines.append(
                f"{row['fold_index']:<5} {row['rl_total_return']:<8.4f} "
                f"{row['always_long_total_return']:<8.4f} {row['always_short_total_return']:<9.4f} "
                f"{row['always_flat_total_return']:<8.4f} {row['constant_signed_mean_action_total_return']:<10.4f} "
                f"{row['best_strategy_by_return']:<13} {row['rl_rank_by_return']}/{row['strategy_count']}"
            )
        lines.extend(
            [
                "",
                "Aggregate:",
                f"RL beat always_long: {baseline_aggregate['rl_beat_always_long_count']}/{baseline_aggregate['total_folds']}",
                f"RL beat always_short: {baseline_aggregate['rl_beat_always_short_count']}/{baseline_aggregate['total_folds']}",
                f"RL beat always_flat: {baseline_aggregate['rl_beat_always_flat_count']}/{baseline_aggregate['total_folds']}",
                f"RL beat random: {baseline_aggregate['rl_beat_random_count']}/{baseline_aggregate['total_folds']}",
                f"RL beat constant mean exposure: {baseline_aggregate['rl_beat_constant_signed_mean_action_count']}/{baseline_aggregate['total_folds']}",
                f"RL beat constant abs short: {baseline_aggregate['rl_beat_constant_abs_mean_short_count']}/{baseline_aggregate['total_folds']}",
                f"RL beat constant abs long: {baseline_aggregate['rl_beat_constant_abs_mean_long_count']}/{baseline_aggregate['total_folds']}",
                f"RL best by return: {baseline_aggregate['rl_best_return_fold_count']}/{baseline_aggregate['total_folds']}",
                f"Best mean-return strategy: {baseline_aggregate['best_overall_strategy_by_mean_return']}",
                "",
                "Note: Baseline walk-forward v1 evaluates existing checkpoints only. It does not retrain per fold and does not imply live trading profitability.",
            ]
        )
    else:
        lines = [f"Walk-forward evaluation: {asset}", ""]
        lines.append("Fold  Steps  Return   Sharpe   Max DD   Start                End")
        for row in fold_rows:
            lines.append(
                f"{row['fold_index']:<5} {row['num_steps']:<5} "
                f"{row['total_return']:<8.4f} {row['sharpe']:<8.2f} "
                f"{row['max_drawdown']:<8.2%} {row['start_timestamp'][:16]}    {row['end_timestamp'][:16]}"
            )
        lines.extend(
            [
                "",
                "Aggregate:",
                f"Mean return: {aggregate['mean_total_return']:.4f}",
                f"Mean Sharpe: {aggregate['mean_sharpe']:.2f}",
                f"Worst drawdown: {aggregate['worst_max_drawdown']:.2%}",
                f"Positive folds: {aggregate['positive_fold_count']}/{aggregate['total_folds']}",
                f"Robustness score: {aggregate['robustness_score']:.2f}",
                "",
                "Note: Walk-forward v1 evaluates existing checkpoints across chronological test folds. It does not retrain per fold.",
            ]
        )
    (output_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def evaluate_walk_forward_asset(
    asset: str,
    *,
    config: dict,
    checkpoint: str | None = None,
    folds: int = 5,
    fold_size: int | None = None,
    output_dir: str | Path | None = None,
    include_baselines: bool = False,
    processed_dataset: ProcessedDataset | None = None,
) -> dict:
    asset = normalize_asset_name(asset)
    set_global_seed(config["evaluation"]["random_seed"])
    if processed_dataset is not None:
        test_X = processed_dataset.test_windows
        test_price = processed_dataset.test_price_windows
        metadata = processed_dataset.metadata
    else:
        test_X, test_price = load_processed_data(asset, "test")
        metadata = load_metadata(asset)
    timestamps = _window_end_timestamps(asset, metadata)
    fold_defs = create_walk_forward_folds(
        total_steps=len(test_X),
        folds=folds,
        fold_size=fold_size,
    )

    max_fold_end = max(f.end_index for f in fold_defs)
    if max_fold_end > len(timestamps):
        raise ValueError(
            f"Walk-forward timestamp/index mismatch: "
            f"max fold.end_index={max_fold_end}, len(timestamps)={len(timestamps)}, "
            f"len(test_X)={len(test_X)}, asset={asset}"
        )

    checkpoint_path = resolve_checkpoint_path(asset, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy_from_checkpoint(
        asset=asset,
        checkpoint_path=checkpoint_path,
        input_dim=test_X.shape[2],
        config=config,
        device=device,
    )

    default_slug = f"{asset}_baselines" if include_baselines else asset
    base_output_dir = (
        Path(output_dir)
        if output_dir
        else WALK_FORWARD_DIR / f"{utc_timestamp_slug()}_{default_slug}"
    )
    ensure_dir(base_output_dir)

    rl_fold_rows = []
    fold_rows = []
    fold_json_rows = []
    comparison_rows = []

    for fold in fold_defs:
        fold_x = test_X[fold.start_index : fold.end_index]
        fold_price = test_price[fold.start_index : fold.end_index]
        env = build_eval_env(fold_x, fold_price, config)
        rl_trace = run_policy_backtest(env, model, deterministic_policy=True)
        baseline_traces, baseline_metrics = evaluate_baselines_on_fold(
            fold_price,
            transaction_cost=config["environment"]["transaction_cost"],
            seed=config["evaluation"]["random_seed"] + fold.fold_index,
            reference_actions=rl_trace["action"],
        )
        rl_metrics = compute_performance_metrics(rl_trace)

        start_ts = timestamps[fold.start_index]
        end_ts = timestamps[fold.end_index - 1]
        rl_row = {
            "asset": asset,
            "fold_index": fold.fold_index,
            "start_index": fold.start_index,
            "end_index": fold.end_index - 1,
            "num_steps": rl_metrics["number_of_steps"],
            "start_timestamp": start_ts.isoformat(sep=" "),
            "end_timestamp": end_ts.isoformat(sep=" "),
            "final_equity": rl_metrics["final_equity"],
            "total_return": rl_metrics["total_return"],
            "sharpe": rl_metrics["sharpe"],
            "max_drawdown": rl_metrics["max_drawdown"],
            "win_rate": rl_metrics["win_rate"],
            "average_position": rl_metrics["average_position"],
            "turnover": rl_metrics["turnover"],
        }
        rl_fold_rows.append(rl_row)

        if include_baselines:
            fold_row = compare_fold_strategies(
                asset=asset,
                fold=fold,
                start_timestamp=start_ts,
                end_timestamp=end_ts,
                rl_metrics=rl_metrics,
                baseline_metrics=baseline_metrics,
            )
            comparison_rows.append(
                {
                    "asset": asset,
                    "fold_index": fold.fold_index,
                    "start_index": fold.start_index,
                    "end_index": fold.end_index - 1,
                    "num_steps": rl_metrics["number_of_steps"],
                    "best_strategy_by_return": fold_row["best_strategy_by_return"],
                    "best_strategy_by_sharpe": fold_row["best_strategy_by_sharpe"],
                    "rl_rank_by_return": fold_row["rl_rank_by_return"],
                    "rl_rank_by_sharpe": fold_row["rl_rank_by_sharpe"],
                    "rl_beat_always_long": fold_row["rl_beat_always_long"],
                    "rl_beat_always_short": fold_row["rl_beat_always_short"],
                    "rl_beat_always_flat": fold_row["rl_beat_always_flat"],
                    "rl_beat_random": fold_row["rl_beat_random"],
                    "rl_beat_constant_signed_mean_action": fold_row["rl_beat_constant_signed_mean_action"],
                    "rl_beat_constant_abs_mean_long": fold_row["rl_beat_constant_abs_mean_long"],
                    "rl_beat_constant_abs_mean_short": fold_row["rl_beat_constant_abs_mean_short"],
                }
            )
        else:
            fold_row = rl_row
        fold_rows.append(fold_row)
        fold_json_rows.append(
            {
                **fold_row,
                "checkpoint": display_path(checkpoint_path),
                "rl_policy": rl_metrics,
                "baselines": baseline_metrics,
            }
        )

        plot_equity_curves(
            {
                "RL": rl_trace["equity"],
                "Always Long": baseline_traces["always_long"]["equity"],
                "Always Short": baseline_traces["always_short"]["equity"],
                "Always Flat": baseline_traces["always_flat"]["equity"],
                "Random": baseline_traces["random"]["equity"],
                "Buy and Hold": baseline_traces["buy_and_hold"]["equity"],
                "Const Mean": baseline_traces["constant_signed_mean_action"]["equity"],
            },
            output_path=base_output_dir / f"equity_fold_{fold.fold_index}.png",
            title=f"{asset} Walk-Forward Fold {fold.fold_index}",
        )

    aggregate = aggregate_walk_forward_metrics(rl_fold_rows)
    baseline_aggregate = (
        aggregate_baseline_comparisons(fold_rows) if include_baselines else None
    )

    write_json(base_output_dir / "fold_metrics.json", {"folds": fold_json_rows})
    _write_csv(base_output_dir / "fold_metrics.csv", fold_rows)
    if include_baselines:
        write_json(
            base_output_dir / "baseline_comparison.json",
            {
                "folds": comparison_rows,
                "buy_and_hold_note": (
                    "Buy and hold is equivalent to always_long in the current spot-style setup and is excluded from rankings."
                ),
            },
        )
        _write_csv(base_output_dir / "baseline_comparison.csv", comparison_rows)
        write_json(base_output_dir / "baseline_aggregate.json", baseline_aggregate)
    write_json(
        base_output_dir / "aggregate_metrics.json",
        {
            "asset": asset,
            "binance_symbol": asset_to_symbol(asset),
            "checkpoint": display_path(checkpoint_path),
            "folds": len(rl_fold_rows),
            "aggregate": aggregate,
        },
    )
    _write_summary_txt(
        base_output_dir,
        asset,
        aggregate,
        fold_rows,
        include_baselines=include_baselines,
        baseline_aggregate=baseline_aggregate,
    )

    result = {
        "asset": asset,
        "binance_symbol": asset_to_symbol(asset),
        "checkpoint": display_path(checkpoint_path),
        "folds": len(rl_fold_rows),
        "fold_rows": fold_rows,
        "aggregate": aggregate,
        "output_dir": str(base_output_dir),
    }
    if include_baselines:
        result["baseline_aggregate"] = baseline_aggregate
        result["comparison_rows"] = comparison_rows
    return result


def evaluate_walk_forward_all(
    *,
    assets: list[str] | None = None,
    config: dict,
    checkpoint: str | None = None,
    folds: int = 5,
    fold_size: int | None = None,
    output_dir: str | Path | None = None,
    include_baselines: bool = False,
) -> dict:
    selected_assets = [normalize_asset_name(asset) for asset in (assets or SUPPORTED_ASSETS)]
    suffix = "all_baselines" if include_baselines else "all"
    root_output_dir = (
        Path(output_dir) if output_dir else WALK_FORWARD_DIR / f"{utc_timestamp_slug()}_{suffix}"
    )
    ensure_dir(root_output_dir)

    results = []
    summary_rows = []
    baseline_summary_rows = []
    for asset in selected_assets:
        asset_output = root_output_dir / asset
        result = evaluate_walk_forward_asset(
            asset,
            config=config,
            checkpoint=checkpoint,
            folds=folds,
            fold_size=fold_size,
            output_dir=asset_output,
            include_baselines=include_baselines,
        )
        results.append(result)
        summary_rows.append(
            {
                "asset": asset,
                "folds": result["aggregate"]["total_folds"],
                "mean_return": result["aggregate"]["mean_total_return"],
                "mean_sharpe": result["aggregate"]["mean_sharpe"],
                "worst_drawdown": result["aggregate"]["worst_max_drawdown"],
                "positive_folds": f"{result['aggregate']['positive_fold_count']}/{result['aggregate']['total_folds']}",
                "robustness_score": result["aggregate"]["robustness_score"],
                "output_dir": str(asset_output),
            }
        )
        if include_baselines:
            baseline = result["baseline_aggregate"]
            baseline_summary_rows.append(
                {
                    "asset": asset,
                    "folds": baseline["total_folds"],
                    "rl_mean_return": baseline["rl_mean_return"],
                    "always_long_mean_return": baseline["always_long_mean_return"],
                    "always_short_mean_return": baseline["always_short_mean_return"],
                    "always_flat_mean_return": baseline["always_flat_mean_return"],
                    "random_mean_return": baseline["random_mean_return"],
                    "constant_signed_mean_action_mean_return": baseline["constant_signed_mean_action_mean_return"],
                    "constant_abs_mean_long_mean_return": baseline["constant_abs_mean_long_mean_return"],
                    "constant_abs_mean_short_mean_return": baseline["constant_abs_mean_short_mean_return"],
                    "rl_best_return_folds": f"{baseline['rl_best_return_fold_count']}/{baseline['total_folds']}",
                    "rl_beat_always_long": f"{baseline['rl_beat_always_long_count']}/{baseline['total_folds']}",
                    "rl_beat_always_flat": f"{baseline['rl_beat_always_flat_count']}/{baseline['total_folds']}",
                    "rl_beat_constant_signed_mean_action": f"{baseline['rl_beat_constant_signed_mean_action_count']}/{baseline['total_folds']}",
                    "best_overall_strategy_by_mean_return": baseline["best_overall_strategy_by_mean_return"],
                    "output_dir": str(asset_output),
                }
            )

    _write_csv(root_output_dir / "all_assets_summary.csv", summary_rows)
    write_json(root_output_dir / "all_assets_summary.json", {"results": results})
    if include_baselines and baseline_summary_rows:
        _write_csv(root_output_dir / "all_assets_baseline_summary.csv", baseline_summary_rows)
        write_json(
            root_output_dir / "all_assets_baseline_summary.json",
            {"results": baseline_summary_rows},
        )

    response = {
        "results": results,
        "summary_rows": summary_rows,
        "output_dir": str(root_output_dir),
    }
    if include_baselines:
        response["baseline_summary_rows"] = baseline_summary_rows
    return response
