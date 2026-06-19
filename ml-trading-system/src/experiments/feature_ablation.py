import csv
import json
import subprocess
import sys
from pathlib import Path

from src.config.assets import normalize_asset_name
from src.config.feature_ablation_presets import (
    apply_feature_preset_to_config,
    validate_requested_feature_ablation_presets,
)
from src.config.paths import CONFIG_DIR, EXPERIMENTS_DIR, ensure_dir
from src.evaluation.benchmark import evaluate_asset
from src.evaluation.diagnostics import collect_model_diagnostics
from src.evaluation.walk_forward import evaluate_walk_forward_asset
from src.features.pipeline import build_processed_dataset
from src.train import train_asset
from src.utils.logger import get_git_commit, utc_timestamp_slug


def calculate_score(
    rl_best_return_fold_count,
    rl_beat_always_flat_count,
    rl_beat_random_count,
    walk_forward_robustness,
    walk_forward_mean_sharpe,
    deterministic_max_drawdown,
    flat_ratio,
):
    return (
        2.0 * rl_best_return_fold_count
        + 1.0 * rl_beat_always_flat_count
        + 1.0 * rl_beat_random_count
        + 1.0 * walk_forward_robustness
        + 1.0 * walk_forward_mean_sharpe
        - 2.0 * deterministic_max_drawdown
        - 0.5 * flat_ratio
    )


def _git_dirty_status():
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or ""


def feature_group_flags(selected_features):
    selected = set(selected_features)
    candle_features = {"body_ratio", "range_pct"}
    volatility_features = {
        "volatility_10",
        "volatility_20",
        "volatility_24",
        "volatility_72",
        "range_pct",
    }
    momentum_features = {"momentum_5", "momentum_10", "trend", "rsi", "log_return"}
    regime_features = {
        "return_24",
        "return_72",
        "ma_ratio_24_72",
        "ma_slope_24",
        "ma_slope_72",
        "trend_strength_24",
        "volatility_regime",
        "rsi_24",
        "drawdown_from_rolling_high_72",
    }
    long_horizon_features = {
        "return_24",
        "return_72",
        "volatility_24",
        "volatility_72",
        "ma_ratio_24_72",
        "ma_slope_24",
        "ma_slope_72",
        "trend_strength_24",
        "rsi_24",
        "drawdown_from_rolling_high_72",
    }
    cross_asset_features = {
        "eth_return_1", "sol_return_1", "eth_return_24", "sol_return_24",
        "eth_return_72", "sol_return_72", "eth_volatility_24", "sol_volatility_24",
        "eth_btc_return_spread_24", "sol_btc_return_spread_24", "market_avg_return_24",
        "market_avg_return_72", "market_volatility_24", "btc_relative_strength_24",
        "btc_relative_strength_72", "eth_btc_correlation_24", "sol_btc_correlation_24"
    }
    relative_strength_features = {
        "btc_relative_strength_24", "btc_relative_strength_72",
        "eth_btc_return_spread_24", "sol_btc_return_spread_24"
    }
    market_regime_features = {
        "market_avg_return_24", "market_avg_return_72", "market_volatility_24"
    }
    return {
        "contains_momentum": bool(selected & momentum_features),
        "contains_volatility": bool(selected & volatility_features),
        "contains_candle_features": bool(selected & candle_features),
        "contains_volume_feature": "vol_z" in selected,
        "contains_regime_features": bool(selected & regime_features),
        "contains_long_horizon_features": bool(selected & long_horizon_features),
        "contains_volatility_regime": "volatility_regime" in selected,
        "contains_drawdown_feature": "drawdown_from_rolling_high_72" in selected,
        "contains_cross_asset_features": bool(selected & cross_asset_features),
        "contains_relative_strength_features": bool(selected & relative_strength_features),
        "contains_market_regime_features": bool(selected & market_regime_features),
    }


def run_feature_ablation_experiment(
    asset: str,
    config: dict,
    presets: list[str],
    quick: bool = False,
):
    asset = normalize_asset_name(asset)
    timestamp = utc_timestamp_slug()
    experiment_dir = EXPERIMENTS_DIR / "feature_ablation" / f"{timestamp}_{asset}"
    ensure_dir(experiment_dir)

    summary_rows = []
    manifest_presets = []
    resolved_presets = dict(zip(presets, validate_requested_feature_ablation_presets(presets)))

    for preset_name in presets:
        preset = resolved_presets[preset_name]
        preset_dir = experiment_dir / preset_name
        ensure_dir(preset_dir)

        preset_config = apply_feature_preset_to_config(config, preset_name)
        if quick:
            preset_config["training"]["iterations"] = 2
            preset_config["training"]["episode_length"] = 128
            preset_config["training"]["rollout_steps"] = 128

        dataset = build_processed_dataset(
            asset=asset,
            window_size=preset_config["data"]["window_size"],
            train_split=preset_config["data"]["train_split"],
            selected_features=preset["features"],
        )

        with (preset_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            json.dump(preset_config, handle, indent=2)

        checkpoint_path = preset_dir / "checkpoint.pth"
        final_checkpoint = preset_dir / "final.pth"
        print(f"\n--- Running preset: {preset_name} ---")
        train_asset(
            asset,
            preset_config,
            best_checkpoint=str(checkpoint_path),
            final_checkpoint=str(final_checkpoint),
            run_dir=preset_dir / "training",
            processed_dataset=dataset,
        )

        evaluation_result, _, _ = evaluate_asset(
            asset=asset,
            config=preset_config,
            checkpoint=str(checkpoint_path),
            output_dir=preset_dir / "evaluation",
            processed_dataset=dataset,
        )
        diagnostics = collect_model_diagnostics(
            asset,
            config=preset_config,
            checkpoint=str(checkpoint_path),
            output_dir=preset_dir / "diagnostics",
            save=True,
            processed_dataset=dataset,
        )
        wf_result = evaluate_walk_forward_asset(
            asset,
            config=preset_config,
            checkpoint=str(checkpoint_path),
            folds=5,
            output_dir=preset_dir / "walk_forward",
            include_baselines=False,
            processed_dataset=dataset,
        )
        wf_baseline_result = evaluate_walk_forward_asset(
            asset,
            config=preset_config,
            checkpoint=str(checkpoint_path),
            folds=5,
            output_dir=preset_dir / "walk_forward_baselines",
            include_baselines=True,
            processed_dataset=dataset,
        )

        diag = diagnostics["summary"]
        wf_agg = wf_result["aggregate"]
        wf_base = wf_baseline_result["baseline_aggregate"]
        score = calculate_score(
            wf_base["rl_best_return_fold_count"],
            wf_base["rl_beat_always_flat_count"],
            wf_base["rl_beat_random_count"],
            wf_agg["robustness_score"],
            wf_agg["mean_sharpe"],
            evaluation_result["rl_policy"]["max_drawdown"],
            diag["flat_ratio_025"],
        )
        flags = feature_group_flags(preset["features"])
        row = {
            "asset": asset,
            "preset": preset_name,
            "feature_count": len(preset["features"]),
            "features": ",".join(preset["features"]),
            "checkpoint_path": str(checkpoint_path),
            "deterministic_return": evaluation_result["rl_policy"]["total_return"],
            "deterministic_sharpe": evaluation_result["rl_policy"]["sharpe"],
            "deterministic_max_drawdown": evaluation_result["rl_policy"]["max_drawdown"],
            "final_equity": evaluation_result["rl_policy"]["final_equity"],
            "walk_forward_mean_return": wf_agg["mean_total_return"],
            "walk_forward_mean_sharpe": wf_agg["mean_sharpe"],
            "walk_forward_worst_drawdown": wf_agg["worst_max_drawdown"],
            "walk_forward_positive_folds": wf_agg["positive_fold_count"],
            "walk_forward_robustness": wf_agg["robustness_score"],
            "rl_best_return_fold_count": wf_base["rl_best_return_fold_count"],
            "rl_beat_always_long_count": wf_base["rl_beat_always_long_count"],
            "rl_beat_always_short_count": wf_base["rl_beat_always_short_count"],
            "rl_beat_always_flat_count": wf_base["rl_beat_always_flat_count"],
            "rl_beat_random_count": wf_base["rl_beat_random_count"],
            "rl_beat_constant_signed_mean_action_count": wf_base["rl_beat_constant_signed_mean_action_count"],
            "rl_beat_constant_abs_mean_long_count": wf_base["rl_beat_constant_abs_mean_long_count"],
            "rl_beat_constant_abs_mean_short_count": wf_base["rl_beat_constant_abs_mean_short_count"],
            "constant_signed_mean_action_mean_return": wf_base["constant_signed_mean_action_mean_return"],
            "constant_abs_mean_long_mean_return": wf_base["constant_abs_mean_long_mean_return"],
            "constant_abs_mean_short_mean_return": wf_base["constant_abs_mean_short_mean_return"],
            "flat_ratio": diag["flat_ratio_025"],
            "flat_ratio_001": diag["flat_ratio_001"],
            "flat_ratio_005": diag["flat_ratio_005"],
            "flat_ratio_010": diag["flat_ratio_010"],
            "flat_ratio_025": diag["flat_ratio_025"],
            "long_ratio": diag["long_ratio_025"],
            "short_ratio": diag["short_ratio_025"],
            "long_ratio_001": diag["long_ratio_001"],
            "short_ratio_001": diag["short_ratio_001"],
            "long_ratio_005": diag["long_ratio_005"],
            "short_ratio_005": diag["short_ratio_005"],
            "long_ratio_010": diag["long_ratio_010"],
            "short_ratio_010": diag["short_ratio_010"],
            "long_ratio_025": diag["long_ratio_025"],
            "short_ratio_025": diag["short_ratio_025"],
            "dominant_action_side": diag["dominant_action_side"],
            "action_mean": diag["action_mean"],
            "action_std": diag["action_std"],
            "action_min": diag["action_min"],
            "action_max": diag["action_max"],
            "average_abs_action": diag["average_abs_action"],
            "action_abs_mean": diag["action_abs_mean"],
            "action_abs_median": diag["action_abs_median"],
            "action_abs_p75": diag["action_abs_p75"],
            "action_abs_p90": diag["action_abs_p90"],
            "action_abs_p95": diag["action_abs_p95"],
            "action_abs_p99": diag["action_abs_p99"],
            "positive_action_ratio": diag["positive_action_ratio"],
            "negative_action_ratio": diag["negative_action_ratio"],
            "turnover": diag["turnover"],
            "policy_std_mean": diag["policy_std_mean"],
            "score": score,
            **flags,
        }
        summary_rows.append(row)
        manifest_presets.append(
            {
                "preset_name": preset_name,
                "description": preset["description"],
                "selected_features": preset["features"],
                "feature_count": len(preset["features"]),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_exists": checkpoint_path.exists(),
                "evaluation_checkpoint_used": str(checkpoint_path),
                "diagnostics_checkpoint_used": str(checkpoint_path),
                "walk_forward_checkpoint_used": str(checkpoint_path),
                "baseline_checkpoint_used": str(checkpoint_path),
                "input_dim": len(preset["features"]),
                "window_size": dataset.metadata["window_size"],
                "train_rows": dataset.metadata["train_rows"],
                "test_rows": dataset.metadata["test_rows"],
            }
        )

    with (experiment_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (experiment_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    best_by_score = max(summary_rows, key=lambda row: row["score"])["preset"]
    best_by_walk_forward_sharpe = max(
        summary_rows, key=lambda row: row["walk_forward_mean_sharpe"]
    )["preset"]
    best_by_low_drawdown = min(
        summary_rows, key=lambda row: row["deterministic_max_drawdown"]
    )["preset"]
    best_by_low_flat_ratio = min(summary_rows, key=lambda row: row["flat_ratio_010"])["preset"]
    best_by_baseline_wins = max(
        summary_rows,
        key=lambda row: (
            row["rl_beat_always_long_count"]
            + row["rl_beat_always_short_count"]
            + row["rl_beat_always_flat_count"]
            + row["rl_beat_random_count"]
        ),
    )["preset"]

    report_lines = [
        "# Feature Ablation Experiment",
        "",
        "This score is only a research comparison helper. It is not proof of trading profitability.",
        "",
        "## Summary Table",
        "",
        "| Preset | Count | Return | WF Sharpe | Flat@1 | Flat@5 | Flat@10 | Flat@25 | Side | Avg |Act| | RL Best | Score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['preset']} | {row['feature_count']} | {row['deterministic_return']:.4f} | "
            f"{row['walk_forward_mean_sharpe']:.2f} | {row['flat_ratio_001']*100:.1f}% | {row['flat_ratio_005']*100:.1f}% | "
            f"{row['flat_ratio_010']*100:.1f}% | {row['flat_ratio_025']*100:.1f}% | {row['dominant_action_side']} | "
            f"{row['average_abs_action']:.4f} | {row['rl_best_return_fold_count']}/5 | {row['score']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Best Presets",
            f"- Best by score: {best_by_score}",
            f"- Best by walk-forward Sharpe: {best_by_walk_forward_sharpe}",
            f"- Best by baseline wins: {best_by_baseline_wins}",
            f"- Best by low drawdown: {best_by_low_drawdown}",
            f"- Best by low flat ratio: {best_by_low_flat_ratio}",
            "",
            "## Feature Group Findings",
            "- Compare the summary rows to see whether smaller directional subsets reduce flat behavior or improve baseline wins.",
            "",
            "## Flat-Policy Behavior",
            "- Use Flat@1/5/10/25 plus dominant side and average absolute action as the primary continuous-action checks.",
            "",
            "## Baseline Comparison",
            "- Use RL best-fold count and exposure-equivalent baseline win counts to judge timing beyond static exposure.",
            "",
            "## Warnings",
            "- Feature Ablation v1 is experimental and should not be treated as evidence of live profitability.",
            "",
            "## Recommended Next Step",
            "- If a smaller feature set improves baseline wins, run repeated seeds on the top presets. If all presets remain flat, revisit objective/action mapping rather than adding more features immediately.",
        ]
    )
    with (experiment_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines))

    manifest = {
        "asset": asset,
        "experiment_type": "feature-ablation",
        "timestamp": timestamp,
        "presets": presets,
        "base_config_path": str(CONFIG_DIR / "default.yaml"),
        "python_command": " ".join(sys.argv),
        "git_commit_if_available": get_git_commit(),
        "git_dirty_status_if_available": _git_dirty_status(),
        "presets_data": manifest_presets,
    }
    with (experiment_dir / "audit_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nFeature ablation experiment: {asset}\n")
    print(f"{'Preset':<22} {'Count':<5} {'Return':<8} {'WF Sharpe':<10} {'Flat@1':<7} {'Flat@5':<7} {'Flat@10':<8} {'Flat@25':<8} {'Side':<13} {'Avg|Act|':<9} {'RL Best':<7} {'Score'}")
    for row in summary_rows:
        print(
            f"{row['preset']:<22} {row['feature_count']:<5} {row['deterministic_return']:<8.4f} "
            f"{row['walk_forward_mean_sharpe']:<10.2f} {row['flat_ratio_001']*100:<7.1f} {row['flat_ratio_005']*100:<7.1f} "
            f"{row['flat_ratio_010']*100:<8.1f} {row['flat_ratio_025']*100:<8.1f} {row['dominant_action_side']:<13} "
            f"{row['average_abs_action']:<9.4f} {row['rl_best_return_fold_count']}/5     {row['score']:.4f}"
        )
    print(f"\nBest by score: {best_by_score}")
    print(f"Best by walk-forward Sharpe: {best_by_walk_forward_sharpe}")
    print(f"Best by baseline wins: {best_by_baseline_wins}")
    print(f"Best by low drawdown: {best_by_low_drawdown}")
    print(f"Best by low flat ratio: {best_by_low_flat_ratio}")
    print(f"\nOutput: {experiment_dir}")

    return {
        "experiment_dir": str(experiment_dir),
        "best_by_score": best_by_score,
        "best_by_walk_forward_sharpe": best_by_walk_forward_sharpe,
        "best_by_baseline_wins": best_by_baseline_wins,
        "best_by_low_drawdown": best_by_low_drawdown,
        "best_by_low_flat_ratio": best_by_low_flat_ratio,
        "summary": summary_rows,
    }
