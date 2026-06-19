from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from src.config.assets import normalize_asset_name
from src.config.feature_ablation_presets import (
    apply_feature_preset_to_config,
    validate_requested_feature_ablation_presets,
)
from src.config.paths import EXPERIMENTS_DIR, ensure_dir
from src.evaluation.benchmark import evaluate_asset
from src.evaluation.diagnostics import collect_model_diagnostics
from src.evaluation.walk_forward import evaluate_walk_forward_asset
from src.experiments.feature_ablation import calculate_score, feature_group_flags
from src.features.pipeline import build_processed_dataset
from src.train import train_asset
from src.utils.logger import get_git_commit, utc_timestamp_slug


DEFAULT_FEATURE_PRESETS = ["full_features", "price_action_minimal"]
DEFAULT_SEEDS = [42, 43, 44]


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


def _coerce_float(value) -> float:
    return float(value)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return variance**0.5


def _winner_by_metric(rows: list[dict], metric: str, *, higher_is_better: bool) -> str:
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -_coerce_float(row[metric]) if higher_is_better else _coerce_float(row[metric]),
            row["feature_preset"],
        ),
    )
    return sorted_rows[0]["feature_preset"]


def compute_winner_counts(summary_rows: list[dict]) -> dict[str, dict[str, int]]:
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for row in summary_rows:
        by_seed[int(row["seed"])].append(row)

    metrics = {
        "score": True,
        "walk_forward_mean_sharpe": True,
        "deterministic_return": True,
        "average_abs_action": True,
        "deterministic_max_drawdown": False,
        "flat_ratio_010": False,
    }
    result = {metric: defaultdict(int) for metric in metrics}
    for seed_rows in by_seed.values():
        for metric, higher_is_better in metrics.items():
            winner = _winner_by_metric(
                seed_rows, metric, higher_is_better=higher_is_better
            )
            result[metric][winner] += 1
    return {metric: dict(counts) for metric, counts in result.items()}


def aggregate_seed_validation_runs(summary_rows: list[dict]) -> list[dict]:
    winner_counts = compute_winner_counts(summary_rows)
    by_preset: dict[str, list[dict]] = defaultdict(list)
    for row in summary_rows:
        by_preset[row["feature_preset"]].append(row)

    aggregate_rows = []
    for preset_name, rows in sorted(by_preset.items()):
        deterministic_returns = [float(row["deterministic_return"]) for row in rows]
        wf_returns = [float(row["walk_forward_mean_return"]) for row in rows]
        wf_sharpes = [float(row["walk_forward_mean_sharpe"]) for row in rows]
        flat_ratios_001 = [float(row["flat_ratio_001"]) for row in rows]
        flat_ratios_005 = [float(row["flat_ratio_005"]) for row in rows]
        flat_ratios_010 = [float(row["flat_ratio_010"]) for row in rows]
        flat_ratios_025 = [float(row["flat_ratio_025"]) for row in rows]
        avg_abs_actions = [float(row["average_abs_action"]) for row in rows]
        max_drawdowns = [float(row["deterministic_max_drawdown"]) for row in rows]
        rl_best = [float(row["rl_best_return_fold_count"]) for row in rows]
        rl_vs_flat = [float(row["rl_beat_always_flat_count"]) for row in rows]
        const_mean_returns = [
            float(row.get("constant_signed_mean_action_mean_return", 0.0))
            for row in rows
        ]
        const_abs_long_returns = [
            float(row.get("constant_abs_mean_long_mean_return", 0.0)) for row in rows
        ]
        const_abs_short_returns = [
            float(row.get("constant_abs_mean_short_mean_return", 0.0)) for row in rows
        ]
        scores = [float(row["score"]) for row in rows]

        aggregate_rows.append(
            {
                "asset": rows[0]["asset"],
                "feature_preset": preset_name,
                "feature_count": rows[0]["feature_count"],
                "features": rows[0]["features"],
                "num_seeds": len(rows),
                "mean_deterministic_return": _mean(deterministic_returns),
                "std_deterministic_return": _std(deterministic_returns),
                "min_deterministic_return": min(deterministic_returns),
                "max_deterministic_return": max(deterministic_returns),
                "mean_walk_forward_mean_return": _mean(wf_returns),
                "std_walk_forward_mean_return": _std(wf_returns),
                "mean_walk_forward_mean_sharpe": _mean(wf_sharpes),
                "std_walk_forward_mean_sharpe": _std(wf_sharpes),
                "min_walk_forward_mean_sharpe": min(wf_sharpes),
                "max_walk_forward_mean_sharpe": max(wf_sharpes),
                "mean_flat_ratio_001": _mean(flat_ratios_001),
                "std_flat_ratio_001": _std(flat_ratios_001),
                "mean_flat_ratio_005": _mean(flat_ratios_005),
                "std_flat_ratio_005": _std(flat_ratios_005),
                "mean_flat_ratio_010": _mean(flat_ratios_010),
                "std_flat_ratio_010": _std(flat_ratios_010),
                "mean_flat_ratio_025": _mean(flat_ratios_025),
                "std_flat_ratio_025": _std(flat_ratios_025),
                "mean_average_abs_action": _mean(avg_abs_actions),
                "std_average_abs_action": _std(avg_abs_actions),
                "mean_max_drawdown": _mean(max_drawdowns),
                "std_max_drawdown": _std(max_drawdowns),
                "mean_rl_best_return_fold_count": _mean(rl_best),
                "total_rl_best_return_fold_count": int(sum(rl_best)),
                "mean_rl_beat_always_flat_count": _mean(rl_vs_flat),
                "total_rl_beat_always_flat_count": int(sum(rl_vs_flat)),
                "mean_constant_signed_mean_action_mean_return": _mean(const_mean_returns),
                "mean_constant_abs_mean_long_mean_return": _mean(const_abs_long_returns),
                "mean_constant_abs_mean_short_mean_return": _mean(const_abs_short_returns),
                "mean_score": _mean(scores),
                "std_score": _std(scores),
                "win_count_by_score": winner_counts["score"].get(preset_name, 0),
                "win_count_by_walk_forward_sharpe": winner_counts[
                    "walk_forward_mean_sharpe"
                ].get(preset_name, 0),
                "win_count_by_deterministic_return": winner_counts[
                    "deterministic_return"
                ].get(preset_name, 0),
                "win_count_by_average_abs_action": winner_counts[
                    "average_abs_action"
                ].get(preset_name, 0),
                "win_count_by_lowest_max_drawdown": winner_counts[
                    "deterministic_max_drawdown"
                ].get(preset_name, 0),
                "win_count_by_lowest_flat_ratio": winner_counts["flat_ratio_010"].get(
                    preset_name, 0
                ),
            }
        )
    return aggregate_rows


def _build_report(summary_rows: list[dict], aggregate_rows: list[dict], seeds: list[int]) -> str:
    best_by_score = max(aggregate_rows, key=lambda row: row["mean_score"])["feature_preset"]
    best_by_wf_sharpe = max(
        aggregate_rows, key=lambda row: row["mean_walk_forward_mean_sharpe"]
    )["feature_preset"]
    best_by_avg_abs_action = max(
        aggregate_rows, key=lambda row: row["mean_average_abs_action"]
    )["feature_preset"]

    lines = [
        "# Repeated Seed Feature Validation",
        "",
        "This score is only a research comparison helper. It is not proof of trading profitability.",
        "",
        "## Experiment Setup",
        "",
        f"- Seeds: {', '.join(str(seed) for seed in seeds)}",
        f"- Runs: {len(summary_rows)}",
        "",
        "## Per-Seed Results",
        "",
        "| Preset | Seed | Det Return | WF Sharpe | Flat@1 | Flat@5 | Flat@10 | Flat@25 | Avg |Act| | RL Best | Score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['feature_preset']} | {row['seed']} | {row['deterministic_return']:.4f} | "
            f"{row['walk_forward_mean_sharpe']:.4f} | {row['flat_ratio_001']*100:.1f}% | {row['flat_ratio_005']*100:.1f}% | "
            f"{row['flat_ratio_010']*100:.1f}% | {row['flat_ratio_025']*100:.1f}% | {row['average_abs_action']:.4f} | {row['rl_best_return_fold_count']}/5 | {row['score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Aggregate Summary",
            "",
            "| Preset | Seeds | Mean WF Sharpe | Std WF Sharpe | Flat@1 | Flat@5 | Flat@10 | Flat@25 | Mean Avg |Act| | Mean Score |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            f"| {row['feature_preset']} | {row['num_seeds']} | {row['mean_walk_forward_mean_sharpe']:.4f} | "
            f"{row['std_walk_forward_mean_sharpe']:.4f} | {row['mean_flat_ratio_001']*100:.1f}% | {row['mean_flat_ratio_005']*100:.1f}% | "
            f"{row['mean_flat_ratio_010']*100:.1f}% | {row['mean_flat_ratio_025']*100:.1f}% | "
            f"{row['mean_average_abs_action']:.4f} | {row['mean_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Winner Counts",
            "",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            f"- {row['feature_preset']}: score {row['win_count_by_score']}/{row['num_seeds']}, "
            f"WF Sharpe {row['win_count_by_walk_forward_sharpe']}/{row['num_seeds']}, "
            f"det return {row['win_count_by_deterministic_return']}/{row['num_seeds']}, "
            f"avg |action| {row['win_count_by_average_abs_action']}/{row['num_seeds']}"
        )
    lines.extend(
        [
            "",
            "## Stability Analysis",
            "",
            "Lower standard deviation across seeds indicates more stable behavior for the measured metric.",
            "",
            "## Flat-Policy Analysis",
            "",
            "Flat@1/5/10/25 and mean average absolute action are reported to show whether either preset consistently escapes low-exposure behavior.",
            "",
            "## Baseline Comparison",
            "",
            "RL beat counts against always-flat and RL best-fold counts are aggregated across seeds.",
            "",
            "## Recommendation",
            "",
            f"- Best preset by mean walk-forward Sharpe: {best_by_wf_sharpe}",
            f"- Best preset by mean score: {best_by_score}",
            f"- Highest mean average absolute action: {best_by_avg_abs_action}",
            "",
            "## Limitations",
            "",
            "- This is still an offline research comparison.",
            "- Three seeds provide directional evidence, not final model selection confidence.",
            "- Results do not imply live trading profitability.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_seed_validation_experiment(
    asset: str,
    config: dict,
    feature_presets: list[str] | None = None,
    seeds: list[int] | None = None,
    quick: bool = False,
):
    asset = normalize_asset_name(asset)
    feature_presets = feature_presets or list(DEFAULT_FEATURE_PRESETS)
    seeds = [int(seed) for seed in (seeds or DEFAULT_SEEDS)]

    timestamp = utc_timestamp_slug()
    experiment_dir = EXPERIMENTS_DIR / "seed_validation" / f"{timestamp}_{asset}"
    ensure_dir(experiment_dir)

    summary_rows = []
    manifest_runs = []
    resolved_presets = dict(
        zip(feature_presets, validate_requested_feature_ablation_presets(feature_presets))
    )

    for preset_name in feature_presets:
        preset = resolved_presets[preset_name]
        preset_base_config = apply_feature_preset_to_config(config, preset_name)
        preset_dir = experiment_dir / preset_name
        ensure_dir(preset_dir)

        for seed in seeds:
            run_config = json.loads(json.dumps(preset_base_config))
            run_config.setdefault("training", {})
            run_config["training"]["seed"] = seed
            if quick:
                run_config["training"]["iterations"] = 2
                run_config["training"]["episode_length"] = 128
                run_config["training"]["rollout_steps"] = 128

            dataset = build_processed_dataset(
                asset=asset,
                window_size=run_config["data"]["window_size"],
                train_split=run_config["data"]["train_split"],
                selected_features=preset["features"],
            )

            run_dir = preset_dir / f"seed_{seed}"
            ensure_dir(run_dir)
            with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
                json.dump(run_config, handle, indent=2)

            checkpoint_path = run_dir / "checkpoint.pth"
            final_checkpoint = run_dir / "final.pth"
            print(f"\n--- Running preset: {preset_name} | seed: {seed} ---")
            train_asset(
                asset,
                run_config,
                best_checkpoint=str(checkpoint_path),
                final_checkpoint=str(final_checkpoint),
                run_dir=run_dir / "training",
                processed_dataset=dataset,
            )

            evaluation_result, _, _ = evaluate_asset(
                asset=asset,
                config=run_config,
                checkpoint=str(checkpoint_path),
                output_dir=run_dir / "evaluation",
                processed_dataset=dataset,
            )
            diagnostics = collect_model_diagnostics(
                asset,
                config=run_config,
                checkpoint=str(checkpoint_path),
                output_dir=run_dir / "diagnostics",
                save=True,
                processed_dataset=dataset,
            )
            wf_result = evaluate_walk_forward_asset(
                asset,
                config=run_config,
                checkpoint=str(checkpoint_path),
                folds=5,
                output_dir=run_dir / "walk_forward",
                include_baselines=False,
                processed_dataset=dataset,
            )
            wf_baseline_result = evaluate_walk_forward_asset(
                asset,
                config=run_config,
                checkpoint=str(checkpoint_path),
                folds=5,
                output_dir=run_dir / "walk_forward_baselines",
                include_baselines=True,
                processed_dataset=dataset,
            )

            diag = diagnostics["summary"]
            wf_agg = wf_result["aggregate"]
            wf_base = wf_baseline_result["baseline_aggregate"]
            row = {
                "asset": asset,
                "feature_preset": preset_name,
                "seed": seed,
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
            }
            row.update(feature_group_flags(preset["features"]))
            row["score"] = calculate_score(
                row["rl_best_return_fold_count"],
                row["rl_beat_always_flat_count"],
                row["rl_beat_random_count"],
                row["walk_forward_robustness"],
                row["walk_forward_mean_sharpe"],
                row["deterministic_max_drawdown"],
                row["flat_ratio_025"],
            )
            summary_rows.append(row)
            manifest_runs.append(
                {
                    "feature_preset": preset_name,
                    "seed": seed,
                    "selected_features": preset["features"],
                    "feature_count": len(preset["features"]),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_exists": checkpoint_path.exists(),
                    "input_dim": len(preset["features"]),
                    "window_size": dataset.metadata["window_size"],
                    "train_rows": dataset.metadata["train_rows"],
                    "test_rows": dataset.metadata["test_rows"],
                    "deterministic_evaluation_checkpoint_used": str(checkpoint_path),
                    "diagnostics_checkpoint_used": str(checkpoint_path),
                    "walk_forward_checkpoint_used": str(checkpoint_path),
                    "baseline_checkpoint_used": str(checkpoint_path),
                }
            )

    aggregate_rows = aggregate_seed_validation_runs(summary_rows)
    winner_counts = compute_winner_counts(summary_rows)

    with (experiment_dir / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (experiment_dir / "runs.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
    with (experiment_dir / "aggregate_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    with (experiment_dir / "aggregate_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"aggregate_rows": aggregate_rows, "winner_counts": winner_counts},
            handle,
            indent=2,
        )
    with (experiment_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write(_build_report(summary_rows, aggregate_rows, seeds))

    manifest = {
        "asset": asset,
        "experiment_type": "seed-validation",
        "timestamp": timestamp,
        "feature_presets": feature_presets,
        "seeds": seeds,
        "base_config_path": "configs/default.yaml",
        "python_command": " ".join(sys.argv),
        "git_commit_if_available": get_git_commit(),
        "git_dirty_status_if_available": _git_dirty_status(),
        "quick_mode": quick,
        "runs_data": manifest_runs,
    }
    with (experiment_dir / "audit_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return {
        "experiment_dir": str(experiment_dir),
        "summary": summary_rows,
        "aggregate_summary": aggregate_rows,
        "winner_counts": winner_counts,
    }
