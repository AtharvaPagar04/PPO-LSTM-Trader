from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from src.config.assets import normalize_asset_name
from src.config.feature_ablation_presets import (
    apply_feature_preset_to_config,
    validate_requested_feature_ablation_presets,
)
from src.config.paths import EXPERIMENTS_DIR, ensure_dir
from src.evaluation.backtest import run_action_backtest
from src.evaluation.benchmark import build_eval_env
from src.evaluation.diagnostics import (
    compute_action_bucket_ratios,
    compute_action_distribution,
    compute_directional_signal_diagnostics,
    compute_threshold_sensitivity,
    collect_model_diagnostics,
)
from src.evaluation.metrics import compute_performance_metrics
from src.features.pipeline import build_processed_dataset
from src.train import train_asset
from src.utils.logger import get_git_commit, utc_timestamp_slug


ACTION_FLOW_SUMMARY = {
    "model_output_mean": "LSTMPolicy.forward -> ActorHead returns continuous mean action.",
    "training_action_sampling": "PPOTrainer.collect_rollout samples Normal(mean, std), then clamps to [-1, 1].",
    "deterministic_evaluation_action": "Evaluation uses policy mean directly, without bucketization.",
    "action_clipping": "TradingEnv.step clips incoming action to [-1, 1] and sets position equal to clipped action.",
    "position_mapping": "Position size equals continuous clipped action; no rounding or discrete long/short/flat mapping is applied.",
    "transaction_cost": "Transaction cost = abs(position - previous_position) * cost.",
    "flat_ratio_reporting": "Diagnostics computes flat_ratio from action buckets only; default flat threshold is 0.25.",
}


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


def _evaluate_scaled_actions(trace_df, scale: float, env) -> dict:
    raw_actions = trace_df["action"].to_numpy(dtype=np.float64)
    scaled_actions = np.clip(raw_actions * scale, -1.0, 1.0)
    scaled_trace = run_action_backtest(env, scaled_actions)
    metrics = compute_performance_metrics(scaled_trace)
    flat_001 = compute_action_bucket_ratios(scaled_actions, 0.01)
    flat_005 = compute_action_bucket_ratios(scaled_actions, 0.05)
    flat_010 = compute_action_bucket_ratios(scaled_actions, 0.10)
    gross_pnl_sum = float(np.sum(scaled_trace["position"] * scaled_trace["market_log_return"]))
    transaction_cost_sum = float(np.sum(scaled_trace["transaction_cost"]))
    net_pnl_sum = float(np.sum(scaled_trace["pnl"]))
    return {
        "scale": float(scale),
        "deterministic_return": metrics["total_return"],
        "final_equity": metrics["final_equity"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "flat_ratio_001": flat_001["flat_ratio"],
        "flat_ratio_005": flat_005["flat_ratio"],
        "flat_ratio_010": flat_010["flat_ratio"],
        "avg_abs_action": float(np.mean(np.abs(raw_actions))),
        "avg_abs_scaled_action": float(np.mean(np.abs(scaled_actions))),
        "turnover": metrics["turnover"],
        "transaction_cost_sum": transaction_cost_sum,
        "gross_pnl_sum": gross_pnl_sum,
        "net_pnl_sum": net_pnl_sum,
    }


def _build_report(
    *,
    asset: str,
    feature_preset: str,
    diagnostics: dict,
    scale_rows: list[dict],
    output_dir: Path,
) -> str:
    threshold_table = [
        "| Threshold | Flat % | Long % | Short % | Avg |Nonflat| | Nonflat Steps |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostics["threshold_sensitivity"]:
        threshold_table.append(
            f"| {row['threshold']:.3f} | {row['flat_ratio']*100:.1f} | {row['long_ratio']*100:.1f} | "
            f"{row['short_ratio']*100:.1f} | {row['avg_abs_action_nonflat']:.4f} | {row['num_nonflat_steps']} |"
        )

    scale_table = [
        "| Scale | Return | Sharpe | Max DD | Flat<=0.01 | Flat<=0.05 | Flat<=0.10 | Avg |Act| | Avg |Scaled| |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in scale_rows:
        scale_table.append(
            f"| {row['scale']:.1f} | {row['deterministic_return']:.4f} | {row['sharpe']:.2f} | "
            f"{row['max_drawdown']:.2%} | {row['flat_ratio_001']*100:.1f}% | {row['flat_ratio_005']*100:.1f}% | "
            f"{row['flat_ratio_010']*100:.1f}% | {row['avg_abs_action']:.4f} | {row['avg_abs_scaled_action']:.4f} |"
        )

    distribution = diagnostics["action_distribution"]
    directional = diagnostics["directional_signal"]
    histogram = [
        f"- {bucket['bucket']}: {bucket['count']}"
        for bucket in distribution["histogram_buckets"]
    ]
    return "\n".join(
        [
            "# Action Threshold / Mapping Audit",
            "",
            f"- Asset: {asset}",
            f"- Feature preset: {feature_preset}",
            f"- Output: {output_dir}",
            "",
            "## Action Flow Summary",
            *[f"- {key.replace('_', ' ').capitalize()}: {value}" for key, value in ACTION_FLOW_SUMMARY.items()],
            "",
            "## Action Distribution",
            f"- Mean/std: {distribution['action_mean']:.4f} / {distribution['action_std']:.4f}",
            f"- Min/max: {distribution['action_min']:.4f} / {distribution['action_max']:.4f}",
            f"- Abs mean/median: {distribution['action_abs_mean']:.4f} / {distribution['action_abs_median']:.4f}",
            f"- Abs p75/p90/p95/p99: {distribution['action_abs_p75']:.4f} / {distribution['action_abs_p90']:.4f} / {distribution['action_abs_p95']:.4f} / {distribution['action_abs_p99']:.4f}",
            f"- Near zero <=0.01 / <=0.05 / <=0.10: {distribution['near_zero_action_ratio_001']:.1%} / {distribution['near_zero_action_ratio_005']:.1%} / {distribution['near_zero_action_ratio_010']:.1%}",
            "### Histogram",
            *histogram,
            "",
            "## Threshold Sensitivity",
            *threshold_table,
            "",
            "## Scale Evaluation",
            *scale_table,
            "",
            "## Directional Signal Diagnostics",
            f"- Sign accuracy for |action| > 0.01: {directional['sign_accuracy_nonzero']:.3f}",
            f"- Mean next return when long: {directional['mean_next_return_when_long']:.5f}",
            f"- Mean next return when short: {directional['mean_next_return_when_short']:.5f}",
            f"- Mean action when next return positive: {directional['mean_action_when_next_return_positive']:.5f}",
            f"- Mean action when next return negative: {directional['mean_action_when_next_return_negative']:.5f}",
            f"- Action vs next return correlation: {directional['action_next_return_correlation']:.4f}",
            f"- |Action| vs |next return| correlation: {directional['abs_action_next_abs_return_correlation']:.4f}",
            "",
            "## Interpretation",
            f"- Default flat ratio currently uses threshold {diagnostics['summary']['flat_threshold_default']:.3f}.",
            f"- Continuous actions are used directly as positions during evaluation; flat/long/short buckets are diagnostic only.",
            "",
            "## Recommendation",
            "- If lower thresholds reveal many non-flat actions, the current 0.25 flat ratio is overstating neutrality.",
            "- If higher action scales improve metrics, the policy may contain directional information but undersized exposure.",
            "- If direction diagnostics remain weak, the model signal is likely still mostly noise.",
        ]
    )


def run_action_mapping_experiment(
    asset: str,
    config: dict,
    *,
    feature_preset: str,
    scales: list[float] | None = None,
    quick: bool = False,
):
    asset = normalize_asset_name(asset)
    scales = [float(scale) for scale in (scales or [1, 2, 3, 5])]
    validate_requested_feature_ablation_presets([feature_preset])
    config = apply_feature_preset_to_config(config, feature_preset)
    if quick:
        config["training"]["iterations"] = 2
        config["training"]["episode_length"] = 128
        config["training"]["rollout_steps"] = 128

    timestamp = utc_timestamp_slug()
    output_dir = EXPERIMENTS_DIR / "action_mapping" / f"{timestamp}_{asset}"
    ensure_dir(output_dir)

    dataset = build_processed_dataset(
        asset=asset,
        window_size=config["data"]["window_size"],
        train_split=config["data"]["train_split"],
        selected_features=config["features"]["selected"],
    )
    checkpoint_path = output_dir / "checkpoint.pth"
    final_checkpoint = output_dir / "final.pth"
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    train_asset(
        asset,
        config,
        best_checkpoint=str(checkpoint_path),
        final_checkpoint=str(final_checkpoint),
        run_dir=output_dir / "training",
        processed_dataset=dataset,
    )

    diagnostics = collect_model_diagnostics(
        asset,
        config=config,
        checkpoint=str(checkpoint_path),
        output_dir=output_dir / "diagnostics",
        save=True,
        processed_dataset=dataset,
    )

    env = build_eval_env(dataset.test_windows, dataset.test_price_windows, config)
    scale_rows = []
    for scale in scales:
        scale_rows.append(
            _evaluate_scaled_actions(diagnostics["trace_df"], scale, env)
        )

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scale_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scale_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(scale_rows, handle, indent=2)

    manifest = {
        "asset": asset,
        "experiment_type": "action-mapping",
        "timestamp": timestamp,
        "feature_preset": feature_preset,
        "scales": scales,
        "python_command": " ".join(sys.argv),
        "git_commit_if_available": get_git_commit(),
        "git_dirty_status_if_available": _git_dirty_status(),
        "action_flow_summary": ACTION_FLOW_SUMMARY,
        "checkpoint_path": str(checkpoint_path),
        "feature_count": len(config["features"]["selected"]),
        "selected_features": config["features"]["selected"],
    }
    with (output_dir / "audit_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    report = _build_report(
        asset=asset,
        feature_preset=feature_preset,
        diagnostics=diagnostics,
        scale_rows=scale_rows,
        output_dir=output_dir,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    best_by_sharpe = max(scale_rows, key=lambda row: row["sharpe"])["scale"]
    return {
        "experiment_dir": str(output_dir),
        "scale_rows": scale_rows,
        "diagnostics": diagnostics,
        "best_by_sharpe": best_by_sharpe,
    }
