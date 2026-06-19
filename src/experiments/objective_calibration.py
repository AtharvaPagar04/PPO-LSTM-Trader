import csv
import json
from pathlib import Path
import sys

from src.config.assets import normalize_asset_name
from src.config.paths import EXPERIMENTS_DIR, ensure_dir
from src.config.objective_presets import apply_objective_preset_to_config
from src.evaluation.diagnostics import collect_model_diagnostics
from src.evaluation.walk_forward import evaluate_walk_forward_asset
from src.features.pipeline import build_processed_dataset
from src.config.feature_ablation_presets import apply_feature_preset_to_config
from src.train import train_asset
from src.utils.logger import utc_timestamp_slug, get_git_commit

def calculate_calibration_score(
    rl_beat_constant_signed_mean_action_count,
    rl_beat_constant_abs_mean_short_count,
    rl_best_return_fold_count,
    walk_forward_mean_sharpe,
    deterministic_max_drawdown,
    action_mean
):
    score = (
        2.0 * rl_beat_constant_signed_mean_action_count
        + 1.5 * rl_beat_constant_abs_mean_short_count
        + 1.0 * rl_best_return_fold_count
        + 1.0 * walk_forward_mean_sharpe
        - 1.5 * deterministic_max_drawdown
        - 0.5 * abs(action_mean)
    )
    return score

def run_objective_calibration_experiment(asset: str, config: dict, presets: list[str], feature_preset: str = "price_action_minimal", quick: bool = False):
    asset = normalize_asset_name(asset)
    timestamp = utc_timestamp_slug()
    experiment_dir = EXPERIMENTS_DIR / "objective_calibration" / f"{timestamp}_{asset}"
    ensure_dir(experiment_dir)
    
    summary_rows = []
    manifest_presets = []
    
    for preset in presets:
        preset_dir = experiment_dir / preset
        ensure_dir(preset_dir)
        
        # Apply preset
        preset_config = apply_objective_preset_to_config(config, preset)
        # Apply feature preset
        preset_config = apply_feature_preset_to_config(preset_config, feature_preset)

        # Safety validation: non-current presets must have at least one active coefficient
        if preset != "current":
            objective_keys = ["exposure_penalty_coef", "turnover_penalty_coef", "directional_reward_coef", "volatility_exposure_penalty_coef"]
            active = {k: preset_config["environment"].get(k, 0.0) for k in objective_keys if preset_config["environment"].get(k, 0.0) != 0.0}
            if not active:
                raise ValueError(
                    f"Objective preset '{preset}' resolved to zero active coefficients. "
                    f"Check preset wiring. Keys checked: {objective_keys}"
                )
        
        if quick:
            preset_config["training"]["iterations"] = 2
            preset_config["training"]["episode_length"] = 128
            preset_config["training"]["rollout_steps"] = 128
        
        with (preset_dir / "config.yaml").open("w", encoding="utf-8") as f:
            json.dump(preset_config, f, indent=2)
            
        dataset = build_processed_dataset(
            asset=asset,
            window_size=preset_config["data"]["window_size"],
            train_split=preset_config["data"]["train_split"],
            selected_features=preset_config["features"]["selected"]
        )
        
        checkpoint_path = preset_dir / "checkpoint.pth"
        
        print(f"\n--- Running preset: {preset} ---")
        train_asset(
            asset,
            preset_config,
            best_checkpoint=str(checkpoint_path),
            final_checkpoint=str(preset_dir / "final.pth"),
            run_dir=preset_dir / "training",
            processed_dataset=dataset,
        )
        
        # Diagnostics
        diagnostics_dir = preset_dir / "diagnostics"
        ensure_dir(diagnostics_dir)
        diagnostics = collect_model_diagnostics(
            asset,
            config=preset_config,
            checkpoint=checkpoint_path,
            output_dir=diagnostics_dir,
            save=True,
            processed_dataset=dataset,
        )
        
        # Walk forward without baselines
        wf_result = evaluate_walk_forward_asset(
            asset,
            config=preset_config,
            checkpoint=str(checkpoint_path),
            folds=5,
            output_dir=preset_dir / "walk_forward",
            include_baselines=False,
            processed_dataset=dataset,
        )
        
        # Walk forward with baselines
        wf_baseline_result = evaluate_walk_forward_asset(
            asset,
            config=preset_config,
            checkpoint=str(checkpoint_path),
            folds=5,
            output_dir=preset_dir / "walk_forward_baselines",
            include_baselines=True,
            processed_dataset=dataset,
        )
        
        # Aggregate stats
        wf_agg = wf_result["aggregate"]
        wf_base = wf_baseline_result["baseline_aggregate"]
        diag = diagnostics["summary"]
        
        rl_beat_signed_mean = wf_base.get("rl_beat_constant_signed_mean_action_count", 0)
        rl_beat_abs_short = wf_base.get("rl_beat_constant_abs_mean_short_count", 0)
        rl_beat_abs_long = wf_base.get("rl_beat_constant_abs_mean_long_count", 0)
        
        score = calculate_calibration_score(
            rl_beat_signed_mean,
            rl_beat_abs_short,
            wf_base.get("rl_best_return_fold_count", 0),
            wf_agg["mean_sharpe"],
            diag["max_drawdown"],
            diag["action_mean"]
        )
        
        row = {
            "preset": preset,
            "deterministic_return": diag["total_return"],
            "deterministic_sharpe": diag["sharpe"],
            "max_drawdown": diag["max_drawdown"],
            "walk_forward_mean_return": wf_agg["mean_total_return"],
            "walk_forward_mean_sharpe": wf_agg["mean_sharpe"],
            "walk_forward_positive_folds": wf_agg["positive_fold_count"],
            "rl_best_return_fold_count": wf_base.get("rl_best_return_fold_count", 0),
            
            "flat_ratio_001": diag["flat_ratio_001"],
            "flat_ratio_005": diag["flat_ratio_005"],
            "flat_ratio_010": diag["flat_ratio_010"],
            "flat_ratio_025": diag["flat_ratio_025"],
            "dominant_action_side": diag["dominant_action_side"],
            "action_mean": diag["action_mean"],
            "action_abs_mean": diag["action_abs_mean"],
            "positive_action_ratio": diag["positive_action_ratio"],
            "negative_action_ratio": diag["negative_action_ratio"],
            "turnover": diag["turnover"],
            "transaction_cost_sum": diag.get("transaction_cost_sum", 0.0),
            
            "rl_beat_constant_signed_mean_action_count": rl_beat_signed_mean,
            "rl_beat_constant_abs_mean_short_count": rl_beat_abs_short,
            "rl_beat_constant_abs_mean_long_count": rl_beat_abs_long,
            "constant_signed_mean_action_mean_return": wf_base.get("constant_signed_mean_action_mean_return", 0.0),
            "constant_abs_mean_short_mean_return": wf_base.get("constant_abs_mean_short_mean_return", 0.0),
            "constant_abs_mean_long_mean_return": wf_base.get("constant_abs_mean_long_mean_return", 0.0),
            
            "mean_pnl_component": diag.get("mean_pnl_component", 0.0),
            "mean_transaction_cost_component": diag.get("mean_transaction_cost_component", 0.0),
            "mean_exposure_penalty_component": diag.get("mean_exposure_penalty_component", 0.0),
            "mean_directional_reward_component": diag.get("mean_directional_reward_component", 0.0),
            "mean_volatility_exposure_penalty_component": diag.get("mean_volatility_exposure_penalty_component", 0.0),
            "mean_total_reward": diag.get("mean_total_reward", 0.0),
            
            "score": score,
        }
        summary_rows.append(row)
        
        manifest_presets.append({
            "preset_name": preset,
            "resolved_config": preset_config,
            "checkpoint_path": str(checkpoint_path),
        })
        
    with (experiment_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
        
    with (experiment_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    best_by_score = max(summary_rows, key=lambda x: x["score"])["preset"]
    best_by_sharpe = max(summary_rows, key=lambda x: x["walk_forward_mean_sharpe"])["preset"]
    
    def baseline_wins(x):
        return x["rl_beat_constant_signed_mean_action_count"] + x["rl_beat_constant_abs_mean_short_count"] + x["rl_beat_constant_abs_mean_long_count"]
    
    best_by_wins = max(summary_rows, key=baseline_wins)["preset"]

    report_lines = [
        f"# Objective / Action Calibration Experiment",
        "",
        "## Experiment Setup",
        f"- Asset: {asset}",
        f"- Feature Preset: {feature_preset}",
        f"- Presets: {', '.join(presets)}",
        "",
        "This is an offline research experiment. It does not execute trades and does not prove live trading profitability.",
        "",
        "## Summary Table",
        "",
        "| Preset | Return | WF Sharpe | RL Best Fold | Flat@10% | AbsAction | Beat Signed | Beat S-Short | Score |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    
    for row in summary_rows:
        report_lines.append(
            f"| {row['preset']} | {row['deterministic_return']:.4f} | "
            f"{row['walk_forward_mean_sharpe']:.2f} | "
            f"{row['rl_best_return_fold_count']}/5 | {row['flat_ratio_010']*100:.1f}% | "
            f"{row['action_abs_mean']:.4f} | {row['rl_beat_constant_signed_mean_action_count']}/5 | "
            f"{row['rl_beat_constant_abs_mean_short_count']}/5 | {row['score']:.4f} |"
        )
        
    report_lines.extend([
        "",
        "## Corrected Action Behavior",
        "",
        "| Preset | Side | Act Mean | AbsAction | Flat@1% | Flat@5% | Flat@10% | Flat@25% |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for row in summary_rows:
        report_lines.append(
            f"| {row['preset']} | {row['dominant_action_side']} | {row['action_mean']:.4f} | {row['action_abs_mean']:.4f} | "
            f"{row['flat_ratio_001']*100:.1f}% | {row['flat_ratio_005']*100:.1f}% | {row['flat_ratio_010']*100:.1f}% | {row['flat_ratio_025']*100:.1f}% |"
        )

    report_lines.extend([
        "",
        "## Exposure-Equivalent Baseline Comparison",
        "",
        "| Preset | Beat Signed | Beat S-Short | Beat S-Long |",
        "|---|---|---|---|",
    ])
    for row in summary_rows:
        report_lines.append(
            f"| {row['preset']} | {row['rl_beat_constant_signed_mean_action_count']}/5 | "
            f"{row['rl_beat_constant_abs_mean_short_count']}/5 | {row['rl_beat_constant_abs_mean_long_count']}/5 |"
        )

    report_lines.extend([
        "",
        "## Reward Component Breakdown",
        "",
        "| Preset | Total R | PnL | Tx Cost | Exposure Pen | Directional R | Vol Pen |",
        "|---|---|---|---|---|---|---|",
    ])
    for row in summary_rows:
        report_lines.append(
            f"| {row['preset']} | {row['mean_total_reward']:.4f} | {row['mean_pnl_component']:.4f} | "
            f"{row['mean_transaction_cost_component']:.4f} | {row['mean_exposure_penalty_component']:.4f} | "
            f"{row['mean_directional_reward_component']:.4f} | {row['mean_volatility_exposure_penalty_component']:.4f} |"
        )

    report_lines.extend([
        "",
        "## Best Presets",
        f"- Best by score: {best_by_score}",
        f"- Best by walk-forward Sharpe: {best_by_sharpe}",
        f"- Best by baseline wins: {best_by_wins}",
        "",
        "## Interpretation",
        "This score is a research comparison helper, not proof of profitability.",
        "",
        "## Recommendation",
        "If a preset beats constant_signed_mean_action in most folds, mark it as candidate for repeated-seed validation.",
        "If a preset only increases return by increasing static exposure, reject it as exposure bias.",
        "If exposure penalty reduces action magnitude but hurts all returns, do not use it.",
        "If directional reward improves timing baselines, run seed-validation on that preset.",
        "If no preset improves exposure-equivalent baselines, move to new directional/cross-asset features."
    ])
    
    with (experiment_dir / "report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    manifest = {
        "asset": asset,
        "experiment_type": "objective_calibration",
        "timestamp": timestamp,
        "presets": presets,
        "feature_preset": feature_preset,
        "base_config_path": str(config),
        "python_command": " ".join(sys.argv),
        "git_commit_if_available": get_git_commit(),
        "git_dirty_status_if_available": None,
        "presets_data": manifest_presets
    }
    with (experiment_dir / "audit_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return {
        "experiment_dir": str(experiment_dir),
        "best_by_score": best_by_score,
        "best_by_sharpe": best_by_sharpe,
        "best_by_wins": best_by_wins,
        "summary": summary_rows
    }
