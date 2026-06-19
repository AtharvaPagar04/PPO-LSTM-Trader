import csv
import json
from pathlib import Path

from src.config.assets import normalize_asset_name
from src.config.paths import EXPERIMENTS_DIR, ensure_dir
from src.config.ppo_std_presets import apply_ppo_std_preset_to_config
from src.evaluation.diagnostics import collect_model_diagnostics
from src.evaluation.walk_forward import evaluate_walk_forward_asset
from src.train import train_asset
from src.utils.logger import utc_timestamp_slug, get_git_commit
import sys
import os


def _load_training_trace_summary(training_dir: Path) -> dict:
    trace_path = training_dir / "training_trace.json"
    if not trace_path.exists():
        return {}
    with trace_path.open("r", encoding="utf-8") as handle:
        trace_rows = json.load(handle).get("trace", [])
    if not trace_rows:
        return {}
    first = trace_rows[0]
    last = trace_rows[-1]
    return {
        "policy_std_start": first.get("policy_std_mean", 0.0),
        "policy_std_end": last.get("policy_std_mean", 0.0),
        "policy_std_change": abs(
            last.get("policy_std_mean", 0.0) - first.get("policy_std_mean", 0.0)
        ),
        "raw_log_std_mean": last.get("raw_log_std_mean", 0.0),
        "raw_log_std_min": last.get("raw_log_std_min", 0.0),
        "raw_log_std_max": last.get("raw_log_std_max", 0.0),
        "log_std_mean": last.get("log_std_mean", 0.0),
        "log_std_min": last.get("log_std_min", 0.0),
        "log_std_max": last.get("log_std_max", 0.0),
        "std_high_saturation_ratio": last.get("std_high_saturation_ratio", 0.0),
        "std_low_saturation_ratio": last.get("std_low_saturation_ratio", 0.0),
        "std_parameterization": last.get("std_parameterization", "hard_clamp"),
    }

def calculate_score(rl_best_return_fold_count, rl_beat_always_flat_count, rl_beat_random_count, robustness, mean_sharpe, max_drawdown, flat_ratio):
    score = (
        2.0 * rl_best_return_fold_count
        + 1.0 * rl_beat_always_flat_count
        + 1.0 * rl_beat_random_count
        + 1.0 * robustness
        + 1.0 * mean_sharpe
        - 2.0 * max_drawdown
        - 0.5 * flat_ratio
    )
    return score

def run_ppo_std_experiment(asset: str, config: dict, presets: list[str], quick: bool = False, disable_early_stopping: bool = False):
    asset = normalize_asset_name(asset)
    timestamp = utc_timestamp_slug()
    experiment_dir = EXPERIMENTS_DIR / "ppo_std_tuning" / f"{timestamp}_{asset}"
    ensure_dir(experiment_dir)
    
    summary_rows = []
    manifest_presets = []
    
    for preset in presets:
        preset_dir = experiment_dir / preset
        ensure_dir(preset_dir)
        
        preset_config = apply_ppo_std_preset_to_config(config, preset)
        if quick:
            preset_config["training"]["iterations"] = 2
            preset_config["training"]["episode_length"] = 128
            preset_config["training"]["rollout_steps"] = 128
            
        if disable_early_stopping:
            preset_config["training"]["early_stopping_patience"] = 999999
            
        with (preset_dir / "config.yaml").open("w", encoding="utf-8") as f:
            json.dump(preset_config, f, indent=2)
            
        checkpoint_path = preset_dir / "checkpoint.pth"
        
        print(f"\n--- Running preset: {preset} ---")
        train_asset(
            asset,
            preset_config,
            best_checkpoint=str(checkpoint_path),
            final_checkpoint=str(preset_dir / "final.pth"),
            run_dir=preset_dir / "training"
        )
        training_trace_summary = _load_training_trace_summary(preset_dir / "training")
        
        diagnostics_dir = preset_dir / "diagnostics"
        ensure_dir(diagnostics_dir)
        diagnostics = collect_model_diagnostics(
            asset,
            config=preset_config,
            checkpoint=str(checkpoint_path),
            output_dir=diagnostics_dir,
            save=True,
        )
        
        wf_result = evaluate_walk_forward_asset(
            asset,
            config=preset_config,
            checkpoint=str(checkpoint_path),
            folds=5,
            output_dir=preset_dir / "walk_forward",
            include_baselines=False
        )
        
        wf_baseline_result = evaluate_walk_forward_asset(
            asset,
            config=preset_config,
            checkpoint=str(checkpoint_path),
            folds=5,
            output_dir=preset_dir / "walk_forward_baselines",
            include_baselines=True
        )
        
        wf_agg = wf_result["aggregate"]
        wf_base = wf_baseline_result["baseline_aggregate"]
        diag = diagnostics["summary"]
        
        ppo_cfg = preset_config.get("ppo", {})
        
        score = calculate_score(
            wf_base["rl_best_return_fold_count"],
            wf_base["rl_beat_always_flat_count"],
            wf_base["rl_beat_random_count"],
            wf_agg["robustness_score"],
            wf_agg["mean_sharpe"],
            diag["max_drawdown"],
            diag["flat_ratio"]
        )
        
        row = {
            "asset": asset,
            "preset": preset,
            "checkpoint_path": str(checkpoint_path),
            "entropy_coef": ppo_cfg.get("entropy_coef", 0.0),
            "std_penalty_coef": ppo_cfg.get("std_penalty_coef", 0.0),
            "std_target": ppo_cfg.get("std_target", 0.0),
            "log_std_min": ppo_cfg.get("log_std_min", 0.0),
            "log_std_max": ppo_cfg.get("log_std_max", 0.0),
            "std_parameterization": ppo_cfg.get("std_parameterization", "hard_clamp"),
            "final_equity": diag["final_equity"],
            "deterministic_return": diag["total_return"],
            "deterministic_sharpe": diag["sharpe"],
            "max_drawdown": diag["max_drawdown"],
            "walk_forward_mean_return": wf_agg["mean_total_return"],
            "walk_forward_mean_sharpe": wf_agg["mean_sharpe"],
            "walk_forward_worst_drawdown": wf_agg["worst_max_drawdown"],
            "walk_forward_positive_folds": wf_agg["positive_fold_count"],
            "walk_forward_robustness": wf_agg["robustness_score"],
            "baseline_rl_best_count": wf_base["rl_best_return_fold_count"],
            "rl_beat_always_long_count": wf_base["rl_beat_always_long_count"],
            "rl_beat_always_short_count": wf_base["rl_beat_always_short_count"],
            "rl_beat_always_flat_count": wf_base["rl_beat_always_flat_count"],
            "rl_beat_random_count": wf_base["rl_beat_random_count"],
            "action_mean": diag["action_mean"],
            "average_abs_action": diag["average_abs_action"],
            "flat_ratio": diag["flat_ratio"],
            "long_ratio": diag["long_ratio"],
            "short_ratio": diag["short_ratio"],
            "policy_std_mean": diag["policy_std_mean"],
            "policy_std_min": diag["policy_std_min"],
            "policy_std_max": diag["policy_std_max"],
            "policy_std_start": training_trace_summary.get("policy_std_start", 0.0),
            "policy_std_end": training_trace_summary.get("policy_std_end", 0.0),
            "policy_std_change": training_trace_summary.get("policy_std_change", 0.0),
            "raw_log_std_mean": training_trace_summary.get("raw_log_std_mean", 0.0),
            "raw_log_std_min": training_trace_summary.get("raw_log_std_min", 0.0),
            "raw_log_std_max": training_trace_summary.get("raw_log_std_max", 0.0),
            "training_log_std_mean": training_trace_summary.get("log_std_mean", 0.0),
            "training_log_std_min": training_trace_summary.get("log_std_min", 0.0),
            "training_log_std_max": training_trace_summary.get("log_std_max", 0.0),
            "std_high_saturation_ratio": training_trace_summary.get("std_high_saturation_ratio", 0.0),
            "std_low_saturation_ratio": training_trace_summary.get("std_low_saturation_ratio", 0.0),
            "turnover": diag["turnover"],
            "reward_clip_ratio": diag.get("reward_clip_ratio", 0.0),
            "evaluation_mode": "deterministic_full_period",
            "score": score
        }
        summary_rows.append(row)
        
        # Manifest data for this preset
        checkpoint_exists = checkpoint_path.exists()
        manifest_presets.append({
            "preset_name": preset,
            "resolved_config": preset_config,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_exists": checkpoint_exists,
            "checkpoint_mtime": checkpoint_path.stat().st_mtime if checkpoint_exists else None,
            "evaluation_checkpoint_used": str(checkpoint_path),
            "diagnostics_checkpoint_used": str(checkpoint_path),
            "walk_forward_checkpoint_used": str(checkpoint_path),
            "baseline_checkpoint_used": str(checkpoint_path),
            "log_std_min": ppo_cfg.get("log_std_min", -1.5),
            "log_std_max": ppo_cfg.get("log_std_max", -0.2),
            "entropy_coef": ppo_cfg.get("entropy_coef", 0.01),
            "std_penalty_coef": ppo_cfg.get("std_penalty_coef", 0.01),
            "std_target": ppo_cfg.get("std_target", 0.5),
            "std_parameterization": ppo_cfg.get("std_parameterization", "hard_clamp"),
            "reward_config": preset_config.get("environment", {}),
            "train_iterations": preset_config["training"]["iterations"],
            "early_stopping_triggered": False, # Will be set below if possible, but training logs contain it
        })
        
    with (experiment_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
        
    with (experiment_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    best_by_score = max(summary_rows, key=lambda x: x["score"])["preset"]
    best_by_policy_std_reduction = min(summary_rows, key=lambda x: x["policy_std_mean"])["preset"]
    best_by_flat_ratio_reduction = min(summary_rows, key=lambda x: x["flat_ratio"])["preset"]
    best_by_walk_forward_sharpe = max(summary_rows, key=lambda x: x["walk_forward_mean_sharpe"])["preset"]
    
    def baseline_wins(x):
        return x["rl_beat_always_long_count"] + x["rl_beat_always_short_count"] + x["rl_beat_always_flat_count"] + x["rl_beat_random_count"]
    best_by_baseline_wins = max(summary_rows, key=baseline_wins)["preset"]

    report_lines = [
        f"# PPO Std / Entropy Tuning Experiment: {asset}",
        "",
        "This is an offline research experiment. It does not execute trades and does not prove live trading profitability.",
        "",
        "## Results",
        "",
        "| Preset | Std Param | Std Start | Std End | Std d | Raw LogStd | High Sat | Flat % | Avg |Act| | Return | Sharpe | Score |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    
    for row in summary_rows:
        report_lines.append(
            f"| {row['preset']} | {row['std_parameterization']} | {row['policy_std_start']:.4f} | "
            f"{row['policy_std_end']:.4f} | {row['policy_std_change']:.6f} | {row['raw_log_std_mean']:.4f} | "
            f"{row['std_high_saturation_ratio']:.4f} | {row['flat_ratio']*100:.1f}% | {row['average_abs_action']:.4f} | "
            f"{row['deterministic_return']:.4f} | {row['deterministic_sharpe']:.2f} | {row['score']:.4f} |"
        )
        
    report_lines.extend([
        "",
        f"- Best by score: {best_by_score}",
        f"- Best by policy std reduction: {best_by_policy_std_reduction}",
        f"- Best by flat ratio reduction: {best_by_flat_ratio_reduction}",
        f"- Best by walk-forward Sharpe: {best_by_walk_forward_sharpe}",
        f"- Best by baseline wins: {best_by_baseline_wins}",
    ])
    
    with (experiment_dir / "report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nPPO Std / Entropy Tuning Experiment: {asset}\n")
    print(f"{'Preset':<20} {'Std Param':<13} {'Std Start':<10} {'Std End':<9} {'Std d':<9} {'Raw LogStd':<11} {'High Sat':<9} {'Flat %':<7} {'Avg|Act|':<9} {'Score'}")
    for row in summary_rows:
        print(
            f"{row['preset']:<20} {row['std_parameterization']:<13} {row['policy_std_start']:<10.4f} "
            f"{row['policy_std_end']:<9.4f} {row['policy_std_change']:<9.6f} {row['raw_log_std_mean']:<11.4f} "
            f"{row['std_high_saturation_ratio']:<9.4f} {row['flat_ratio']*100:<7.1f} {row['average_abs_action']:<9.4f} {row['score']:.4f}"
        )
              
    print(f"\nBest by score: {best_by_score}")
    print(f"Best by policy std reduction: {best_by_policy_std_reduction}")
    print(f"Best by flat ratio reduction: {best_by_flat_ratio_reduction}")
    print(f"Best by walk-forward Sharpe: {best_by_walk_forward_sharpe}")
    print(f"Best by baseline wins: {best_by_baseline_wins}")
    print(f"\nOutput: {experiment_dir}")

    result_dict = {
        "experiment_dir": str(experiment_dir),
        "best_by_score": best_by_score,
        "best_by_policy_std_reduction": best_by_policy_std_reduction,
        "best_by_flat_ratio_reduction": best_by_flat_ratio_reduction,
        "best_by_walk_forward_sharpe": best_by_walk_forward_sharpe,
        "best_by_baseline_wins": best_by_baseline_wins,
        "summary": summary_rows
    }
    
    manifest = {
        "asset": asset,
        "experiment_type": "ppo-std",
        "timestamp": timestamp,
        "presets": presets,
        "base_config_path": str(config),
        "python_command": " ".join(sys.argv),
        "git_commit_if_available": get_git_commit(),
        "git_dirty_status_if_available": None,
        "presets_data": manifest_presets
    }
    with (experiment_dir / "audit_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return result_dict
