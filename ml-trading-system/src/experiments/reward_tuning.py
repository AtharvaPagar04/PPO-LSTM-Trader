import csv
import json
from pathlib import Path

from src.config.assets import normalize_asset_name
from src.config.paths import EXPERIMENTS_DIR, ensure_dir
from src.config.reward_presets import apply_reward_preset_to_config
from src.evaluation.diagnostics import collect_model_diagnostics
from src.evaluation.walk_forward import evaluate_walk_forward_asset
from src.train import train_asset
from src.utils.logger import utc_timestamp_slug, get_git_commit
import sys
import os

def calculate_score(rl_best_return_fold_count, rl_beat_always_flat_count, rl_beat_random_count, robustness, mean_sharpe, max_drawdown):
    score = (
        2.0 * rl_best_return_fold_count
        + 1.0 * rl_beat_always_flat_count
        + 1.0 * rl_beat_random_count
        + 1.0 * robustness
        + 1.0 * mean_sharpe
        - 2.0 * max_drawdown
    )
    return score

def run_reward_experiment(asset: str, config: dict, presets: list[str], quick: bool = False):
    asset = normalize_asset_name(asset)
    timestamp = utc_timestamp_slug()
    experiment_dir = EXPERIMENTS_DIR / "reward_tuning" / f"{timestamp}_{asset}"
    ensure_dir(experiment_dir)
    
    summary_rows = []
    manifest_presets = []
    
    for preset in presets:
        preset_dir = experiment_dir / preset
        ensure_dir(preset_dir)
        
        # Apply preset
        preset_config = apply_reward_preset_to_config(config, preset)
        if quick:
            preset_config["training"]["iterations"] = 2
            preset_config["training"]["episode_length"] = 128
            preset_config["training"]["rollout_steps"] = 128
        
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
        
        # Diagnostics
        diagnostics_dir = preset_dir / "diagnostics"
        ensure_dir(diagnostics_dir)
        diagnostics = collect_model_diagnostics(
            asset,
            config=preset_config,
            checkpoint=checkpoint_path,
            output_dir=diagnostics_dir,
            save=True,
        )
        
        # Walk forward without baselines
        wf_result = evaluate_walk_forward_asset(
            asset,
            config=preset_config,
            checkpoint=str(checkpoint_path),
            folds=5,
            output_dir=preset_dir / "walk_forward",
            include_baselines=False
        )
        
        # Walk forward with baselines
        wf_baseline_result = evaluate_walk_forward_asset(
            asset,
            config=preset_config,
            checkpoint=str(checkpoint_path),
            folds=5,
            output_dir=preset_dir / "walk_forward_baselines",
            include_baselines=True
        )
        
        # Aggregate stats
        wf_agg = wf_result["aggregate"]
        wf_base = wf_baseline_result["baseline_aggregate"]
        diag = diagnostics["summary"]
        
        rl_final_equity = diag["final_equity"]
        rl_total_return = diag["total_return"]
        rl_sharpe = diag["sharpe"]
        rl_mdd = diag["max_drawdown"]
        
        rl_best_return_fold_count = wf_base["rl_best_return_fold_count"]
        rl_beat_always_long = wf_base["rl_beat_always_long_count"]
        rl_beat_always_short = wf_base["rl_beat_always_short_count"]
        rl_beat_always_flat = wf_base["rl_beat_always_flat_count"]
        rl_beat_random = wf_base["rl_beat_random_count"]
        
        score = calculate_score(
            rl_best_return_fold_count,
            rl_beat_always_flat,
            rl_beat_random,
            wf_agg["robustness_score"],
            wf_agg["mean_sharpe"],
            rl_mdd
        )
        
        row = {
            "asset": asset,
            "preset": preset,
            "checkpoint_path": str(checkpoint_path),
            "final_equity": rl_final_equity,
            "deterministic_return": rl_total_return,
            "deterministic_sharpe": rl_sharpe,
            "max_drawdown": rl_mdd,
            "walk_forward_mean_return": wf_agg["mean_total_return"],
            "walk_forward_mean_sharpe": wf_agg["mean_sharpe"],
            "walk_forward_worst_drawdown": wf_agg["worst_max_drawdown"],
            "walk_forward_positive_folds": wf_agg["positive_fold_count"],
            "walk_forward_robustness": wf_agg["robustness_score"],
            "baseline_rl_best_count": rl_best_return_fold_count,
            "rl_beat_always_long_count": rl_beat_always_long,
            "rl_beat_always_short_count": rl_beat_always_short,
            "rl_beat_always_flat_count": rl_beat_always_flat,
            "rl_beat_random_count": rl_beat_random,
            "action_mean": diag["action_mean"],
            "average_abs_action": diag["average_abs_action"],
            "flat_ratio": diag["flat_ratio"],
            "long_ratio": diag["long_ratio"],
            "short_ratio": diag["short_ratio"],
            "policy_std_mean": diag["policy_std_mean"],
            "turnover": diag["turnover"],
            "transaction_cost_sum": diag.get("transaction_cost_sum", 0.0),
            "position_penalty_sum": diag.get("position_penalty_sum", 0.0),
            "drawdown_penalty_sum": diag.get("drawdown_penalty_sum", 0.0),
            "action_change_penalty_sum": diag.get("action_change_penalty_sum", 0.0),
            "reward_clip_ratio": diag.get("reward_clip_ratio", 0.0),
            "evaluation_mode": "deterministic_full_period",
            "score": score
        }
        summary_rows.append(row)
        
        ppo_cfg = preset_config.get("ppo", {})
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
            "reward_config": preset_config.get("environment", {}),
            "train_iterations": preset_config["training"]["iterations"],
            "early_stopping_triggered": False,
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
        return x["rl_beat_always_long_count"] + x["rl_beat_always_short_count"] + x["rl_beat_always_flat_count"] + x["rl_beat_random_count"]
    
    best_by_wins = max(summary_rows, key=baseline_wins)["preset"]
    best_by_low_dd = min(summary_rows, key=lambda x: x["max_drawdown"])["preset"]

    report_lines = [
        f"# Reward Tuning Experiment: {asset}",
        "",
        "This is an offline research experiment. It does not execute trades and does not prove live trading profitability.",
        "",
        "## Results",
        "",
        "| Preset | Return | Sharpe | WF Ret | WF Sharpe | RL Best | Flat % | Avg |Act| | PolStd | Score |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    
    for row in summary_rows:
        report_lines.append(
            f"| {row['preset']} | {row['deterministic_return']:.4f} | {row['deterministic_sharpe']:.2f} | "
            f"{row['walk_forward_mean_return']:.4f} | {row['walk_forward_mean_sharpe']:.2f} | "
            f"{row['baseline_rl_best_count']}/5 | {row['flat_ratio']*100:.1f}% | "
            f"{row['average_abs_action']:.4f} | {row['policy_std_mean']:.4f} | {row['score']:.4f} |"
        )
        
    report_lines.extend([
        "",
        f"- Best by score: {best_by_score}",
        f"- Best by walk-forward Sharpe: {best_by_sharpe}",
        f"- Best by baseline wins: {best_by_wins}",
        f"- Best by low drawdown: {best_by_low_dd}",
    ])
    
    with (experiment_dir / "report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nReward tuning experiment: {asset}\n")
    print(f"{'Preset':<25} {'Return':<8} {'Sharpe':<8} {'WF Ret':<8} {'WF Sharpe':<9} {'RL Best':<7} {'Flat %':<6} {'Avg |Act|':<9} {'PolStd':<7} {'Score'}")
    for row in summary_rows:
        print(f"{row['preset']:<25} {row['deterministic_return']:<8.4f} {row['deterministic_sharpe']:<8.2f} {row['walk_forward_mean_return']:<8.4f} "
              f"{row['walk_forward_mean_sharpe']:<9.2f} {row['baseline_rl_best_count']}/5     {row['flat_ratio']*100:<6.1f} "
              f"{row['average_abs_action']:<9.4f} {row['policy_std_mean']:<7.4f} {row['score']:.4f}")
              
    print(f"\nBest by score: {best_by_score}")
    print(f"Best by walk-forward Sharpe: {best_by_sharpe}")
    print(f"Best by baseline wins: {best_by_wins}")
    print(f"\nOutput: {experiment_dir}")

    return {
        "experiment_dir": str(experiment_dir),
        "best_by_score": best_by_score,
        "best_by_sharpe": best_by_sharpe,
        "best_by_wins": best_by_wins,
        "summary": summary_rows
    }

    manifest = {
        "asset": asset,
        "experiment_type": "reward",
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
