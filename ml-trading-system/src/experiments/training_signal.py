import json
from pathlib import Path

from src.config.assets import normalize_asset_name
from src.config.paths import EXPERIMENTS_DIR, ensure_dir
from src.train import train_asset
from src.utils.logger import utc_timestamp_slug

def run_training_signal_experiment(asset: str, config: dict, quick: bool = False, disable_early_stopping: bool = False):
    asset = normalize_asset_name(asset)
    timestamp = utc_timestamp_slug()
    experiment_dir = EXPERIMENTS_DIR / "training_signal" / f"{timestamp}_{asset}"
    ensure_dir(experiment_dir)

    preset_config = config.copy()
    if quick:
        preset_config["training"]["iterations"] = 3
        preset_config["training"]["episode_length"] = 128
        preset_config["training"]["rollout_steps"] = 128

    if disable_early_stopping:
        preset_config["training"]["early_stopping_patience"] = 999999

    with (experiment_dir / "config.yaml").open("w", encoding="utf-8") as f:
        json.dump(preset_config, f, indent=2)

    checkpoint_path = experiment_dir / "checkpoint.pth"

    print(f"\n--- Running Training Signal Diagnostic: {asset} ---")
    train_asset(
        asset,
        preset_config,
        best_checkpoint=str(checkpoint_path),
        final_checkpoint=str(experiment_dir / "final.pth"),
        run_dir=experiment_dir / "training",
    )

    trace_json_path = experiment_dir / "training" / "training_trace.json"
    if not trace_json_path.exists():
        print("Training trace not found. Did training fail?")
        return

    with open(trace_json_path, "r", encoding="utf-8") as f:
        trace_data = json.load(f).get("trace", [])

    if len(trace_data) < 2:
        print("Not enough iterations to compute signal changes.")
        return

    first = trace_data[0]
    last = trace_data[-1]

    def avg(key):
        vals = [r.get(key, 0.0) for r in trace_data]
        return sum(vals) / len(vals)

    def min_val(key):
        return min([r.get(key, 0.0) for r in trace_data])

    actor_mean_abs_start = first.get("deterministic_action_abs_mean", 0.0)
    actor_mean_abs_end = last.get("deterministic_action_abs_mean", 0.0)
    actor_mean_abs_change = abs(actor_mean_abs_end - actor_mean_abs_start)

    policy_std_start = first.get("policy_std_mean", 0.0)
    policy_std_end = last.get("policy_std_mean", 0.0)
    policy_std_change = abs(policy_std_end - policy_std_start)
    raw_log_std_mean_start = first.get("raw_log_std_mean", 0.0)
    raw_log_std_mean_end = last.get("raw_log_std_mean", 0.0)
    std_high_saturation_ratio_start = first.get("std_high_saturation_ratio", 0.0)
    std_high_saturation_ratio_end = last.get("std_high_saturation_ratio", 0.0)

    raw_advantage_std_mean = avg("raw_advantage_std")
    raw_advantage_std_min = min_val("raw_advantage_std")
    normalized_advantage_std_mean = avg("normalized_advantage_std")

    returns_std_mean = avg("returns_std")
    td_delta_std_mean = avg("td_delta_std")

    actor_grad_norm_mean = avg("actor_grad_norm")
    critic_grad_norm_mean = avg("critic_grad_norm")
    total_grad_norm_mean = avg("total_grad_norm")

    approx_kl_mean = avg("approx_kl")
    clip_fraction_mean = avg("clip_fraction")
    ratio_std_mean = avg("ratio_std")

    value_loss_mean = avg("value_loss")
    policy_loss_mean = avg("policy_loss")
    entropy_mean = avg("entropy")
    explained_variance_mean = avg("explained_variance")

    actor_mean_stagnant = actor_mean_abs_change < 0.01
    raw_advantage_collapsed = raw_advantage_std_mean < 1e-6
    policy_updates_tiny = approx_kl_mean < 1e-5 and clip_fraction_mean < 0.01
    critic_unstable = value_loss_mean > 10.0 * returns_std_mean ** 2  # Heuristic
    value_scale_problem = avg("value_error_abs_mean") > returns_std_mean * 2

    summary = {
        "asset": asset,
        "num_iterations": len(trace_data),
        "actor_mean_abs_start": actor_mean_abs_start,
        "actor_mean_abs_end": actor_mean_abs_end,
        "actor_mean_abs_change": actor_mean_abs_change,
        "policy_std_start": policy_std_start,
        "policy_std_end": policy_std_end,
        "policy_std_change": policy_std_change,
        "raw_log_std_mean_start": raw_log_std_mean_start,
        "raw_log_std_mean_end": raw_log_std_mean_end,
        "std_high_saturation_ratio_start": std_high_saturation_ratio_start,
        "std_high_saturation_ratio_end": std_high_saturation_ratio_end,
        "std_parameterization": last.get("std_parameterization", "hard_clamp"),
        "raw_advantage_std_mean": raw_advantage_std_mean,
        "raw_advantage_std_min": raw_advantage_std_min,
        "normalized_advantage_std_mean": normalized_advantage_std_mean,
        "returns_std_mean": returns_std_mean,
        "td_delta_std_mean": td_delta_std_mean,
        "actor_grad_norm_mean": actor_grad_norm_mean,
        "critic_grad_norm_mean": critic_grad_norm_mean,
        "total_grad_norm_mean": total_grad_norm_mean,
        "approx_kl_mean": approx_kl_mean,
        "clip_fraction_mean": clip_fraction_mean,
        "ratio_std_mean": ratio_std_mean,
        "value_loss_mean": value_loss_mean,
        "policy_loss_mean": policy_loss_mean,
        "entropy_mean": entropy_mean,
        "explained_variance_mean": explained_variance_mean,
        "actor_mean_stagnant": actor_mean_stagnant,
        "raw_advantage_collapsed": raw_advantage_collapsed,
        "policy_updates_tiny": policy_updates_tiny,
        "critic_unstable": critic_unstable,
        "value_scale_problem": value_scale_problem,
    }

    with (experiment_dir / "signal_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_lines = [
        f"# Training Signal Diagnostics: {asset}",
        "",
        "## Summary",
        f"- Iterations: {len(trace_data)}",
        f"- Asset: {asset}",
        "",
        "## Actor Mean Learning",
        f"- Actor Mean Abs Start: {actor_mean_abs_start:.4f}",
        f"- Actor Mean Abs End: {actor_mean_abs_end:.4f}",
        f"- Actor Mean Abs Change: {actor_mean_abs_change:.4f}",
        "",
        "## Advantage Health",
        f"- Raw Advantage Std Mean: {raw_advantage_std_mean:.4f}",
        f"- Normalized Advantage Std Mean: {normalized_advantage_std_mean:.4f}",
        f"- TD Delta Std Mean: {td_delta_std_mean:.4f}",
        "",
        "## Std Audit",
        f"- Std Parameterization: {summary['std_parameterization']}",
        f"- Raw Log Std Mean Start: {raw_log_std_mean_start:.4f}",
        f"- Raw Log Std Mean End: {raw_log_std_mean_end:.4f}",
        f"- High Saturation Ratio Start: {std_high_saturation_ratio_start:.4f}",
        f"- High Saturation Ratio End: {std_high_saturation_ratio_end:.4f}",
        "",
        "## PPO Update Health",
        f"- Approx KL Mean: {approx_kl_mean:.6f}",
        f"- Clip Fraction Mean: {clip_fraction_mean:.6f}",
        f"- Actor Grad Norm Mean: {actor_grad_norm_mean:.4f}",
        "",
        "## Value Function Scale",
        f"- Returns Std Mean: {returns_std_mean:.4f}",
        f"- Value Loss Mean: {value_loss_mean:.4f}",
        f"- Explained Variance Mean: {explained_variance_mean:.4f}",
        "",
        "## Warnings",
    ]

    warnings = []
    if actor_mean_stagnant:
        warnings.append("- ⚠️ Actor mean appears stagnant: actor_mean_abs changed only < 0.01 across training.")
    if raw_advantage_collapsed:
        warnings.append("- ⚠️ Raw advantage collapsed: raw_advantage_std is near zero.")
    else:
        warnings.append("- ✅ Raw advantage std is healthy, normalized advantage is correctly scaled.")
    if policy_updates_tiny:
        warnings.append("- ⚠️ PPO updates appear tiny: approx_kl and clip_fraction are near zero.")
    if value_scale_problem:
        warnings.append("- ⚠️ Value function scale mismatch detected: value error is large relative to return std.")
    
    if not warnings:
        warnings.append("- None")
        
    report_lines.extend(warnings)
    report_lines.extend([
        "",
        "## Recommended Next Step",
    ])

    if actor_mean_stagnant and not raw_advantage_collapsed:
        report_lines.append("investigate actor learning rate, policy loss scale, gradient flow.")
    elif raw_advantage_collapsed:
        report_lines.append("audit reward scaling, GAE, episode returns.")
    elif policy_updates_tiny:
        report_lines.append("inspect PPO clip, learning rate, minibatch/update epochs.")
    elif value_scale_problem:
        report_lines.append("audit critic loss, reward normalization, value clipping.")
    else:
        report_lines.append("move to feature ablation.")

    with (experiment_dir / "report.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"\nTraining Signal Diagnostic: {asset}")
    for warning in warnings:
        print(warning)
    print(
        f"Std audit | parameterization={summary['std_parameterization']} | raw_log_std_mean={raw_log_std_mean_end:.4f} | high_sat={std_high_saturation_ratio_end:.4f} | policy_std_change={policy_std_change:.6f}"
    )
    print(f"\nOutput: {experiment_dir}")

    import shutil
    shutil.copy(trace_json_path.parent / "training_trace.csv", experiment_dir / "training_trace.csv")
    shutil.copy(trace_json_path, experiment_dir / "training_trace.json")

    return {"experiment_dir": str(experiment_dir), "summary": summary}
