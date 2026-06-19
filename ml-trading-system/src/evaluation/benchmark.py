import csv
from pathlib import Path

import torch

from src.config.assets import asset_to_symbol, normalize_asset_name
from src.config.paths import EVALUATION_DIR, ensure_dir, resolve_checkpoint_path
from src.data.dataset import load_metadata, load_processed_data
from src.evaluation.backtest import run_policy_backtest
from src.evaluation.baselines import run_baselines
from src.evaluation.metrics import compute_performance_metrics
from src.evaluation.plot import plot_equity_curves
from src.evaluation.utils import write_trace_csv
from src.models.policy import LSTMPolicy
from src.utils.logger import write_json
from src.utils.seed import set_global_seed


def _remap_legacy_state_dict_keys(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("lstm."):
            remapped[key.replace("lstm.", "encoder.lstm.", 1)] = value
        elif key.startswith("actor_mean."):
            remapped[key.replace("actor_mean.", "actor.actor_mean.", 1)] = value
        elif key.startswith("actor_std."):
            remapped[key.replace("actor_std.", "actor.actor_std.", 1)] = value
        elif key.startswith("critic.") and not key.startswith("critic.value."):
            remapped[key.replace("critic.", "critic.value.", 1)] = value
        else:
            remapped[key] = value
    return remapped


def load_policy_from_checkpoint(asset, checkpoint_path, input_dim, config, device):
    ppo_cfg = config.get("ppo", {})
    model = LSTMPolicy(
        input_dim=input_dim,
        hidden_dim=config["model"]["hidden_size"],
        lstm_layers=config["model"]["lstm_layers"],
        dropout=config["model"]["dropout"],
        log_std_min=ppo_cfg.get("log_std_min", -1.5),
        log_std_max=ppo_cfg.get("log_std_max", -0.2),
        std_parameterization=ppo_cfg.get("std_parameterization", "hard_clamp"),
    ).to(device)
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
    state_dict = _remap_legacy_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_eval_env(feature_windows, price_windows, config):
    from src.env.trading_env import TradingEnv

    env_cfg = config["environment"]
    return TradingEnv(
        feature_windows,
        price_windows,
        cost=env_cfg["transaction_cost"],
        max_steps=config["training"]["episode_length"],
        drawdown_penalty=env_cfg["drawdown_penalty"],
        position_penalty=env_cfg["position_penalty"],
        action_change_penalty=env_cfg["action_change_penalty"],
        reward_scale=env_cfg["reward_scale"],
        reward_clip=env_cfg["reward_clip"],
    )


def evaluate_asset(asset, config, checkpoint=None, output_dir=None, model=None):
    asset = normalize_asset_name(asset)
    set_global_seed(config["evaluation"]["random_seed"])

    test_X, test_price = load_processed_data(asset, "test")
    metadata = load_metadata(asset)
    env = build_eval_env(test_X, test_price, config)

    if model is None:
        checkpoint_path = resolve_checkpoint_path(asset, checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_policy_from_checkpoint(
            asset, checkpoint_path, test_X.shape[2], config, device
        )
    else:
        checkpoint_path = Path(checkpoint) if checkpoint else None

    rl_trace = run_policy_backtest(env, model, deterministic_policy=True)
    baselines = run_baselines(
        test_price,
        transaction_cost=config["environment"]["transaction_cost"],
        seed=config["evaluation"]["random_seed"],
    )
    rl_metrics = compute_performance_metrics(rl_trace)
    baseline_metrics = {
        name: compute_performance_metrics(trace) for name, trace in baselines.items()
    }

    result = {
        "asset": asset,
        "binance_symbol": asset_to_symbol(asset),
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "metadata": metadata,
        "rl_policy": rl_metrics,
        "baselines": baseline_metrics,
    }

    if output_dir is None:
        output_dir = EVALUATION_DIR / asset
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    write_json(output_dir / "metrics.json", result)
    write_trace_csv(output_dir / "actions.csv", "action", rl_trace["action"])
    write_trace_csv(output_dir / "positions.csv", "position", rl_trace["position"])
    plot_equity_curves(
        {
            "RL": rl_trace["equity"],
            "Always Long": baselines["always_long"]["equity"],
            "Always Short": baselines["always_short"]["equity"],
            "Always Flat": baselines["always_flat"]["equity"],
            "Random": baselines["random"]["equity"],
            "Buy and Hold": baselines["buy_and_hold"]["equity"],
        },
        output_path=output_dir / "equity_curve.png",
        title=f"{asset} Full-Period Evaluation",
    )
    return result, rl_trace, baselines


def write_summary(results, output_dir=None):
    output_dir = Path(output_dir or EVALUATION_DIR)
    ensure_dir(output_dir)
    summary_rows = []
    for result in results:
        row = {
            "asset": result["asset"],
            "checkpoint": result["checkpoint"],
            "rl_final_equity": result["rl_policy"]["final_equity"],
            "rl_sharpe": result["rl_policy"]["sharpe"],
            "rl_max_drawdown": result["rl_policy"]["max_drawdown"],
            "always_long_final_equity": result["baselines"]["always_long"]["final_equity"],
            "always_short_final_equity": result["baselines"]["always_short"]["final_equity"],
            "always_flat_final_equity": result["baselines"]["always_flat"]["final_equity"],
            "random_final_equity": result["baselines"]["random"]["final_equity"],
            "buy_and_hold_final_equity": result["baselines"]["buy_and_hold"]["final_equity"],
        }
        summary_rows.append(row)

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["asset"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    write_json(output_dir / "summary.json", {"results": results})
