import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import numpy as np

from src.config.assets import asset_to_symbol, normalize_asset_name
from src.config.paths import (
    final_checkpoint_path,
    best_checkpoint_path,
    ensure_dir,
    MODELS_DIR,
)
from src.config.settings import load_config
from src.data.dataset import load_metadata, load_processed_data
from src.evaluation.benchmark import evaluate_asset
from src.features.pipeline import ProcessedDataset
from src.models.policy import LSTMPolicy
from src.ppo.ppo_trainer import PPOTrainer
from src.utils.logger import create_run_dir, get_git_commit, write_csv, write_json
from src.utils.seed import set_global_seed


def build_env(feature_windows, price_windows, config):
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
        exposure_penalty_coef=env_cfg.get("exposure_penalty_coef", 0.0),
        turnover_penalty_coef=env_cfg.get("turnover_penalty_coef", 0.0),
        directional_reward_coef=env_cfg.get("directional_reward_coef", 0.0),
        volatility_exposure_penalty_coef=env_cfg.get("volatility_exposure_penalty_coef", 0.0),
    )


def build_model(input_dim, config, device):
    ppo_cfg = config.get("ppo", {})
    return LSTMPolicy(
        input_dim=input_dim,
        hidden_dim=config["model"]["hidden_size"],
        lstm_layers=config["model"]["lstm_layers"],
        dropout=config["model"]["dropout"],
        log_std_min=ppo_cfg.get("log_std_min", -1.5),
        log_std_max=ppo_cfg.get("log_std_max", -0.2),
        std_parameterization=ppo_cfg.get("std_parameterization", "hard_clamp"),
    ).to(device)


def save_checkpoint(path: Path, model, asset, config, metadata, checkpoint_type, extra=None):
    payload = {
        "state_dict": model.state_dict(),
        "asset": asset,
        "config": config,
        "metadata": metadata,
        "checkpoint_type": checkpoint_type,
    }
    if extra:
        payload.update(extra)
    ensure_dir(path.parent)
    torch.save(payload, path)


def save_legacy_state_dict(path: Path, model):
    ensure_dir(MODELS_DIR)
    torch.save(model.state_dict(), path)


def train_asset(
    asset: str,
    config: dict,
    best_checkpoint: str | None = None,
    run_dir: str | Path | None = None,
    final_checkpoint: str | None = None,
    processed_dataset: ProcessedDataset | None = None,
):
    asset = normalize_asset_name(asset)
    set_global_seed(config["training"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if processed_dataset is not None:
        train_X = processed_dataset.train_windows
        train_price = processed_dataset.train_price_windows
        test_X = processed_dataset.test_windows
        test_price = processed_dataset.test_price_windows
        metadata = processed_dataset.metadata
    else:
        train_X, train_price = load_processed_data(asset, "train")
        test_X, test_price = load_processed_data(asset, "test")
        metadata = load_metadata(asset)

    env = build_env(train_X, train_price, config)
    model = build_model(train_X.shape[2], config, device)
    ppo_cfg = config["ppo"]
    trainer = PPOTrainer(
        env,
        model,
        lr=config["training"]["learning_rate"],
        gamma=ppo_cfg["gamma"],
        lam=ppo_cfg["gae_lambda"],
        clip=ppo_cfg["clip_ratio"],
        entropy_coef=ppo_cfg["entropy_coef"],
        value_coef=ppo_cfg["value_coef"],
        std_penalty_coef=ppo_cfg["std_penalty_coef"],
        std_target=ppo_cfg.get("std_target", 0.5),
        max_grad_norm=ppo_cfg["max_grad_norm"],
    )

    if run_dir:
        run_dir = Path(run_dir)
        ensure_dir(run_dir)
    else:
        run_dir = create_run_dir(asset)
    best_path = Path(best_checkpoint) if best_checkpoint else best_checkpoint_path(asset)
    final_path = Path(final_checkpoint) if final_checkpoint else final_checkpoint_path(asset)

    run_config = {
        "asset": asset,
        "binance_symbol": asset_to_symbol(asset),
        "seed": config["training"]["seed"],
        "git_commit": get_git_commit(),
        "config": config,
        "data_paths": {
            "train": f"data/processed/{asset}_train_windows.npy",
            "test": f"data/processed/{asset}_test_windows.npy",
        },
        "model_paths": {"best": str(best_path), "final": str(final_path)},
        "train_dataset_shape": list(train_X.shape),
        "test_dataset_shape": list(test_X.shape),
        "feature_names": metadata["features"],
        "window_size": metadata["window_size"],
        "split_ratio": metadata["split_ratio"],
        "train_end_timestamp": metadata.get("train_end_timestamp"),
        "test_start_timestamp": metadata.get("test_start_timestamp"),
    }
    write_json(run_dir / "run_config.json", run_config)

    best_reward = -float("inf")
    no_improve = 0
    training_rows = []
    iterations = config["training"]["iterations"]
    patience = config["training"]["early_stopping_patience"]

    diagnostic_size = min(256, train_X.shape[0])
    diagnostic_X = torch.tensor(train_X[:diagnostic_size], dtype=torch.float32).to(device)
    prev_actor_mean = None

    for iteration in range(iterations):
        rollout = trainer.collect_rollout(steps=config["training"]["rollout_steps"])
        update_metrics = trainer.update(
            rollout,
            epochs=config["training"]["update_epochs"],
            batch_size=config["training"]["batch_size"],
        )

        total_reward = float(rollout["rewards"].sum())
        rolling_window = [row["episode_reward"] for row in training_rows[-9:]] + [total_reward]
        avg_reward = float(sum(rolling_window) / len(rolling_window))
        avg_position = float(rollout["actions"].mean())
        abs_position = float(np.abs(rollout["actions"]).mean())
        value_mean = float(rollout["values"].mean())
        value_std = float(rollout["values"].std())

        std_mean = update_metrics.get("policy_std_mean", 0.0)

        with torch.no_grad():
            diag_mean, _, _ = model(diagnostic_X)
            deterministic_action_mean = float(diag_mean.mean().item())
            deterministic_action_abs_mean = float(diag_mean.abs().mean().item())
            deterministic_action_std = float(diag_mean.std().item())
            
            if prev_actor_mean is not None:
                delta = diag_mean - prev_actor_mean
                actor_mean_delta_from_prev_iter = float(delta.mean().item())
                actor_mean_abs_delta_from_prev_iter = float(delta.abs().mean().item())
            else:
                actor_mean_delta_from_prev_iter = 0.0
                actor_mean_abs_delta_from_prev_iter = 0.0
            prev_actor_mean = diag_mean.clone()

        improved = avg_reward > best_reward
        if improved:
            best_reward = avg_reward
            no_improve = 0
            save_checkpoint(
                best_path,
                model,
                asset,
                config,
                metadata,
                "best",
                {"best_reward": best_reward, "iteration": iteration + 1},
            )
            if not best_checkpoint:
                save_legacy_state_dict(MODELS_DIR / f"{asset}_model.pth", model)
        else:
            no_improve += 1

        early_stop_flag = False
        if no_improve > patience:
            early_stop_flag = True

        row = {
            "iteration": iteration + 1,
            "episode_reward": total_reward,
            "rolling_avg_reward": avg_reward,
            "mean_position": avg_position,
            "mean_abs_position": abs_position,
            "value_mean": value_mean,
            "value_std": value_std,
            "early_stop_flag": early_stop_flag,
            "deterministic_action_mean": deterministic_action_mean,
            "deterministic_action_abs_mean": deterministic_action_abs_mean,
            "deterministic_action_std": deterministic_action_std,
            "actor_mean_delta_from_prev_iter": actor_mean_delta_from_prev_iter,
            "actor_mean_abs_delta_from_prev_iter": actor_mean_abs_delta_from_prev_iter,
            "std_parameterization": config["ppo"].get("std_parameterization", "hard_clamp"),
        }
        row.update(update_metrics)
        training_rows.append(row)

        print(
            f"Iter {iteration + 1} | R: {total_reward:.2f} | AvgR: {avg_reward:.2f} | Pos: {avg_position:.2f} | Std: {std_mean:.3f}"
        )

        if early_stop_flag:
            print("⛔ Early stopping triggered")
            break

    save_checkpoint(
        final_path,
        model,
        asset,
        config,
        metadata,
        "final",
        {"best_reward": best_reward, "iterations_completed": len(training_rows)},
    )
    if not final_checkpoint:
        save_legacy_state_dict(MODELS_DIR / f"{asset}_model_final.pth", model)

    if training_rows:
        write_csv(
            run_dir / "training_trace.csv",
            training_rows,
            list(training_rows[0].keys()),
        )
        write_json(run_dir / "training_trace.json", {"trace": training_rows})
    write_json(
        run_dir / "metrics.json",
        {
            "asset": asset,
            "iterations_completed": len(training_rows),
            "best_reward": best_reward,
            "best_checkpoint": str(best_path),
            "final_checkpoint": str(final_path),
        },
    )

    eval_dir = run_dir / "evaluation"
    result, _, _ = evaluate_asset(
        asset=asset,
        config=config,
        checkpoint=str(best_path),
        output_dir=eval_dir,
        processed_dataset=processed_dataset,
    )
    write_json(run_dir / "evaluation_metrics.json", result)
    shutil.copyfile(eval_dir / "equity_curve.png", run_dir / "equity_curve.png")
    print(
        f"Evaluation | asset={asset} | final_equity={result['rl_policy']['final_equity']:.4f} | "
        f"sharpe={result['rl_policy']['sharpe']:.2f} | mdd={result['rl_policy']['max_drawdown']:.2%}"
    )
    return {
        "asset": asset,
        "best_checkpoint": str(best_path),
        "final_checkpoint": str(final_path),
        "run_dir": str(run_dir),
        "training_metrics": {
            "iterations_completed": len(training_rows),
            "best_reward": best_reward,
        },
        "evaluation": result["rl_policy"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default=os.environ.get("DATA_PREFIX", "btc_usdt"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--best-checkpoint", default=None)
    args = parser.parse_args()
    config = load_config(*([args.config] if args.config else []))
    train_asset(args.asset, config, best_checkpoint=args.best_checkpoint)


if __name__ == "__main__":
    main()
