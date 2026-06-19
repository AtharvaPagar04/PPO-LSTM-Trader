import json
import pytest
from pathlib import Path
from src.experiments.training_signal import run_training_signal_experiment
import numpy as np
import torch

from src.env.trading_env import TradingEnv
from src.models.policy import LSTMPolicy
from src.ppo.ppo_trainer import PPOTrainer

def test_training_signal_experiment_generates_artifacts(tmp_path):
    import src.experiments.training_signal as ts
    original_exp_dir = ts.EXPERIMENTS_DIR
    ts.EXPERIMENTS_DIR = tmp_path
    
    from unittest.mock import patch
    
    fake_config = {
        "training": {"iterations": 2, "episode_length": 128, "rollout_steps": 128},
        "environment": {"transaction_cost": 0.0},
        "ppo": {"log_std_max": -0.2, "entropy_coef": 0.01}
    }
    
    # Mock train_asset to pretend it wrote a training_trace.json
    def fake_train_asset(asset, config, best_checkpoint, final_checkpoint, run_dir):
        from src.config.paths import ensure_dir
        run_dir = Path(run_dir)
        ensure_dir(run_dir)
        fake_trace = {
            "trace": [
                {
                    "deterministic_action_abs_mean": 0.01,
                    "policy_std_mean": 0.8,
                    "raw_log_std_mean": 1.2,
                    "std_high_saturation_ratio": 1.0,
                    "std_parameterization": "hard_clamp",
                    "raw_advantage_std": 0.1,
                    "normalized_advantage_std": 1.0,
                    "returns_std": 0.5,
                    "td_delta_std": 0.4,
                    "actor_grad_norm": 0.001,
                    "critic_grad_norm": 0.01,
                    "total_grad_norm": 0.01,
                    "approx_kl": 0.0001,
                    "clip_fraction": 0.0,
                    "ratio_std": 0.05,
                    "value_loss": 0.1,
                    "policy_loss": 0.0,
                    "entropy": 1.0,
                    "value_error_abs_mean": 0.1,
                    "explained_variance": 0.9
                },
                {
                    "deterministic_action_abs_mean": 0.015,
                    "policy_std_mean": 0.79,
                    "raw_log_std_mean": 1.1,
                    "std_high_saturation_ratio": 0.9,
                    "std_parameterization": "hard_clamp",
                    "raw_advantage_std": 0.11,
                    "normalized_advantage_std": 1.0,
                    "returns_std": 0.51,
                    "td_delta_std": 0.41,
                    "actor_grad_norm": 0.0011,
                    "critic_grad_norm": 0.011,
                    "total_grad_norm": 0.011,
                    "approx_kl": 0.0001,
                    "clip_fraction": 0.0,
                    "ratio_std": 0.05,
                    "value_loss": 0.11,
                    "policy_loss": 0.0,
                    "entropy": 1.0,
                    "value_error_abs_mean": 0.11,
                    "explained_variance": 0.9
                }
            ]
        }
        with open(run_dir / "training_trace.json", "w") as f:
            json.dump(fake_trace, f)
            
        with open(run_dir / "training_trace.csv", "w") as f:
            f.write("mock,csv\n")
            
    with patch("src.experiments.training_signal.train_asset", side_effect=fake_train_asset):
        res = run_training_signal_experiment("btc_usdt", fake_config, quick=True)
        
    exp_dir = Path(res["experiment_dir"])
    assert (exp_dir / "signal_summary.json").exists()
    assert (exp_dir / "report.md").exists()
    assert (exp_dir / "training_trace.csv").exists()
    assert (exp_dir / "training_trace.json").exists()
    
    with open(exp_dir / "signal_summary.json") as f:
        summary = json.load(f)
        assert summary["actor_mean_abs_start"] == 0.01
        assert summary["actor_mean_abs_end"] == 0.015
        assert summary["actor_mean_stagnant"] is True  # change is 0.005 < 0.01
        assert summary["raw_log_std_mean_start"] == 1.2
        assert summary["std_high_saturation_ratio_end"] == 0.9
        assert summary["std_parameterization"] == "hard_clamp"
        
    ts.EXPERIMENTS_DIR = original_exp_dir


def test_ppo_update_emits_std_audit_metrics():
    x = np.random.randn(20, 20, 10).astype(np.float32)
    prices = np.ones((20, 20, 5), dtype=np.float32)
    prices[:, :, 3] = np.linspace(100.0, 120.0, 20).reshape(-1, 1)
    env = TradingEnv(x, prices, max_steps=8)
    model = LSTMPolicy(
        input_dim=10,
        hidden_dim=32,
        lstm_layers=1,
        dropout=0.0,
        std_parameterization="hard_clamp",
    )
    trainer = PPOTrainer(env, model, lr=1e-3)
    rollout = trainer.collect_rollout(steps=8)
    metrics = trainer.update(rollout, epochs=1, batch_size=4)
    for key in [
        "raw_log_std_mean",
        "raw_log_std_min",
        "raw_log_std_max",
        "log_std_mean",
        "log_std_min",
        "log_std_max",
        "std_high_saturation_ratio",
        "std_low_saturation_ratio",
    ]:
        assert key in metrics
