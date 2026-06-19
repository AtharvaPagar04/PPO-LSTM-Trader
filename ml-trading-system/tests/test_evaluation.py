from pathlib import Path
import json

import numpy as np
import torch

from src.config import paths as config_paths
from src.config.settings import load_config
from src.evaluation.backtest import run_policy_backtest
from src.evaluation.baselines import run_baselines
from src.evaluation.benchmark import evaluate_asset
from src.evaluation.metrics import compute_performance_metrics
from src.models.policy import LSTMPolicy
from src.utils.seed import set_global_seed


def make_eval_data(num_windows=12):
    x = np.random.randn(num_windows, 20, 10).astype(np.float32)
    prices = np.ones((num_windows, 20, 5), dtype=np.float32)
    prices[:, :, 3] = np.linspace(100.0, 111.0, num_windows).reshape(-1, 1)
    return x, prices


def test_full_period_backtest_is_deterministic():
    set_global_seed(42)
    x, prices = make_eval_data()
    from src.env.trading_env import TradingEnv

    env = TradingEnv(x, prices, max_steps=3)
    model = LSTMPolicy(input_dim=10)

    trace1 = run_policy_backtest(env, model, deterministic_policy=True)
    trace2 = run_policy_backtest(env, model, deterministic_policy=True)

    assert trace1["equity"].shape == trace2["equity"].shape
    assert len(trace1["action"]) == len(prices) - 1
    assert np.allclose(trace1["equity"], trace2["equity"])


def test_metrics_include_required_fields():
    x, prices = make_eval_data()
    from src.env.trading_env import TradingEnv

    env = TradingEnv(x, prices)
    model = LSTMPolicy(input_dim=10)
    trace = run_policy_backtest(env, model, deterministic_policy=True)
    metrics = compute_performance_metrics(trace)

    assert {"final_equity", "sharpe", "max_drawdown", "number_of_steps"} <= set(metrics)


def test_evaluate_asset_writes_outputs_with_synthetic_artifacts(tmp_path, monkeypatch):
    asset = "btc_usdt"
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    eval_dir = tmp_path / "evaluation"
    processed_dir.mkdir()
    models_dir.mkdir()

    x, prices = make_eval_data()
    np.save(processed_dir / f"{asset}_test_windows.npy", x)
    np.save(processed_dir / f"{asset}_test_price_windows.npy", prices)
    np.save(processed_dir / f"{asset}_train_windows.npy", x)
    np.save(processed_dir / f"{asset}_train_price_windows.npy", prices)
    with (processed_dir / f"{asset}_meta.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "asset": asset,
                "window_size": 20,
                "features": [f"feature_{idx}" for idx in range(10)],
                "split_ratio": 0.8,
            },
            handle,
        )
    torch.save(LSTMPolicy(input_dim=10).state_dict(), models_dir / f"{asset}_best.pt")

    monkeypatch.setattr(config_paths, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(config_paths, "MODELS_DIR", models_dir)
    monkeypatch.setattr(config_paths, "EVALUATION_DIR", eval_dir)

    config = load_config()
    result, _, _ = evaluate_asset(asset=asset, config=config, output_dir=eval_dir / asset)

    assert result["asset"] == asset
    assert (eval_dir / asset / "metrics.json").exists()
    assert (eval_dir / asset / "equity_curve.png").exists()
    assert (eval_dir / asset / "actions.csv").exists()


def test_random_baseline_is_deterministic_and_flat_has_zero_turnover():
    _, prices = make_eval_data()
    ref_actions = np.full(len(prices) - 1, -0.1, dtype=np.float32)
    traces_1 = run_baselines(prices, transaction_cost=0.001, seed=42, reference_actions=ref_actions)
    traces_2 = run_baselines(prices, transaction_cost=0.001, seed=42, reference_actions=ref_actions)

    assert np.allclose(traces_1["random"]["equity"], traces_2["random"]["equity"])
    flat_metrics = compute_performance_metrics(traces_1["always_flat"])
    assert flat_metrics["turnover"] == 0.0
    assert "constant_signed_mean_action" in traces_1
