import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.evaluation.baselines import simulate_positions
from src.evaluation.metrics import compute_performance_metrics
from src.experiments.supervised_signal_strategy import (
    build_price_windows_from_frame,
    confidence_scaled_2x_action,
    confidence_scaled_action,
    evaluate_strategy_actions,
    hard_sign_action,
    regression_scaled_action,
    run_supervised_signal_strategy_experiment,
    target_direction_multiplier,
    thresholded_confidence_action,
    walk_forward_split,
)


def test_probability_action_mappings_are_correct():
    probabilities = np.array([0.40, 0.50, 0.60], dtype=np.float64)
    assert np.allclose(hard_sign_action(probabilities), [-1.0, -1.0, 1.0])
    assert np.allclose(confidence_scaled_action(probabilities), [-0.2, 0.0, 0.2])
    assert np.allclose(confidence_scaled_2x_action(probabilities), [-0.4, 0.0, 0.4])


def test_binary_down_threshold_maps_high_probability_to_short():
    probabilities = np.array([0.40, 0.50, 0.60], dtype=np.float64)
    direction_multiplier = target_direction_multiplier("binary_down_threshold")
    assert direction_multiplier == -1.0
    assert np.allclose(
        hard_sign_action(probabilities, direction_multiplier=direction_multiplier),
        [1.0, 1.0, -1.0],
    )
    assert np.allclose(
        confidence_scaled_action(probabilities, direction_multiplier=direction_multiplier),
        [0.2, -0.0, -0.2],
    )


def test_thresholded_confidence_maps_low_edge_to_zero():
    probabilities = np.array([0.49, 0.50, 0.51, 0.60], dtype=np.float64)
    actions = thresholded_confidence_action(probabilities, threshold=0.02)
    assert np.allclose(actions[:3], [0.0, 0.0, 0.0])
    assert actions[3] > 0.0


def test_regression_scaled_action_is_clipped():
    predictions = np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float64)
    actions = regression_scaled_action(predictions, train_return_std=0.5)
    assert np.all(actions <= 1.0)
    assert np.all(actions >= -1.0)
    assert actions[0] == pytest.approx(-1.0)
    assert actions[-1] == pytest.approx(1.0)


def test_unknown_target_label_fails_clearly():
    with pytest.raises(ValueError, match="Unsupported target label"):
        target_direction_multiplier("bad_label")


def test_chronological_split_does_not_shuffle():
    df = pd.DataFrame({"value": range(100)})
    splits = walk_forward_split(df, folds=3)
    train0, test0 = splits[0]
    train2, test2 = splits[2]
    assert train0.iloc[0]["value"] == 0
    assert train0.iloc[-1]["value"] == 24
    assert test0.iloc[0]["value"] == 25
    assert test0.iloc[-1]["value"] == 49
    assert train2.iloc[-1]["value"] == 74
    assert test2.iloc[0]["value"] == 75


def test_exposure_equivalent_baseline_uses_strategy_mean_action():
    prices = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100, 101, 102, 103, 104],
            "volume": [1, 1, 1, 1, 1],
        }
    )
    price_windows = build_price_windows_from_frame(prices)
    actions = np.array([-0.2, -0.1, 0.0, 0.1], dtype=np.float32)
    result = evaluate_strategy_actions(
        price_windows=price_windows,
        actions=actions,
        transaction_cost=0.001,
        seed=42,
    )
    direct_trace = simulate_positions(
        price_windows,
        np.full(len(actions), np.mean(actions), dtype=np.float32),
        0.001,
    )
    direct_metrics = compute_performance_metrics(direct_trace)
    assert (
        result["baselines"]["constant_signed_mean_action"]["final_equity"]
        == pytest.approx(direct_metrics["final_equity"])
    )


def test_unknown_feature_preset_fails_clearly_before_build():
    with patch("src.experiments.supervised_signal_strategy.build_labeled_dataset") as mock_build:
        with pytest.raises(ValueError, match="Unknown feature ablation preset"):
            run_supervised_signal_strategy_experiment(
                "btc_usdt",
                {
                    "data": {"train_split": 0.8},
                    "environment": {"transaction_cost": 0.001},
                    "evaluation": {"random_seed": 42},
                },
                feature_preset="missing_preset",
                horizon=1,
                quick=True,
            )
    mock_build.assert_not_called()


def test_summary_and_report_files_are_created(tmp_path):
    import src.experiments.supervised_signal_strategy as exp

    original_dir = exp.EXPERIMENTS_DIR
    exp.EXPERIMENTS_DIR = tmp_path
    features = [
        "log_return",
        "momentum_10",
        "trend",
        "rsi",
        "eth_return_24",
        "sol_return_24",
        "eth_return_72",
        "sol_return_72",
        "eth_btc_return_spread_24",
        "sol_btc_return_spread_24",
        "market_avg_return_24",
        "btc_relative_strength_24",
    ]
    rows = 240
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h")
    future_return = np.where(np.arange(rows) % 2 == 0, 0.01, -0.01)
    synthetic = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100 + np.arange(rows),
            "high": 101 + np.arange(rows),
            "low": 99 + np.arange(rows),
            "close": 100 + np.arange(rows),
            "volume": 1000 + np.arange(rows),
            "future_return_1": future_return,
            "next_up_1": future_return > 0,
        }
    )
    for feature in features:
        synthetic[feature] = future_return + np.random.default_rng(42).normal(0, 0.001, rows)

    config = {
        "data": {"train_split": 0.8},
        "environment": {"transaction_cost": 0.001},
        "evaluation": {"random_seed": 42},
    }
    try:
        with patch(
            "src.experiments.supervised_signal_strategy.build_labeled_dataset",
            return_value=synthetic,
        ):
            result = run_supervised_signal_strategy_experiment(
                "btc_usdt",
                config,
                feature_preset="cross_asset_context_v1",
                horizon=1,
                target_label="binary_down_threshold",
                threshold=0.001,
                quick=True,
            )
        exp_dir = Path(result["experiment_dir"])
        assert (exp_dir / "summary.csv").exists()
        assert (exp_dir / "summary.json").exists()
        assert (exp_dir / "report.md").exists()
        assert (exp_dir / "audit_manifest.json").exists()
        payload = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
        assert payload
        assert payload[0]["feature_preset"] == "cross_asset_context_v1"
        assert payload[0]["target_label"] == "binary_down_threshold"
        assert payload[0]["threshold"] == pytest.approx(0.001)
        manifest = json.loads((exp_dir / "audit_manifest.json").read_text(encoding="utf-8"))
        assert manifest["target_label"] == "binary_down_threshold"
        assert manifest["threshold"] == pytest.approx(0.001)
    finally:
        exp.EXPERIMENTS_DIR = original_dir
