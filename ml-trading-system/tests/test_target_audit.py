import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.experiments.signal_audit import walk_forward_split
from src.experiments.target_audit import (
    build_threshold_labels,
    compute_label_balance,
    compute_target_relevance,
    estimated_round_trip_cost,
    run_target_audit_experiment,
)


def test_future_returns_shift_matches_expected_horizon():
    close = pd.Series([100.0, 102.0, 101.0, 104.0])
    future_return_1 = np.log(close.shift(-1) / close)
    assert future_return_1.iloc[0] == pytest.approx(np.log(102.0 / 100.0))
    assert future_return_1.iloc[1] == pytest.approx(np.log(101.0 / 102.0))


def test_threshold_labels_are_correct():
    returns = np.array([-0.002, -0.0002, 0.0001, 0.002])
    labels = build_threshold_labels(
        returns,
        threshold=0.001,
        label_type="binary_up_threshold",
    )
    assert labels.tolist() == [0, 0, 0, 1]


def test_ternary_labels_are_correct():
    returns = np.array([-0.002, -0.0002, 0.0001, 0.002])
    labels = build_threshold_labels(
        returns,
        threshold=0.001,
        label_type="ternary_direction",
    )
    assert labels.tolist() == [-1, 0, 0, 1]


def test_cost_aware_threshold_includes_round_trip_cost():
    returns = np.array([-0.002, -0.0012, 0.0012, 0.003])
    labels = build_threshold_labels(
        returns,
        threshold=0.001,
        label_type="cost_aware_direction",
        round_trip_cost=0.001,
    )
    assert labels.tolist() == [0, 0, 0, 1]
    assert estimated_round_trip_cost(0.0005) == pytest.approx(0.001)


def test_neutral_coverage_is_computed_correctly():
    labels = np.array([-1, 0, 0, 1, 1], dtype=np.int8)
    metrics = compute_label_balance(labels, label_type="ternary_direction")
    assert metrics["neutral_ratio"] == pytest.approx(0.4)
    assert metrics["signal_coverage"] == pytest.approx(0.6)


def test_target_relevance_uses_non_neutral_rows_only():
    returns = np.array([-0.003, 0.0, 0.002, 0.004], dtype=np.float64)
    labels = np.array([-1, 0, 1, 1], dtype=np.int8)
    metrics = compute_target_relevance(
        returns,
        labels,
        label_type="ternary_direction",
        threshold=0.001,
        round_trip_cost=0.0,
    )
    assert metrics["mean_future_return_when_long"] == pytest.approx(0.003)
    assert metrics["mean_future_return_when_short"] == pytest.approx(-0.003)
    assert metrics["mean_abs_future_return_for_non_neutral"] == pytest.approx(0.003)


def test_chronological_split_does_not_shuffle():
    df = pd.DataFrame({"value": range(60)})
    splits = walk_forward_split(df, folds=3)
    train0, test0 = splits[0]
    assert train0.iloc[0]["value"] == 0
    assert test0.iloc[0]["value"] == len(train0)
    assert test0["value"].is_monotonic_increasing


@patch("src.experiments.target_audit.build_labeled_dataset")
def test_unknown_feature_preset_fails_clearly(mock_build):
    with pytest.raises(ValueError, match="Unknown feature ablation preset"):
        run_target_audit_experiment(
            "btc_usdt",
            {
                "environment": {"transaction_cost": 0.0005},
                "data": {"train_split": 0.8},
            },
            feature_preset="missing_preset",
            quick=True,
        )
    mock_build.assert_not_called()


@patch("src.experiments.target_audit.build_labeled_dataset")
def test_summary_report_files_are_created(mock_build, tmp_path):
    import src.experiments.target_audit as exp

    original_dir = exp.EXPERIMENTS_DIR
    exp.EXPERIMENTS_DIR = tmp_path
    rows = 240
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h")
    rng = np.random.default_rng(42)
    future_1 = np.where(np.arange(rows) % 2 == 0, 0.002, -0.002)
    future_3 = np.where(np.arange(rows) % 3 == 0, 0.003, -0.001)
    future_6 = np.where(np.arange(rows) % 4 == 0, 0.004, -0.0015)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "future_return_1": future_1,
            "future_return_3": future_3,
            "future_return_6": future_6,
            "future_return_12": future_6,
            "future_return_24": future_6,
            "log_return": future_1 + rng.normal(0, 0.0001, rows),
            "momentum_10": future_1 + rng.normal(0, 0.0001, rows),
            "trend": future_1 + rng.normal(0, 0.0001, rows),
            "rsi": 50 + rng.normal(0, 1.0, rows),
            "eth_return_24": future_1 + rng.normal(0, 0.0001, rows),
            "sol_return_24": future_1 + rng.normal(0, 0.0001, rows),
            "eth_return_72": future_1 + rng.normal(0, 0.0001, rows),
            "sol_return_72": future_1 + rng.normal(0, 0.0001, rows),
            "eth_btc_return_spread_24": rng.normal(0, 0.0001, rows),
            "sol_btc_return_spread_24": rng.normal(0, 0.0001, rows),
            "market_avg_return_24": future_1 + rng.normal(0, 0.0001, rows),
            "btc_relative_strength_24": future_1 + rng.normal(0, 0.0001, rows),
        }
    )
    mock_build.return_value = df

    config = {
        "environment": {"transaction_cost": 0.0005},
        "data": {"train_split": 0.8},
    }
    try:
        result = run_target_audit_experiment(
            "btc_usdt",
            config,
            feature_preset="cross_asset_context_v1",
            horizons=[1, 3, 6],
            thresholds=[0.0, 0.0005],
            quick=True,
        )
        run_dir = Path(result["experiment_dir"])
        assert (run_dir / "summary.csv").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "report.md").exists()
        assert (run_dir / "audit_manifest.json").exists()
        payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert payload
        assert payload[0]["feature_preset"] == "cross_asset_context_v1"
    finally:
        exp.EXPERIMENTS_DIR = original_dir
