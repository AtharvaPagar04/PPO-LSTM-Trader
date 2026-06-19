import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from src.experiments.signal_audit import (
    walk_forward_split,
    evaluate_horizon,
    run_signal_audit_experiment
)


def test_chronological_split_does_not_shuffle():
    df = pd.DataFrame({"val": range(100)})
    splits = walk_forward_split(df, folds=3)
    
    assert len(splits) == 3
    # fold_size = 100 // 4 = 25
    # fold 0: train=0..25, test=25..50
    # fold 1: train=0..50, test=50..75
    # fold 2: train=0..75, test=75..100
    
    train0, test0 = splits[0]
    assert train0.iloc[0]["val"] == 0
    assert train0.iloc[-1]["val"] == 24
    assert test0.iloc[0]["val"] == 25
    assert test0.iloc[-1]["val"] == 49
    
    train2, test2 = splits[2]
    assert train2.iloc[-1]["val"] == 74
    assert test2.iloc[0]["val"] == 75
    assert test2.iloc[-1]["val"] == 99


def test_baseline_metrics_compute_correctly():
    # Make a tiny predictable dataset with alternating labels
    labeled_df = pd.DataFrame({
        "feat1": np.tile([-10, 10], 50),
        "future_return_1": np.tile([-1, 1], 50),
        "next_up_1": np.tile([False, True], 50),
    })
    # feat1 perfectly correlates with future_return_1
    
    res = evaluate_horizon(labeled_df, ["feat1"], 1)
    
    assert res["label_horizon"] == 1
    assert "target_mean" in res
    assert "positive_label_ratio" in res
    assert res["feature_return_correlation_mean"] > 0.9  # Should be ~1.0
    assert res["information_coefficient_mean"] > 0.9     # Should be ~1.0
    assert res["classification_accuracy"] > 0.9          # Predictable
    assert "baseline_always_up" in res
    assert "baseline_always_down" in res
    assert "baseline_random" in res
    assert res["wf_positive_folds"] >= 0


@patch("src.experiments.signal_audit.build_labeled_dataset")
def test_unknown_feature_preset_fails_clearly(mock_build):
    with pytest.raises(ValueError, match="Unknown feature ablation preset"):
        run_signal_audit_experiment("btc_usdt", {}, ["missing_preset"], quick=True)
    mock_build.assert_not_called()


@patch("src.experiments.signal_audit.build_labeled_dataset")
def test_summary_report_files_are_created(mock_build):
    # Mock data
    dates = pd.date_range("2024-01-01", periods=100)
    df = pd.DataFrame({
        "timestamp": dates,
        "close": np.random.normal(100, 5, 100),
        "log_return": np.random.normal(0, 0.01, 100),
        "momentum_10": np.random.normal(0, 0.05, 100),
        "trend": np.random.normal(0, 0.01, 100),
        "rsi": np.random.uniform(20, 80, 100),
        "future_return_1": np.random.normal(0, 0.01, 100),
        "next_up_1": np.random.choice([True, False], 100),
        "future_return_3": np.random.normal(0, 0.01, 100),
        "next_up_3": np.random.choice([True, False], 100)
    })
    mock_build.return_value = df
    
    # Run experiment
    run_dir = run_signal_audit_experiment(
        "btc_usdt", {}, ["price_action_minimal"], quick=True
    )
    
    # Verify outputs
    assert run_dir.exists()
    assert (run_dir / "summary.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "audit_manifest.json").exists()
