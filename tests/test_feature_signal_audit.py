import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.experiments.feature_signal_audit import (
    compute_feature_fold_metrics,
    compute_signal_strength_bucket,
    prepare_feature_signal_dataset,
    run_feature_signal_audit_experiment,
    safe_corr,
    safe_mutual_information,
    safe_single_feature_auc,
    shuffle_train_feature,
)


def make_signal_df(rows=180):
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h")
    base = np.sin(np.linspace(0, 12, rows))
    future_1 = np.roll(base, -1)
    future_6 = np.roll(base, -6)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": 100 + np.arange(rows),
            "high": 101 + np.arange(rows),
            "low": 99 + np.arange(rows),
            "close": 100 + np.arange(rows),
            "volume": 1000 + np.arange(rows),
            "log_return": base,
            "momentum_10": base * 0.8,
            "trend": base * 0.6,
            "rsi": 50 + base * 10,
            "eth_return_24": base * 0.7,
            "sol_return_24": base * 0.65,
            "eth_return_72": base * 0.55,
            "sol_return_72": base * 0.5,
            "eth_btc_return_spread_24": base * 0.45,
            "sol_btc_return_spread_24": base * 0.4,
            "market_avg_return_24": base * 0.35,
            "btc_relative_strength_24": base * 0.3,
            "future_return_1": future_1,
            "future_return_3": np.roll(base, -3),
            "future_return_6": future_6,
            "future_return_12": np.roll(base, -12),
            "future_return_24": np.roll(base, -24),
            "next_up_1": future_1 > 0,
            "next_up_3": np.roll(base, -3) > 0,
            "next_up_6": future_6 > 0,
            "next_up_12": np.roll(base, -12) > 0,
            "next_up_24": np.roll(base, -24) > 0,
        }
    )
    return df.iloc[:-24].reset_index(drop=True)


def test_metrics_compute_on_synthetic_signal():
    x = np.array([-2.0, -1.0, 1.0, 2.0, 3.0])
    y = np.array([-2.0, -1.0, 1.0, 2.0, 3.0])
    labels = np.array([0, 0, 1, 1, 1])
    assert safe_corr("spearman", x, y) > 0.9
    assert safe_mutual_information(x, labels) >= 0.0
    train_auc, test_auc = safe_single_feature_auc(x[:4], labels[:4], x[1:], labels[1:])
    assert train_auc >= 0.5
    assert test_auc >= 0.5


def test_walk_forward_split_is_chronological():
    df = pd.DataFrame({"value": range(100)})
    from src.experiments.signal_audit import walk_forward_split

    splits = walk_forward_split(df, folds=3)
    train0, test0 = splits[0]
    assert train0["value"].is_monotonic_increasing
    assert test0["value"].is_monotonic_increasing
    assert train0.iloc[-1]["value"] < test0.iloc[0]["value"]


def test_permutation_check_shuffles_only_train_fold():
    train_feature = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    shuffled = shuffle_train_feature(train_feature, seed=42)
    assert sorted(shuffled.tolist()) == sorted(train_feature.tolist())
    assert not np.array_equal(shuffled, train_feature)


def test_signal_strength_bucket_thresholds():
    assert compute_signal_strength_bucket(
        mean_abs_spearman_test=0.04, mean_mi_test=0.0, mean_auc_test=0.50
    ) == "strong"
    assert compute_signal_strength_bucket(
        mean_abs_spearman_test=0.0, mean_mi_test=0.0015, mean_auc_test=0.50
    ) == "weak"
    assert compute_signal_strength_bucket(
        mean_abs_spearman_test=0.0, mean_mi_test=0.0, mean_auc_test=0.50
    ) == "no_signal"


@patch("src.experiments.feature_signal_audit.load_raw_dataframe")
@patch("src.experiments.feature_signal_audit.engineer_features")
@patch("src.experiments.feature_signal_audit.add_cross_asset_features")
@patch("src.experiments.feature_signal_audit.engineer_labels")
def test_configured_preset_features_are_loaded_correctly(
    mock_labels, mock_cross, mock_engineer, mock_raw
):
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=200, freq="h"),
            "open": np.arange(200),
            "high": np.arange(200),
            "low": np.arange(200),
            "close": np.arange(200),
            "volume": np.arange(200),
        }
    )
    signal_df = make_signal_df()
    mock_raw.return_value = raw
    mock_engineer.return_value = signal_df.copy()
    mock_cross.return_value = signal_df.copy()
    mock_labels.return_value = signal_df.copy()
    dataset = prepare_feature_signal_dataset("btc_usdt", "cross_asset_context_v1")
    assert dataset["configured_features"] == [
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
    assert dataset["missing_configured_features"] == []


@patch("src.experiments.feature_signal_audit.load_raw_dataframe")
@patch("src.experiments.feature_signal_audit.engineer_features")
@patch("src.experiments.feature_signal_audit.add_cross_asset_features")
def test_missing_preset_feature_fails_clearly(mock_cross, mock_engineer, mock_raw):
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
            "open": np.arange(50),
            "high": np.arange(50),
            "low": np.arange(50),
            "close": np.arange(50),
            "volume": np.arange(50),
        }
    )
    incomplete = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
            "log_return": np.arange(50),
        }
    )
    mock_raw.return_value = raw
    mock_engineer.return_value = incomplete
    mock_cross.return_value = incomplete
    with pytest.raises(ValueError, match="Configured preset features are missing"):
        prepare_feature_signal_dataset("btc_usdt", "cross_asset_context_v1")


def test_fold_metrics_compute_for_synthetic_data():
    df = make_signal_df()
    rows = compute_feature_fold_metrics(df, feature_name="log_return", horizon=1, folds=3)
    assert rows
    assert "spearman_test" in rows[0]
    assert "auc_test" in rows[0]
    assert "shuffled_auc_test" in rows[0]


@patch("src.experiments.feature_signal_audit.load_raw_dataframe")
@patch("src.experiments.feature_signal_audit.engineer_features")
@patch("src.experiments.feature_signal_audit.add_cross_asset_features")
@patch("src.experiments.feature_signal_audit.engineer_labels")
def test_summary_report_files_are_created(
    mock_labels, mock_cross, mock_engineer, mock_raw, tmp_path
):
    import src.experiments.feature_signal_audit as exp

    original_dir = exp.EXPERIMENTS_DIR
    exp.EXPERIMENTS_DIR = tmp_path
    raw = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=300, freq="h"),
            "open": np.arange(300),
            "high": np.arange(300),
            "low": np.arange(300),
            "close": np.arange(300),
            "volume": np.arange(300),
        }
    )
    signal_df = make_signal_df(rows=300)
    mock_raw.return_value = raw
    mock_engineer.return_value = signal_df.copy()
    mock_cross.return_value = signal_df.copy()
    mock_labels.return_value = signal_df.copy()
    try:
        result = run_feature_signal_audit_experiment(
            "btc_usdt",
            {},
            feature_preset="cross_asset_context_v1",
            horizons=[1, 6],
            quick=True,
        )
        run_dir = Path(result["experiment_dir"])
        assert (run_dir / "summary.csv").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "feature_rankings.csv").exists()
        assert (run_dir / "fold_level_metrics.csv").exists()
        assert (run_dir / "report.md").exists()
        assert (run_dir / "audit_manifest.json").exists()
        payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        assert payload
    finally:
        exp.EXPERIMENTS_DIR = original_dir
