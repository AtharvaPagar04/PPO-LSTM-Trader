import numpy as np
import pandas as pd

import src.features.pipeline as pipeline


def make_raw_df(rows=180):
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="H")
    close = np.linspace(100.0, 140.0, rows)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(1000.0, 2000.0, rows),
        }
    )


def test_engineer_features_produces_new_regime_columns_without_inf():
    features = pipeline.engineer_features(make_raw_df(rows=200))
    expected = [
        "return_24",
        "return_72",
        "volatility_24",
        "volatility_72",
        "ma_ratio_24_72",
        "ma_slope_24",
        "ma_slope_72",
        "trend_strength_24",
        "volatility_regime",
        "rsi_24",
        "drawdown_from_rolling_high_72",
    ]
    assert all(column in features.columns for column in expected)
    assert np.isfinite(features[expected].to_numpy()).all()


def test_selected_feature_list_changes_window_dimension(monkeypatch):
    monkeypatch.setattr(pipeline, "load_raw_dataframe", lambda asset: make_raw_df())
    dataset = pipeline.build_processed_dataset(
        asset="btc_usdt",
        window_size=20,
        train_split=0.8,
        selected_features=["log_return", "trend", "rsi"],
    )
    assert dataset.train_windows.shape[-1] == 3
    assert dataset.metadata["features"] == ["log_return", "trend", "rsi"]


def test_default_behavior_matches_full_feature_count(monkeypatch):
    monkeypatch.setattr(pipeline, "load_raw_dataframe", lambda asset: make_raw_df())
    dataset = pipeline.build_processed_dataset(
        asset="btc_usdt",
        window_size=20,
        train_split=0.8,
    )
    assert dataset.train_windows.shape[-1] == len(pipeline.FEATURE_COLUMNS)
    assert dataset.metadata["features"] == pipeline.FEATURE_COLUMNS


def test_scaler_is_fit_on_train_and_test_uses_selected_feature_count(monkeypatch):
    monkeypatch.setattr(pipeline, "load_raw_dataframe", lambda asset: make_raw_df())
    dataset = pipeline.build_processed_dataset(
        asset="btc_usdt",
        window_size=20,
        train_split=0.8,
        selected_features=["log_return", "momentum_10", "trend", "rsi"],
    )
    assert dataset.train_windows.shape[-1] == 4
    assert dataset.test_windows.shape[-1] == 4
    assert dataset.metadata["train_rows"] == dataset.train_windows.shape[0]
    assert dataset.metadata["test_rows"] == dataset.test_windows.shape[0]


def test_selected_regime_feature_order_is_stable(monkeypatch):
    monkeypatch.setattr(pipeline, "load_raw_dataframe", lambda asset: make_raw_df(rows=200))
    selected = [
        "log_return",
        "momentum_10",
        "return_24",
        "volatility_72",
        "drawdown_from_rolling_high_72",
    ]
    dataset = pipeline.build_processed_dataset(
        asset="btc_usdt",
        window_size=20,
        train_split=0.8,
        selected_features=selected,
    )
    assert dataset.metadata["features"] == selected
    assert dataset.train_windows.shape[-1] == len(selected)
