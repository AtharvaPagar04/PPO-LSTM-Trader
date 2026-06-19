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
    assert dataset.metadata["features"] == selected
    assert dataset.train_windows.shape[-1] == len(selected)


def test_engineer_labels_shifts_correctly():
    df = make_raw_df(rows=50)
    labels = pipeline.engineer_labels(df)
    
    # original length was 50, max horizon is 24, so length should be 50 - 24 = 26
    assert len(labels) == 26
    
    # check that future_return_1 is correct
    # log(close[t+1] / close[t])
    for i in range(len(labels)):
        expected_ret1 = np.log(df.iloc[i + 1]["close"] / df.iloc[i]["close"])
        assert np.isclose(labels.iloc[i]["future_return_1"], expected_ret1)
        
        expected_ret24 = np.log(df.iloc[i + 24]["close"] / df.iloc[i]["close"])
        assert np.isclose(labels.iloc[i]["future_return_24"], expected_ret24)
        
        assert labels.iloc[i]["next_up_1"] == (expected_ret1 > 0)


def test_no_future_labels_leak_into_features(monkeypatch):
    monkeypatch.setattr(pipeline, "load_raw_dataframe", lambda asset: make_raw_df(rows=100))
    dataset = pipeline.build_processed_dataset(
        asset="btc_usdt",
        window_size=20,
        train_split=0.8,
    )
    # Check that labels are not in features metadata
    assert not any(f.startswith("future_return") or f.startswith("next_up") for f in dataset.metadata["features"])
    # Check that ALL_FEATURE_COLUMNS does not contain labels
    assert not any(f.startswith("future_return") or f.startswith("next_up") for f in pipeline.ALL_FEATURE_COLUMNS)
