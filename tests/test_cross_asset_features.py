import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from src.features.pipeline import add_cross_asset_features, CROSS_ASSET_FEATURE_COLUMNS


@pytest.fixture
def synthetic_dfs():
    dates = pd.date_range("2020-01-01", periods=100, freq="H")
    
    # Target BTC df
    btc_df = pd.DataFrame({
        "timestamp": dates,
        "close": np.linspace(10000, 20000, 100),
        "log_return": np.random.normal(0, 0.01, 100),
        "return_24": np.random.normal(0, 0.05, 100),
        "return_72": np.random.normal(0, 0.1, 100),
        "volatility_24": np.random.normal(0.02, 0.005, 100)
    })
    
    # Context ETH df
    eth_df = pd.DataFrame({
        "timestamp": dates,
        "close": np.linspace(100, 500, 100)
    })
    
    # Context SOL df
    sol_df = pd.DataFrame({
        "timestamp": dates,
        "close": np.linspace(10, 50, 100)
    })
    
    return btc_df, eth_df, sol_df


@patch("src.features.pipeline.load_raw_dataframe")
@patch("src.features.pipeline.resolve_existing_raw_data_path")
def test_cross_asset_features_generation(mock_resolve, mock_load, synthetic_dfs):
    btc_df, eth_df, sol_df = synthetic_dfs
    
    def mock_load_raw(asset):
        if asset == "eth_usdt": return eth_df
        if asset == "sol_usdt": return sol_df
        raise FileNotFoundError
        
    mock_load.side_effect = mock_load_raw
    mock_resolve.return_value = "fake/path.csv"
    
    result_df = add_cross_asset_features(btc_df, "btc_usdt", ("eth_usdt", "sol_usdt"))
    
    # Should have all CROSS_ASSET_FEATURE_COLUMNS
    for col in CROSS_ASSET_FEATURE_COLUMNS:
        assert col in result_df.columns, f"Missing feature: {col}"
        
    # Should not have infinite values
    assert not np.isinf(result_df[CROSS_ASSET_FEATURE_COLUMNS].values).any()
    
    # Should drop NAs created by shift(72)
    # 72 rows dropped, out of 100 -> length 28
    assert len(result_df) == 100 - 72


@patch("src.features.pipeline.resolve_existing_raw_data_path")
def test_missing_eth_sol_data_raises_clear_error(mock_resolve, synthetic_dfs):
    btc_df, _, _ = synthetic_dfs
    mock_resolve.side_effect = FileNotFoundError
    
    with pytest.raises(FileNotFoundError, match="Cross-asset features require raw data for eth_usdt and sol_usdt"):
        add_cross_asset_features(btc_df, "btc_usdt", ("eth_usdt", "sol_usdt"))


@patch("src.features.pipeline.load_raw_dataframe")
@patch("src.features.pipeline.resolve_existing_raw_data_path")
def test_timestamp_alignment_is_stable(mock_resolve, mock_load, synthetic_dfs):
    btc_df, eth_df, sol_df = synthetic_dfs
    
    # Misalign ETH df by removing first 5 rows and adding 5 new rows at the end
    dates = pd.date_range("2020-01-01", periods=105, freq="H")
    eth_df_shifted = pd.DataFrame({
        "timestamp": dates[5:],
        "close": np.linspace(100, 500, 100)
    })
    
    def mock_load_raw(asset):
        if asset == "eth_usdt": return eth_df_shifted
        if asset == "sol_usdt": return sol_df
        raise FileNotFoundError
        
    mock_load.side_effect = mock_load_raw
    mock_resolve.return_value = "fake/path.csv"
    
    result_df = add_cross_asset_features(btc_df, "btc_usdt", ("eth_usdt", "sol_usdt"))
    
    # btc_df has dates from index 0 to 99. eth_df has dates from 5 to 104.
    # We join on timestamp. btc_df has target timestamps.
    # Shift features require 72 rows. So valid data starts from index 5 + 72 = 77.
    # Total valid rows = 100 - 77 = 23 rows.
    assert len(result_df) > 0
    assert result_df["timestamp"].is_monotonic_increasing
    
    # Check that alignment worked (no NaNs left)
    assert not result_df.isna().any().any()
