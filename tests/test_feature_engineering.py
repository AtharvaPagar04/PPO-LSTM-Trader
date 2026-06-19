import numpy as np
import pandas as pd

from src.features.pipeline import FEATURE_COLUMNS, engineer_features


def make_raw_df(rows=80):
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="H")
    close = np.linspace(100.0, 140.0, rows)
    data = {
        "timestamp": timestamps,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.linspace(1000.0, 2000.0, rows),
    }
    return pd.DataFrame(data)


def test_engineer_features_produces_expected_columns():
    features = engineer_features(make_raw_df())
    assert all(column in features.columns for column in FEATURE_COLUMNS)
    assert np.isfinite(features[FEATURE_COLUMNS].to_numpy()).all()
    assert len(features) > 0


def test_engineer_features_output_shape_is_valid():
    features = engineer_features(make_raw_df())
    assert features.shape[1] >= len(FEATURE_COLUMNS)
