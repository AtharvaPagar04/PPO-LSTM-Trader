import numpy as np

from src.features.pipeline import create_windows, split_windows


def test_create_windows_shape_is_correct():
    data = np.arange(60, dtype=np.float32).reshape(20, 3)
    windows = create_windows(data, window_size=5)
    assert windows.shape == (16, 5, 3)


def test_split_windows_preserves_alignment():
    features = np.arange(120, dtype=np.float32).reshape(10, 4, 3)
    prices = np.arange(200, dtype=np.float32).reshape(10, 4, 5)
    train_x, test_x, train_p, test_p = split_windows(features, prices, train_split=0.6)
    assert train_x.shape[0] == train_p.shape[0] == 6
    assert test_x.shape[0] == test_p.shape[0] == 4
    assert np.array_equal(train_x[-1], features[5])
    assert np.array_equal(test_x[0], features[6])
