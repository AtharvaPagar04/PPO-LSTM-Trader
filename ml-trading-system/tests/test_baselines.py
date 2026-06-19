import numpy as np
import pytest

from src.evaluation.baselines import (
    build_exposure_equivalent_positions,
    run_baselines,
)


def make_price_windows():
    prices = np.ones((6, 20, 5), dtype=np.float32)
    prices[:, :, 3] = np.array([100, 101, 102, 103, 104, 105], dtype=np.float32).reshape(-1, 1)
    return prices


def test_exposure_equivalent_positions_use_mean_and_abs_mean():
    actions = np.array([-0.2, -0.1, 0.1, 0.0], dtype=np.float32)
    positions = build_exposure_equivalent_positions(actions)
    assert positions["constant_signed_mean_action"][0] == pytest.approx(np.mean(actions))
    assert positions["constant_abs_mean_long"][0] == pytest.approx(np.mean(np.abs(actions)))
    assert positions["constant_abs_mean_short"][0] == pytest.approx(-np.mean(np.abs(actions)))


def test_constant_baseline_applies_initial_transaction_cost_once():
    prices = make_price_windows()
    traces = run_baselines(
        prices,
        transaction_cost=0.001,
        seed=42,
        reference_actions=np.full(len(prices) - 1, -0.1, dtype=np.float32),
    )
    costs = traces["constant_signed_mean_action"]["transaction_cost"]
    assert costs[0] == pytest.approx(0.0001)
    assert np.allclose(costs[1:], 0.0)


def test_run_baselines_returns_exposure_equivalent_traces():
    prices = make_price_windows()
    traces = run_baselines(
        prices,
        transaction_cost=0.001,
        seed=42,
        reference_actions=np.full(len(prices) - 1, -0.1, dtype=np.float32),
    )
    assert "constant_signed_mean_action" in traces
    assert "constant_abs_mean_long" in traces
    assert "constant_abs_mean_short" in traces
