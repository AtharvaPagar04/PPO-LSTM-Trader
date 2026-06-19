import numpy as np
import pandas as pd
import pytest

from src.evaluation.walk_forward import (
    WalkForwardFold,
    aggregate_baseline_comparisons,
    aggregate_walk_forward_metrics,
    compare_fold_strategies,
    create_walk_forward_folds,
    evaluate_baselines_on_fold,
)


def test_fold_generation_preserves_order_and_no_overlap():
    folds = create_walk_forward_folds(total_steps=20, folds=4)
    assert [(fold.start_index, fold.end_index) for fold in folds] == [
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
    ]


def test_fold_generation_handles_uneven_split():
    folds = create_walk_forward_folds(total_steps=23, folds=5)
    ranges = [(fold.start_index, fold.end_index) for fold in folds]
    assert ranges[0] == (0, 5)
    assert ranges[-1] == (19, 23)
    assert ranges == sorted(ranges)


def test_fold_generation_with_fold_size_covers_expected_range():
    folds = create_walk_forward_folds(total_steps=11, fold_size=4)
    assert [(fold.start_index, fold.end_index) for fold in folds] == [
        (0, 4),
        (4, 8),
        (8, 11),
    ]


def test_too_small_dataset_raises_clear_error():
    with pytest.raises(ValueError, match="too small|requires at least 2"):
        create_walk_forward_folds(total_steps=1, folds=5)


def test_invalid_fold_count_raises_clear_error():
    with pytest.raises(ValueError, match="positive integer"):
        create_walk_forward_folds(total_steps=10, folds=0)


def test_aggregate_metrics_compute_correct_counts_and_means():
    fold_rows = [
        {"final_equity": 1.1, "total_return": 0.1, "sharpe": 0.4, "max_drawdown": 0.02},
        {"final_equity": 0.9, "total_return": -0.1, "sharpe": -0.2, "max_drawdown": 0.05},
        {"final_equity": 1.05, "total_return": 0.05, "sharpe": 0.3, "max_drawdown": 0.03},
    ]
    aggregate = aggregate_walk_forward_metrics(fold_rows)
    assert aggregate["mean_total_return"] == pytest.approx(0.0166666667)
    assert aggregate["median_total_return"] == pytest.approx(0.05)
    assert aggregate["positive_fold_count"] == 2
    assert aggregate["negative_fold_count"] == 1
    assert aggregate["robustness_score"] == pytest.approx(2 / 3)
    assert aggregate["worst_max_drawdown"] == pytest.approx(0.05)


def test_baseline_comparison_returns_required_strategies_and_is_deterministic():
    price_windows = np.ones((6, 20, 5), dtype=np.float32)
    price_windows[:, :, 3] = np.array([100, 101, 102, 103, 104, 105], dtype=np.float32).reshape(-1, 1)
    traces_1, metrics_1 = evaluate_baselines_on_fold(
        price_windows,
        transaction_cost=0.001,
        seed=42,
        reference_actions=np.full(len(price_windows) - 1, -0.1, dtype=np.float32),
    )
    traces_2, metrics_2 = evaluate_baselines_on_fold(
        price_windows,
        transaction_cost=0.001,
        seed=42,
        reference_actions=np.full(len(price_windows) - 1, -0.1, dtype=np.float32),
    )
    assert set(metrics_1) >= {
        "always_long",
        "always_short",
        "always_flat",
        "random",
        "buy_and_hold",
        "constant_signed_mean_action",
        "constant_abs_mean_long",
        "constant_abs_mean_short",
    }
    assert np.allclose(traces_1["random"]["equity"], traces_2["random"]["equity"])
    assert metrics_1["always_flat"]["turnover"] == pytest.approx(0.0)
    assert metrics_1["always_long"]["number_of_steps"] == metrics_2["always_long"]["number_of_steps"]


def test_compare_fold_strategies_computes_ranks_and_beats():
    fold = WalkForwardFold(1, 0, 5)
    row = compare_fold_strategies(
        asset="btc_usdt",
        fold=fold,
        start_timestamp=pd.Timestamp("2026-01-01 00:00:00"),
        end_timestamp=pd.Timestamp("2026-01-01 04:00:00"),
        rl_metrics={
            "number_of_steps": 4,
            "final_equity": 1.05,
            "total_return": 0.05,
            "sharpe": 1.2,
            "max_drawdown": 0.02,
        },
        baseline_metrics={
            "always_long": {"final_equity": 1.01, "total_return": 0.01, "sharpe": 0.4, "max_drawdown": 0.03},
            "always_short": {"final_equity": 0.99, "total_return": -0.01, "sharpe": -0.2, "max_drawdown": 0.01},
            "always_flat": {"final_equity": 1.0, "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
            "random": {"final_equity": 1.02, "total_return": 0.02, "sharpe": 0.3, "max_drawdown": 0.04},
            "constant_signed_mean_action": {"final_equity": 1.01, "total_return": 0.01, "sharpe": 0.2, "max_drawdown": 0.03},
            "constant_abs_mean_long": {"final_equity": 1.01, "total_return": 0.01, "sharpe": 0.2, "max_drawdown": 0.03},
            "constant_abs_mean_short": {"final_equity": 0.99, "total_return": -0.01, "sharpe": -0.2, "max_drawdown": 0.02},
        },
    )
    assert row["best_strategy_by_return"] == "rl_policy"
    assert row["rl_rank_by_return"] == 1
    assert row["rl_beat_always_long"] is True
    assert row["start_index"] == 0
    assert row["end_index"] == 4


def test_compare_fold_ties_are_deterministic():
    fold = WalkForwardFold(1, 0, 5)
    row = compare_fold_strategies(
        asset="btc_usdt",
        fold=fold,
        start_timestamp=pd.Timestamp("2026-01-01 00:00:00"),
        end_timestamp=pd.Timestamp("2026-01-01 04:00:00"),
        rl_metrics={
            "number_of_steps": 4,
            "final_equity": 1.0,
            "total_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        },
        baseline_metrics={
            "always_long": {"final_equity": 1.0, "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
            "always_short": {"final_equity": 1.0, "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
            "always_flat": {"final_equity": 1.0, "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
            "random": {"final_equity": 1.0, "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
            "constant_signed_mean_action": {"final_equity": 1.0, "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
            "constant_abs_mean_long": {"final_equity": 1.0, "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
            "constant_abs_mean_short": {"final_equity": 1.0, "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0},
        },
    )
    assert row["best_strategy_by_return"] == "always_flat"
    assert row["rl_rank_by_return"] == 8


def test_aggregate_baseline_comparison_counts_and_ratios():
    rows = [
        {
            "asset": "btc_usdt",
            "rl_total_return": 0.10,
            "rl_sharpe": 1.0,
            "rl_max_drawdown": 0.02,
            "always_long_total_return": 0.05,
            "always_long_sharpe": 0.4,
            "always_short_total_return": -0.05,
            "always_short_sharpe": -0.4,
            "always_flat_total_return": 0.0,
            "always_flat_sharpe": 0.0,
            "random_total_return": 0.01,
            "random_sharpe": 0.1,
            "constant_signed_mean_action_total_return": 0.02,
            "constant_signed_mean_action_sharpe": 0.2,
            "constant_abs_mean_long_total_return": 0.03,
            "constant_abs_mean_long_sharpe": 0.3,
            "constant_abs_mean_short_total_return": -0.02,
            "constant_abs_mean_short_sharpe": -0.2,
            "best_strategy_by_return": "rl_policy",
            "best_strategy_by_sharpe": "rl_policy",
            "rl_beat_always_long": True,
            "rl_beat_always_short": True,
            "rl_beat_always_flat": True,
            "rl_beat_random": True,
            "rl_beat_constant_signed_mean_action": True,
            "rl_beat_constant_abs_mean_long": True,
            "rl_beat_constant_abs_mean_short": True,
        },
        {
            "asset": "btc_usdt",
            "rl_total_return": -0.02,
            "rl_sharpe": -0.5,
            "rl_max_drawdown": 0.05,
            "always_long_total_return": 0.01,
            "always_long_sharpe": 0.2,
            "always_short_total_return": -0.01,
            "always_short_sharpe": -0.1,
            "always_flat_total_return": 0.0,
            "always_flat_sharpe": 0.0,
            "random_total_return": -0.03,
            "random_sharpe": -0.4,
            "constant_signed_mean_action_total_return": -0.01,
            "constant_signed_mean_action_sharpe": -0.1,
            "constant_abs_mean_long_total_return": 0.01,
            "constant_abs_mean_long_sharpe": 0.2,
            "constant_abs_mean_short_total_return": -0.01,
            "constant_abs_mean_short_sharpe": -0.1,
            "best_strategy_by_return": "always_long",
            "best_strategy_by_sharpe": "always_long",
            "rl_beat_always_long": False,
            "rl_beat_always_short": False,
            "rl_beat_always_flat": False,
            "rl_beat_random": True,
            "rl_beat_constant_signed_mean_action": False,
            "rl_beat_constant_abs_mean_long": False,
            "rl_beat_constant_abs_mean_short": False,
        },
    ]
    aggregate = aggregate_baseline_comparisons(rows)
    assert aggregate["rl_best_return_fold_count"] == 1
    assert aggregate["rl_beat_always_long_count"] == 1
    assert aggregate["rl_beat_random_count"] == 2
    assert aggregate["rl_beat_constant_signed_mean_action_count"] == 1
    assert aggregate["rl_beat_random_ratio"] == pytest.approx(1.0)
    assert aggregate["best_overall_strategy_by_mean_return"] == "rl_policy"


def test_evaluate_walk_forward_asset_raises_value_error_on_mismatch():
    """Guard fires early (before model loading) when test_X length != timestamps length."""
    from src.evaluation.walk_forward import evaluate_walk_forward_asset
    from unittest.mock import patch

    # test_X has 10 rows but timestamps has only 5 → mismatch
    with patch("src.evaluation.walk_forward.load_processed_data", return_value=(np.zeros((10, 5, 2)), np.zeros((10, 5)))), \
         patch("src.evaluation.walk_forward.load_metadata", return_value={"window_size": 2, "split_ratio": 0.8}), \
         patch("src.evaluation.walk_forward._window_end_timestamps", return_value=[pd.Timestamp("2026-01-01")] * 5):

        with pytest.raises(ValueError, match="Walk-forward timestamp/index mismatch"):
            evaluate_walk_forward_asset(
                asset="btc_usdt",
                config={"evaluation": {"random_seed": 42}},
                checkpoint=None,
            )

