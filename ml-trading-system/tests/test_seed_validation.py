import json
from unittest.mock import patch

import pytest

from src.experiments.seed_validation import (
    DEFAULT_FEATURE_PRESETS,
    DEFAULT_SEEDS,
    aggregate_seed_validation_runs,
    compute_winner_counts,
    run_seed_validation_experiment,
)


def test_seed_validation_defaults_are_expected():
    assert DEFAULT_SEEDS == [42, 43, 44]
    assert DEFAULT_FEATURE_PRESETS == ["full_features", "price_action_minimal"]


def test_aggregate_seed_validation_runs_computes_mean_and_std():
    rows = [
        {
            "asset": "btc_usdt",
            "feature_preset": "full_features",
            "seed": 42,
            "feature_count": 10,
            "features": "a,b",
            "deterministic_return": 0.1,
            "walk_forward_mean_return": 0.05,
            "walk_forward_mean_sharpe": 0.2,
            "flat_ratio_001": 0.1,
            "flat_ratio_005": 0.2,
            "flat_ratio_010": 0.3,
            "flat_ratio_025": 1.0,
            "flat_ratio": 1.0,
            "average_abs_action": 0.02,
            "deterministic_max_drawdown": 0.03,
            "rl_best_return_fold_count": 0,
            "rl_beat_always_flat_count": 2,
            "score": 1.0,
        },
        {
            "asset": "btc_usdt",
            "feature_preset": "full_features",
            "seed": 43,
            "feature_count": 10,
            "features": "a,b",
            "deterministic_return": 0.3,
            "walk_forward_mean_return": 0.10,
            "walk_forward_mean_sharpe": 0.4,
            "flat_ratio_001": 0.1,
            "flat_ratio_005": 0.2,
            "flat_ratio_010": 0.3,
            "flat_ratio_025": 0.8,
            "flat_ratio": 0.8,
            "average_abs_action": 0.04,
            "deterministic_max_drawdown": 0.05,
            "rl_best_return_fold_count": 1,
            "rl_beat_always_flat_count": 3,
            "score": 2.0,
        },
        {
            "asset": "btc_usdt",
            "feature_preset": "price_action_minimal",
            "seed": 42,
            "feature_count": 4,
            "features": "a",
            "deterministic_return": 0.2,
            "walk_forward_mean_return": 0.07,
            "walk_forward_mean_sharpe": 0.5,
            "flat_ratio_001": 0.1,
            "flat_ratio_005": 0.2,
            "flat_ratio_010": 0.3,
            "flat_ratio_025": 0.7,
            "flat_ratio": 0.7,
            "average_abs_action": 0.10,
            "deterministic_max_drawdown": 0.04,
            "rl_best_return_fold_count": 2,
            "rl_beat_always_flat_count": 4,
            "score": 3.0,
        },
        {
            "asset": "btc_usdt",
            "feature_preset": "price_action_minimal",
            "seed": 43,
            "feature_count": 4,
            "features": "a",
            "deterministic_return": 0.0,
            "walk_forward_mean_return": 0.01,
            "walk_forward_mean_sharpe": 0.1,
            "flat_ratio_001": 0.1,
            "flat_ratio_005": 0.2,
            "flat_ratio_010": 0.3,
            "flat_ratio_025": 0.9,
            "flat_ratio": 0.9,
            "average_abs_action": 0.08,
            "deterministic_max_drawdown": 0.02,
            "rl_best_return_fold_count": 1,
            "rl_beat_always_flat_count": 1,
            "score": 0.5,
        },
    ]

    aggregate = aggregate_seed_validation_runs(rows)
    by_preset = {row["feature_preset"]: row for row in aggregate}

    full = by_preset["full_features"]
    assert full["num_seeds"] == 2
    assert full["mean_deterministic_return"] == pytest.approx(0.2)
    assert full["std_deterministic_return"] == pytest.approx(0.1)
    assert full["mean_walk_forward_mean_sharpe"] == pytest.approx(0.3)
    assert full["mean_flat_ratio_010"] == pytest.approx(0.3)
    assert full["mean_average_abs_action"] == pytest.approx(0.03)
    assert full["total_rl_beat_always_flat_count"] == 5


def test_compute_winner_counts_is_deterministic_for_ties():
    rows = [
        {
            "asset": "btc_usdt",
            "feature_preset": "full_features",
            "seed": 42,
            "deterministic_return": 0.1,
            "walk_forward_mean_sharpe": 0.2,
            "average_abs_action": 0.05,
            "deterministic_max_drawdown": 0.03,
            "flat_ratio_010": 0.7,
            "flat_ratio": 0.7,
            "score": 1.0,
        },
        {
            "asset": "btc_usdt",
            "feature_preset": "price_action_minimal",
            "seed": 42,
            "deterministic_return": 0.1,
            "walk_forward_mean_sharpe": 0.2,
            "average_abs_action": 0.05,
            "deterministic_max_drawdown": 0.03,
            "flat_ratio_010": 0.7,
            "flat_ratio": 0.7,
            "score": 1.0,
        },
    ]

    winners = compute_winner_counts(rows)
    assert winners["score"]["full_features"] == 1
    assert winners["walk_forward_mean_sharpe"]["full_features"] == 1
    assert winners["deterministic_return"]["full_features"] == 1


def test_seed_validation_manifest_shape(tmp_path):
    payload = {
        "feature_presets": ["full_features", "price_action_minimal"],
        "seeds": [42, 43, 44],
    }
    path = tmp_path / "audit_manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["feature_presets"] == ["full_features", "price_action_minimal"]
    assert loaded["seeds"] == [42, 43, 44]


def test_seed_validation_preflight_fails_before_training():
    fake_config = {
        "data": {"window_size": 20, "train_split": 0.8},
        "training": {"seed": 42},
    }
    with patch("src.experiments.seed_validation.train_asset") as mock_train:
        with pytest.raises(ValueError, match="Unknown feature ablation preset: missing_preset"):
            run_seed_validation_experiment(
                "btc_usdt",
                fake_config,
                feature_presets=["price_action_minimal", "missing_preset"],
                seeds=[42],
                quick=True,
            )
    mock_train.assert_not_called()
