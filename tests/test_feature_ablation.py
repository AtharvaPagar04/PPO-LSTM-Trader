import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.experiments.feature_ablation import run_feature_ablation_experiment


def test_feature_ablation_summary_includes_corrected_threshold_fields(tmp_path):
    import src.experiments.feature_ablation as exp

    original_exp_dir = exp.EXPERIMENTS_DIR
    exp.EXPERIMENTS_DIR = tmp_path

    fake_config = {
        "data": {"window_size": 20, "train_split": 0.8},
        "training": {"iterations": 2, "episode_length": 128, "rollout_steps": 128},
        "environment": {"transaction_cost": 0.0},
        "ppo": {},
        "features": {"selected": ["log_return", "trend"]},
    }
    fake_dataset = MagicMock()
    fake_dataset.metadata = {"window_size": 20, "train_rows": 10, "test_rows": 5}

    with patch("src.experiments.feature_ablation.build_processed_dataset", return_value=fake_dataset), \
         patch("src.experiments.feature_ablation.train_asset"), \
         patch("src.experiments.feature_ablation.evaluate_asset") as mock_eval, \
         patch("src.experiments.feature_ablation.collect_model_diagnostics") as mock_diag, \
         patch("src.experiments.feature_ablation.evaluate_walk_forward_asset") as mock_wf, \
         patch("src.experiments.feature_ablation.apply_feature_preset_to_config", return_value=fake_config), \
         patch("src.experiments.feature_ablation.validate_requested_feature_ablation_presets", return_value=[{"description": "test", "features": ["log_return", "trend"]}]):
        mock_eval.return_value = (
            {"rl_policy": {"total_return": 0.1, "sharpe": 0.5, "max_drawdown": 0.02, "final_equity": 1.1}},
            None,
            None,
        )
        mock_diag.return_value = {
            "summary": {
                "flat_ratio_001": 0.1,
                "flat_ratio_005": 0.2,
                "flat_ratio_010": 0.3,
                "flat_ratio_025": 0.4,
                "flat_ratio": 0.4,
                "long_ratio_001": 0.0,
                "short_ratio_001": 0.9,
                "long_ratio_005": 0.0,
                "short_ratio_005": 0.8,
                "long_ratio_010": 0.0,
                "short_ratio_010": 0.7,
                "long_ratio_025": 0.0,
                "short_ratio_025": 0.6,
                "long_ratio": 0.0,
                "short_ratio": 0.6,
                "dominant_action_side": "mostly_short",
                "action_mean": -0.1,
                "action_std": 0.01,
                "action_min": -0.2,
                "action_max": -0.05,
                "average_abs_action": 0.1,
                "action_abs_mean": 0.1,
                "action_abs_median": 0.1,
                "action_abs_p75": 0.11,
                "action_abs_p90": 0.12,
                "action_abs_p95": 0.13,
                "action_abs_p99": 0.14,
                "positive_action_ratio": 0.0,
                "negative_action_ratio": 1.0,
                "turnover": 1.0,
                "policy_std_mean": 0.5,
            }
        }

        def wf_side_effect(*args, **kwargs):
            if kwargs["include_baselines"]:
                return {
                    "baseline_aggregate": {
                        "rl_best_return_fold_count": 1,
                        "rl_beat_always_long_count": 1,
                        "rl_beat_always_short_count": 1,
                        "rl_beat_always_flat_count": 1,
                        "rl_beat_random_count": 1,
                        "rl_beat_constant_signed_mean_action_count": 1,
                        "rl_beat_constant_abs_mean_long_count": 1,
                        "rl_beat_constant_abs_mean_short_count": 1,
                        "constant_signed_mean_action_mean_return": 0.01,
                        "constant_abs_mean_long_mean_return": 0.02,
                        "constant_abs_mean_short_mean_return": -0.02,
                    }
                }
            return {
                "aggregate": {
                    "mean_total_return": 0.05,
                    "mean_sharpe": 0.4,
                    "worst_max_drawdown": 0.03,
                    "positive_fold_count": 3,
                    "robustness_score": 0.6,
                }
            }

        mock_wf.side_effect = wf_side_effect
        result = run_feature_ablation_experiment("btc_usdt", fake_config, ["full_features"])
        summary = json.loads((Path(result["experiment_dir"]) / "summary.json").read_text())
        assert "flat_ratio_001" in summary[0]
        assert "flat_ratio_010" in summary[0]
        assert "dominant_action_side" in summary[0]

    exp.EXPERIMENTS_DIR = original_exp_dir
