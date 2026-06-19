import json
import pytest
from pathlib import Path
from src.experiments.ppo_std_tuning import run_ppo_std_experiment
from src.experiments.feature_ablation import run_feature_ablation_experiment
from src.experiments.seed_validation import run_seed_validation_experiment

def test_experiment_reporting_manifest_and_summary(tmp_path):
    import src.experiments.ppo_std_tuning as tun
    original_exp_dir = tun.EXPERIMENTS_DIR
    tun.EXPERIMENTS_DIR = tmp_path
    
    # Mock train_asset, collect_model_diagnostics, evaluate_walk_forward_asset
    from unittest.mock import patch, MagicMock
    
    fake_config = {
        "training": {"iterations": 2, "episode_length": 128, "rollout_steps": 128},
        "environment": {"transaction_cost": 0.0},
        "ppo": {"log_std_max": -0.2, "entropy_coef": 0.01}
    }
    
    with patch("src.experiments.ppo_std_tuning.train_asset") as mock_train, \
         patch("src.experiments.ppo_std_tuning.collect_model_diagnostics") as mock_diag, \
         patch("src.experiments.ppo_std_tuning.evaluate_walk_forward_asset") as mock_wf, \
         patch("src.experiments.ppo_std_tuning.apply_ppo_std_preset_to_config", return_value=fake_config):
         
        mock_diag.return_value = {
            "summary": {
                "final_equity": 1.0, "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                "action_mean": 0.0, "average_abs_action": 0.0, "flat_ratio": 0.0,
                "long_ratio": 0.0, "short_ratio": 0.0, "policy_std_mean": 0.0,
                "policy_std_min": 0.0, "policy_std_max": 0.0, "turnover": 0.0, "reward_clip_ratio": 0.0
            }
        }
        
        # mock_wf returns two formats depending on include_baselines
        def wf_side_effect(asset, config, checkpoint, folds, output_dir, include_baselines):
            if include_baselines:
                return {"baseline_aggregate": {"rl_best_return_fold_count": 0, "rl_beat_always_long_count": 0, "rl_beat_always_short_count": 0, "rl_beat_always_flat_count": 0, "rl_beat_random_count": 0}}
            else:
                return {"aggregate": {"mean_total_return": 0.0, "mean_sharpe": 0.0, "worst_max_drawdown": 0.0, "positive_fold_count": 0, "robustness_score": 0.0}}
        mock_wf.side_effect = wf_side_effect
        
        result = run_ppo_std_experiment("btc_usdt", fake_config, ["current"])
        
        exp_dir = Path(result["experiment_dir"])
        assert (exp_dir / "audit_manifest.json").exists()
        assert (exp_dir / "summary.csv").exists()
        assert (exp_dir / "summary.json").exists()
        assert (exp_dir / "report.md").exists()
        
        with open(exp_dir / "audit_manifest.json") as f:
            manifest = json.load(f)
            assert manifest["experiment_type"] == "ppo-std"
            assert "current" in manifest["presets"]
            assert len(manifest["presets_data"]) == 1
            assert manifest["presets_data"][0]["preset_name"] == "current"
            assert "checkpoint_path" in manifest["presets_data"][0]
            
        with open(exp_dir / "summary.json") as f:
            summary = json.load(f)
            assert len(summary) == 1
            assert summary[0]["preset"] == "current"
            assert "deterministic_return" in summary[0]
            assert "evaluation_mode" in summary[0]
            assert summary[0]["evaluation_mode"] == "deterministic_full_period"
            
    tun.EXPERIMENTS_DIR = original_exp_dir


def test_feature_ablation_reporting_manifest_and_summary(tmp_path):
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

    from unittest.mock import patch, MagicMock

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
                "flat_ratio": 0.5,
                "flat_ratio_001": 0.1,
                "flat_ratio_005": 0.2,
                "flat_ratio_010": 0.3,
                "flat_ratio_025": 0.5,
                "long_ratio": 0.3,
                "short_ratio": 0.2,
                "long_ratio_001": 0.0,
                "short_ratio_001": 0.9,
                "long_ratio_005": 0.0,
                "short_ratio_005": 0.8,
                "long_ratio_010": 0.0,
                "short_ratio_010": 0.7,
                "long_ratio_025": 0.3,
                "short_ratio_025": 0.2,
                "dominant_action_side": "mostly_short",
                "action_mean": -0.1,
                "action_std": 0.01,
                "action_min": -0.2,
                "action_max": -0.05,
                "average_abs_action": 0.4,
                "action_abs_mean": 0.4,
                "action_abs_median": 0.4,
                "action_abs_p75": 0.4,
                "action_abs_p90": 0.4,
                "action_abs_p95": 0.4,
                "action_abs_p99": 0.4,
                "positive_action_ratio": 0.0,
                "negative_action_ratio": 1.0,
                "turnover": 1.0,
                "policy_std_mean": 0.5,
            }
        }

        def wf_side_effect(asset, config, checkpoint, folds, output_dir, include_baselines, processed_dataset):
            if include_baselines:
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
        exp_dir = Path(result["experiment_dir"])
        assert (exp_dir / "summary.csv").exists()
        assert (exp_dir / "summary.json").exists()
        assert (exp_dir / "report.md").exists()
        assert (exp_dir / "audit_manifest.json").exists()

        with open(exp_dir / "summary.json", "r", encoding="utf-8") as handle:
            summary = json.load(handle)
            assert summary[0]["feature_count"] == 2
            assert summary[0]["features"] == "log_return,trend"

        with open(exp_dir / "audit_manifest.json", "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
            assert manifest["experiment_type"] == "feature-ablation"
            assert manifest["presets_data"][0]["input_dim"] == 2

    exp.EXPERIMENTS_DIR = original_exp_dir


def test_seed_validation_reporting_manifest_and_summary(tmp_path):
    import src.experiments.seed_validation as exp

    original_exp_dir = exp.EXPERIMENTS_DIR
    exp.EXPERIMENTS_DIR = tmp_path

    fake_config = {
        "data": {"window_size": 20, "train_split": 0.8},
        "training": {"iterations": 2, "episode_length": 128, "rollout_steps": 128, "seed": 42},
        "environment": {"transaction_cost": 0.0},
        "ppo": {},
        "features": {"selected": ["log_return", "trend"]},
    }

    from unittest.mock import MagicMock, patch

    fake_dataset = MagicMock()
    fake_dataset.metadata = {"window_size": 20, "train_rows": 10, "test_rows": 5}

    with patch("src.experiments.seed_validation.build_processed_dataset", return_value=fake_dataset), \
         patch("src.experiments.seed_validation.train_asset"), \
         patch("src.experiments.seed_validation.evaluate_asset") as mock_eval, \
         patch("src.experiments.seed_validation.collect_model_diagnostics") as mock_diag, \
         patch("src.experiments.seed_validation.evaluate_walk_forward_asset") as mock_wf, \
         patch("src.experiments.seed_validation.apply_feature_preset_to_config", return_value=fake_config), \
         patch("src.experiments.seed_validation.validate_requested_feature_ablation_presets", return_value=[{"description": "test", "features": ["log_return", "trend"]}]):

        mock_eval.return_value = (
            {"rl_policy": {"total_return": 0.1, "sharpe": 0.5, "max_drawdown": 0.02, "final_equity": 1.1}},
            None,
            None,
        )
        mock_diag.return_value = {
            "summary": {
                "flat_ratio": 0.5,
                "flat_ratio_001": 0.1,
                "flat_ratio_005": 0.2,
                "flat_ratio_010": 0.3,
                "flat_ratio_025": 0.5,
                "long_ratio": 0.3,
                "short_ratio": 0.2,
                "long_ratio_001": 0.0,
                "short_ratio_001": 0.9,
                "long_ratio_005": 0.0,
                "short_ratio_005": 0.8,
                "long_ratio_010": 0.0,
                "short_ratio_010": 0.7,
                "long_ratio_025": 0.3,
                "short_ratio_025": 0.2,
                "dominant_action_side": "mostly_short",
                "action_mean": -0.1,
                "action_std": 0.01,
                "action_min": -0.2,
                "action_max": -0.05,
                "average_abs_action": 0.4,
                "action_abs_mean": 0.4,
                "action_abs_median": 0.4,
                "action_abs_p75": 0.4,
                "action_abs_p90": 0.4,
                "action_abs_p95": 0.4,
                "action_abs_p99": 0.4,
                "positive_action_ratio": 0.0,
                "negative_action_ratio": 1.0,
                "turnover": 1.0,
                "policy_std_mean": 0.5,
            }
        }

        def wf_side_effect(asset, config, checkpoint, folds, output_dir, include_baselines, processed_dataset):
            if include_baselines:
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

        result = run_seed_validation_experiment(
            "btc_usdt",
            fake_config,
            feature_presets=["full_features"],
            seeds=[42, 43],
            quick=True,
        )
        exp_dir = Path(result["experiment_dir"])
        assert (exp_dir / "runs.csv").exists()
        assert (exp_dir / "runs.json").exists()
        assert (exp_dir / "aggregate_summary.csv").exists()
        assert (exp_dir / "aggregate_summary.json").exists()
        assert (exp_dir / "report.md").exists()
        assert (exp_dir / "audit_manifest.json").exists()

        with open(exp_dir / "runs.json", "r", encoding="utf-8") as handle:
            runs = json.load(handle)
            assert len(runs) == 2
            assert runs[0]["feature_preset"] == "full_features"
            assert "seed" in runs[0]

        with open(exp_dir / "aggregate_summary.json", "r", encoding="utf-8") as handle:
            aggregate = json.load(handle)
            assert aggregate["aggregate_rows"][0]["num_seeds"] == 2

        with open(exp_dir / "audit_manifest.json", "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
            assert manifest["experiment_type"] == "seed-validation"
            assert manifest["feature_presets"] == ["full_features"]
            assert manifest["seeds"] == [42, 43]
            assert "seed_42/checkpoint.pth" in manifest["runs_data"][0]["checkpoint_path"]
            assert manifest["runs_data"][0]["input_dim"] == 2

    exp.EXPERIMENTS_DIR = original_exp_dir

def test_objective_calibration_reporting_manifest_and_summary(tmp_path):
    import src.experiments.objective_calibration as exp
    original_exp_dir = exp.EXPERIMENTS_DIR
    exp.EXPERIMENTS_DIR = tmp_path
    
    fake_config = {
        "training": {"iterations": 2, "episode_length": 128, "rollout_steps": 128},
        "environment": {"transaction_cost": 0.0},
        "ppo": {},
        "data": {"window_size": 20, "train_split": 0.8},
        "features": {"selected": ["close"]}
    }
    
    from unittest.mock import patch, MagicMock
    
    with patch("src.experiments.objective_calibration.train_asset") as mock_train, \
         patch("src.experiments.objective_calibration.collect_model_diagnostics") as mock_diag, \
         patch("src.experiments.objective_calibration.evaluate_walk_forward_asset") as mock_wf, \
         patch("src.experiments.objective_calibration.build_processed_dataset", return_value=MagicMock()) as mock_build, \
         patch("src.experiments.objective_calibration.apply_objective_preset_to_config", return_value=fake_config):
         
        mock_diag.return_value = {
            "summary": {
                "total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0,
                "action_mean": 0.0, "action_abs_mean": 0.0, "flat_ratio_001": 0.0,
                "flat_ratio_005": 0.0, "flat_ratio_010": 0.0, "flat_ratio_025": 0.0,
                "dominant_action_side": "flat", "positive_action_ratio": 0.0, "negative_action_ratio": 0.0,
                "turnover": 0.0, "transaction_cost_sum": 0.0,
            }
        }
        
        def wf_side_effect(asset, config, checkpoint, folds, output_dir, include_baselines, processed_dataset=None):
            if include_baselines:
                return {
                    "baseline_aggregate": {
                        "rl_beat_constant_signed_mean_action_count": 1,
                        "rl_beat_constant_abs_mean_short_count": 2,
                        "rl_beat_constant_abs_mean_long_count": 3,
                        "rl_best_return_fold_count": 0,
                        "constant_signed_mean_action_mean_return": 0.1,
                        "constant_abs_mean_short_mean_return": 0.2,
                        "constant_abs_mean_long_mean_return": 0.3,
                    }
                }
            else:
                return {
                    "aggregate": {
                        "mean_total_return": 0.0, "mean_sharpe": 0.0, "positive_fold_count": 0
                    }
                }
        mock_wf.side_effect = wf_side_effect
        
        result = exp.run_objective_calibration_experiment("btc_usdt", fake_config, ["current"])
        
        exp_dir = Path(result["experiment_dir"])
        assert (exp_dir / "audit_manifest.json").exists()
        assert (exp_dir / "summary.json").exists()
        assert (exp_dir / "report.md").exists()
        
        with open(exp_dir / "summary.json") as f:
            summary = json.load(f)
            assert len(summary) == 1
            row = summary[0]
            assert "rl_beat_constant_signed_mean_action_count" in row
            assert row["rl_beat_constant_signed_mean_action_count"] == 1
            assert row["rl_beat_constant_abs_mean_short_count"] == 2
            assert row["rl_beat_constant_abs_mean_long_count"] == 3
            assert row["constant_signed_mean_action_mean_return"] == 0.1
            
    exp.EXPERIMENTS_DIR = original_exp_dir
