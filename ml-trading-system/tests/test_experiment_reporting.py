import json
import pytest
from pathlib import Path
from src.experiments.ppo_std_tuning import run_ppo_std_experiment

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
