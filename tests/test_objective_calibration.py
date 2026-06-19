import pytest
from src.config.objective_presets import load_objective_presets, apply_objective_preset_to_config
from src.experiments.objective_calibration import calculate_calibration_score

def test_objective_calibration_presets_load():
    presets = load_objective_presets()
    assert "current" in presets
    assert "exposure_penalty_light" in presets
    assert "directional_edge_reward" in presets

def test_unknown_objective_preset_fails_clearly():
    with pytest.raises(ValueError, match="Unknown objective preset"):
        apply_objective_preset_to_config({}, "non_existent_preset")

def test_apply_objective_preset():
    config = {}
    new_config = apply_objective_preset_to_config(config, "exposure_penalty_light")
    assert "environment" in new_config
    assert new_config["environment"]["exposure_penalty_coef"] == 0.005

def test_score_rewards_beating_exposure_equivalent_baselines():
    score1 = calculate_calibration_score(
        rl_beat_constant_signed_mean_action_count=0,
        rl_beat_constant_abs_mean_short_count=0,
        rl_best_return_fold_count=0,
        walk_forward_mean_sharpe=0.0,
        deterministic_max_drawdown=0.0,
        action_mean=0.0
    )
    score2 = calculate_calibration_score(
        rl_beat_constant_signed_mean_action_count=2,
        rl_beat_constant_abs_mean_short_count=1,
        rl_best_return_fold_count=0,
        walk_forward_mean_sharpe=0.0,
        deterministic_max_drawdown=0.0,
        action_mean=0.0
    )
    assert score2 > score1


def test_run_objective_calibration_passes_processed_dataset():
    from src.experiments.objective_calibration import run_objective_calibration_experiment
    from unittest.mock import patch
    
    # Mock dependencies
    with patch("src.experiments.objective_calibration.build_processed_dataset", return_value="mock_dataset") as mock_build, \
         patch("src.experiments.objective_calibration.train_asset") as mock_train, \
         patch("src.experiments.objective_calibration.collect_model_diagnostics", return_value={"summary": {"total_return": 0, "sharpe": 0, "max_drawdown": 0, "flat_ratio_001": 0, "flat_ratio_005": 0, "flat_ratio_010": 0, "flat_ratio_025": 0, "dominant_action_side": "flat", "action_mean": 0, "action_abs_mean": 0, "positive_action_ratio": 0, "negative_action_ratio": 0, "turnover": 0}}) as mock_diag, \
         patch("src.experiments.objective_calibration.evaluate_walk_forward_asset", side_effect=[
             {"aggregate": {"mean_total_return": 0, "mean_sharpe": 0, "positive_fold_count": 0}},
             {"baseline_aggregate": {}}
         ]) as mock_wf:
    
        config = {
            "data": {"window_size": 20, "train_split": 0.8},
            "features": {"selected": ["log_return"]},
            "training": {}
        }
    
        run_objective_calibration_experiment(
            asset="btc_usdt",
            config=config,
            presets=["current"],
            feature_preset="price_action_minimal",
            quick=True
        )
        
        assert mock_build.called
        assert mock_train.call_args[1]["processed_dataset"] == "mock_dataset"
        assert mock_diag.call_args[1]["processed_dataset"] == "mock_dataset"
        assert mock_wf.call_args_list[0][1]["processed_dataset"] == "mock_dataset"
        assert mock_wf.call_args_list[1][1]["processed_dataset"] == "mock_dataset"


def test_objective_preset_modifies_resolved_config():
    """Each non-current preset must put at least one non-zero coefficient into config['environment']."""
    base = {"environment": {"transaction_cost": 0.0004}}
    for preset_name in ["exposure_penalty_light", "directional_edge_reward", "timing_calibration_combo"]:
        resolved = apply_objective_preset_to_config(base, preset_name)
        env = resolved["environment"]
        objective_keys = ["exposure_penalty_coef", "turnover_penalty_coef", "directional_reward_coef", "volatility_exposure_penalty_coef"]
        active = {k: env[k] for k in objective_keys if env.get(k, 0.0) != 0.0}
        assert active, f"Preset '{preset_name}' has no active coefficients in resolved config"


def test_current_preset_leaves_coefficients_zero():
    base = {"environment": {"transaction_cost": 0.0004}}
    resolved = apply_objective_preset_to_config(base, "current")
    env = resolved["environment"]
    objective_keys = ["exposure_penalty_coef", "turnover_penalty_coef", "directional_reward_coef", "volatility_exposure_penalty_coef"]
    for k in objective_keys:
        assert env.get(k, 0.0) == 0.0, f"current preset should not set {k}"


def test_build_env_forwards_objective_coefficients():
    import numpy as np
    from src.train import build_env
    config = {
        "environment": {
            "transaction_cost": 0.0004,
            "drawdown_penalty": 0.1,
            "position_penalty": 0.05,
            "action_change_penalty": 0.001,
            "reward_scale": 50.0,
            "reward_clip": 5.0,
            "exposure_penalty_coef": 0.005,
            "directional_reward_coef": 0.03,
        },
        "training": {"episode_length": 128},
    }
    env = build_env(np.zeros((10, 5, 2)), np.zeros((10, 5, 5)), config)
    assert env.exposure_penalty_coef == 0.005
    assert env.directional_reward_coef == 0.03
    assert env.turnover_penalty_coef == 0.0  # default
    assert env.volatility_exposure_penalty_coef == 0.0  # default


def test_build_eval_env_forwards_objective_coefficients():
    import numpy as np
    from src.evaluation.benchmark import build_eval_env
    config = {
        "environment": {
            "transaction_cost": 0.0004,
            "drawdown_penalty": 0.1,
            "position_penalty": 0.05,
            "action_change_penalty": 0.001,
            "reward_scale": 50.0,
            "reward_clip": 5.0,
            "volatility_exposure_penalty_coef": 0.01,
            "turnover_penalty_coef": 0.002,
        },
        "training": {"episode_length": 128},
    }
    env = build_eval_env(np.zeros((10, 5, 2)), np.zeros((10, 5, 5)), config)
    assert env.volatility_exposure_penalty_coef == 0.01
    assert env.turnover_penalty_coef == 0.002
    assert env.exposure_penalty_coef == 0.0  # default
    assert env.directional_reward_coef == 0.0  # default


def test_experiment_runner_rejects_zero_coefficient_preset():
    from src.experiments.objective_calibration import run_objective_calibration_experiment
    from unittest.mock import patch

    # Create a fake preset file with a non-current preset that has no coefficients
    fake_presets = {
        "current": {"description": "test"},
        "broken_preset": {"description": "has no coefficients"},
    }

    config = {
        "data": {"window_size": 20, "train_split": 0.8},
        "features": {"selected": ["log_return"]},
        "training": {"iterations": 2, "episode_length": 128, "rollout_steps": 128},
        "environment": {},
    }

    with patch("src.config.objective_presets.load_objective_presets", return_value=fake_presets):
        with pytest.raises(ValueError, match="resolved to zero active coefficients"):
            run_objective_calibration_experiment(
                asset="btc_usdt",
                config=config,
                presets=["broken_preset"],
                feature_preset="price_action_minimal",
                quick=True,
            )


def test_report_uses_abs_action_not_pipe_act():
    """Verify report.md uses AbsAction instead of pipe-breaking |Act|."""
    from src.experiments.objective_calibration import run_objective_calibration_experiment
    from unittest.mock import patch, MagicMock
    import tempfile, json
    from pathlib import Path

    fake_diag = {"summary": {
        "total_return": 0, "sharpe": 0, "max_drawdown": 0,
        "flat_ratio_001": 0, "flat_ratio_005": 0, "flat_ratio_010": 0, "flat_ratio_025": 0,
        "dominant_action_side": "flat", "action_mean": 0, "action_abs_mean": 0,
        "positive_action_ratio": 0, "negative_action_ratio": 0, "turnover": 0,
    }}

    def wf_side_effect(asset, config, checkpoint, folds, output_dir, include_baselines, processed_dataset=None):
        if include_baselines:
            return {"baseline_aggregate": {}}
        return {"aggregate": {"mean_total_return": 0, "mean_sharpe": 0, "positive_fold_count": 0}}

    config = {
        "data": {"window_size": 20, "train_split": 0.8},
        "features": {"selected": ["log_return"]},
        "training": {},
        "environment": {},
    }

    with patch("src.experiments.objective_calibration.build_processed_dataset", return_value=MagicMock()), \
         patch("src.experiments.objective_calibration.train_asset"), \
         patch("src.experiments.objective_calibration.collect_model_diagnostics", return_value=fake_diag), \
         patch("src.experiments.objective_calibration.evaluate_walk_forward_asset", side_effect=wf_side_effect):
        result = run_objective_calibration_experiment("btc_usdt", config, ["current"], quick=True)

    report_path = Path(result["experiment_dir"]) / "report.md"
    report_text = report_path.read_text()
    assert "AbsAction" in report_text
    assert "|Act|" not in report_text

