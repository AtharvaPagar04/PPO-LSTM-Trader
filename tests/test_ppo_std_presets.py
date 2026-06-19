import pytest
from src.config.ppo_std_presets import load_ppo_std_presets, resolve_ppo_std_preset, apply_ppo_std_preset_to_config

def test_load_ppo_std_presets():
    presets = load_ppo_std_presets()
    assert "current" in presets
    assert presets["current"]["log_std_max"] == -0.2

def test_preset_inheritance():
    presets = load_ppo_std_presets()
    assert "low_entropy" in presets
    assert presets["low_entropy"]["entropy_coef"] == 0.0025
    assert presets["low_entropy"]["log_std_max"] == -0.2

def test_unknown_preset_raises_error():
    with pytest.raises(ValueError, match="Unknown PPO std preset"):
        resolve_ppo_std_preset("non_existent_preset")

def test_apply_preset():
    config = {"ppo": {}}
    new_config = apply_ppo_std_preset_to_config(config, "lower_std_ceiling")
    assert new_config["ppo"]["log_std_max"] == -0.7
    assert new_config["ppo"]["entropy_coef"] == 0.01

def test_apply_preset_preserves_other_config():
    config = {"training": {"seed": 42}, "ppo": {"gamma": 0.99}}
    new_config = apply_ppo_std_preset_to_config(config, "lower_std_ceiling")
    assert new_config["training"]["seed"] == 42
    assert new_config["ppo"]["gamma"] == 0.99


def test_smooth_std_bound_preset_resolves_correctly():
    preset = resolve_ppo_std_preset("smooth_std_bound")
    assert preset["std_parameterization"] == "smooth_bound"
    assert preset["log_std_max"] == -0.2
