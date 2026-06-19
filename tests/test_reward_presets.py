import pytest
from src.config.reward_presets import load_reward_presets, resolve_reward_preset, apply_reward_preset_to_config

def test_load_reward_presets():
    presets = load_reward_presets()
    assert "current" in presets
    assert presets["current"]["reward_scale"] == 50.0

def test_preset_inheritance():
    presets = load_reward_presets()
    assert "no_action_change_penalty" in presets
    assert presets["no_action_change_penalty"]["action_change_penalty"] == 0.0
    # Must inherit transaction_cost
    assert presets["no_action_change_penalty"]["transaction_cost"] == 0.0004

def test_unknown_preset_raises_error():
    with pytest.raises(ValueError, match="Unknown reward preset"):
        resolve_reward_preset("non_existent_preset")

def test_apply_preset():
    config = {"environment": {}}
    new_config = apply_reward_preset_to_config(config, "low_position_penalty")
    assert new_config["environment"]["position_penalty"] == 0.005
    assert new_config["environment"]["transaction_cost"] == 0.0004
