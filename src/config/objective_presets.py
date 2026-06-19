from pathlib import Path
import json
from src.config.paths import BASE_DIR

def load_objective_presets():
    path = BASE_DIR / "configs" / "objective_calibration_presets.yaml"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_objective_preset_to_config(config: dict, preset_name: str) -> dict:
    presets = load_objective_presets()
    if preset_name not in presets:
        raise ValueError(f"Unknown objective preset: {preset_name}")
        
    preset_data = presets[preset_name]
    new_config = config.copy()
    if "environment" not in new_config:
        new_config["environment"] = {}
    
    env_cfg = new_config["environment"].copy()
    
    for key in ["exposure_penalty_coef", "turnover_penalty_coef", "directional_reward_coef", "volatility_exposure_penalty_coef"]:
        if key in preset_data:
            env_cfg[key] = preset_data[key]
            
    new_config["environment"] = env_cfg
    return new_config
