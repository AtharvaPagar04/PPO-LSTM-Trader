import json
from pathlib import Path
from src.config.paths import CONFIG_DIR

def load_reward_presets(path=None):
    if path is None:
        path = CONFIG_DIR / "reward_presets.yaml"
    if not Path(path).exists():
        raise FileNotFoundError(f"Reward presets file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        presets = json.load(f) or {}

    resolved = {}
    def _resolve(name):
        if name in resolved:
            return resolved[name]
        if name not in presets:
            raise ValueError(f"Unknown reward preset: {name}")
        config = presets[name]
        resolved_config = {}
        if "inherits" in config:
            parent_name = config["inherits"]
            resolved_config.update(_resolve(parent_name))
        for k, v in config.items():
            if k != "inherits":
                resolved_config[k] = v
        resolved[name] = resolved_config
        return resolved_config

    for name in presets:
        _resolve(name)
    return resolved

def resolve_reward_preset(preset_name, path=None):
    presets = load_reward_presets(path)
    if preset_name not in presets:
        raise ValueError(f"Unknown reward preset: {preset_name}")
    return presets[preset_name]

def apply_reward_preset_to_config(config, preset_name, path=None):
    import copy
    preset_config = resolve_reward_preset(preset_name, path)
    new_config = copy.deepcopy(config)
    if "environment" not in new_config:
        new_config["environment"] = {}
    for k, v in preset_config.items():
        new_config["environment"][k] = float(v) if isinstance(v, (int, float)) else v
    return new_config
