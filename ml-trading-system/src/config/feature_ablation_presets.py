from __future__ import annotations

import copy
import json
from pathlib import Path

from src.config.paths import CONFIG_DIR
from src.features.pipeline import FEATURE_COLUMNS, resolve_selected_features


def load_feature_ablation_presets(path: str | Path | None = None) -> dict:
    preset_path = Path(path) if path else CONFIG_DIR / "feature_ablation_presets.yaml"
    if not preset_path.exists():
        raise FileNotFoundError(f"Feature ablation presets file not found: {preset_path}")
    with preset_path.open("r", encoding="utf-8") as handle:
        presets = json.load(handle) or {}
    return presets


def validate_feature_preset(preset_name: str, preset: dict) -> dict:
    if "features" not in preset:
        raise ValueError(f"Feature ablation preset '{preset_name}' is missing 'features'.")
    selected_features = resolve_selected_features(preset["features"])
    return {
        "description": preset.get("description", ""),
        "features": selected_features,
    }


def available_feature_ablation_presets(path: str | Path | None = None) -> list[str]:
    return sorted(load_feature_ablation_presets(path).keys())


def resolve_feature_ablation_preset(
    preset_name: str, path: str | Path | None = None
) -> dict:
    presets = load_feature_ablation_presets(path)
    if preset_name not in presets:
        available = ", ".join(sorted(presets.keys()))
        raise ValueError(
            f"Unknown feature ablation preset: {preset_name}\n"
            f"Available presets: {available}"
        )
    return validate_feature_preset(preset_name, presets[preset_name])


def apply_feature_preset_to_config(
    config: dict, preset_name: str, path: str | Path | None = None
) -> dict:
    preset = resolve_feature_ablation_preset(preset_name, path)
    new_config = copy.deepcopy(config)
    new_config.setdefault("features", {})
    new_config["features"]["selected"] = list(preset["features"])
    return new_config


def default_full_feature_preset() -> dict:
    return {
        "description": "Current default feature set.",
        "features": list(FEATURE_COLUMNS),
    }


def validate_requested_feature_ablation_presets(
    preset_names: list[str], path: str | Path | None = None
) -> list[dict]:
    return [resolve_feature_ablation_preset(name, path) for name in preset_names]
