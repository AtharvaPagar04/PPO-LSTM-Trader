from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.config.paths import CONFIG_DIR


DEFAULT_CONFIG_FILES = [
    CONFIG_DIR / "default.yaml",
    CONFIG_DIR / "ppo_default.yaml",
    CONFIG_DIR / "assets.yaml",
]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(*extra_paths: str | Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for path in [*DEFAULT_CONFIG_FILES, *[Path(p) for p in extra_paths]]:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        config = _deep_merge(config, loaded)
    return config
