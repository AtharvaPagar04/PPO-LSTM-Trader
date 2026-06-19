from __future__ import annotations

import json

import joblib
import numpy as np

from src.config.assets import normalize_asset_name
from src.config.paths import resolve_processed_artifact_path


def load_processed_data(asset: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    asset = normalize_asset_name(asset)
    features = np.load(resolve_processed_artifact_path(asset, f"{split}_windows.npy"))
    prices = np.load(
        resolve_processed_artifact_path(asset, f"{split}_price_windows.npy")
    )
    return features, prices


def load_metadata(asset: str) -> dict:
    with resolve_processed_artifact_path(asset, "meta.json").open(
        "r", encoding="utf-8"
    ) as handle:
        return json.load(handle)


def load_scaler(asset: str):
    return joblib.load(resolve_processed_artifact_path(asset, "scaler.pkl"))
