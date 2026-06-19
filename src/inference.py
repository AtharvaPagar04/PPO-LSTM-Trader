from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config.assets import asset_to_symbol, normalize_asset_name
from src.config.paths import (
    BASE_DIR,
    PREDICTIONS_DIR,
    ensure_dir,
    resolve_checkpoint_path,
    resolve_existing_raw_data_path,
)
from src.data.dataset import load_metadata, load_scaler
from src.evaluation.benchmark import load_policy_from_checkpoint
from src.features.pipeline import FEATURE_COLUMNS, engineer_features
from src.utils.logger import utc_timestamp_slug, write_json
from src.utils.seed import set_global_seed


REQUIRED_CSV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class PredictionResult:
    asset: str
    timestamp: str
    window_size: int
    model_action: float
    position_label: str
    action_std: float
    value_estimate: float
    checkpoint: str
    source: str
    binance_symbol: str
    status: str = "ok"


@dataclass
class PredictionError:
    asset: str
    status: str
    error: str


def position_label(action: float) -> str:
    if action <= -0.75:
        return "strong short"
    if action < -0.25:
        return "mild short"
    if action <= 0.25:
        return "flat / neutral"
    if action < 0.75:
        return "mild long"
    return "strong long"


def display_path(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def _normalize_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    missing = [column for column in REQUIRED_CSV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing required columns: " + ", ".join(missing)
        )
    return df[REQUIRED_CSV_COLUMNS]


def _load_raw_or_custom_dataframe(asset: str, csv_path: str | None = None) -> tuple[pd.DataFrame, str]:
    if csv_path:
        df = pd.read_csv(csv_path)
        return _normalize_csv_columns(df), "custom_csv"
    df = pd.read_csv(resolve_existing_raw_data_path(asset))
    return _normalize_csv_columns(df), "raw_data"


def _prepare_latest_window(asset: str, metadata: dict, scaler, csv_path: str | None = None):
    raw_df, source = _load_raw_or_custom_dataframe(asset, csv_path)
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    for column in ["open", "high", "low", "close", "volume"]:
        raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")
    raw_df = raw_df.sort_values("timestamp").dropna().reset_index(drop=True)

    features_df = engineer_features(raw_df)
    window_size = int(metadata["window_size"])
    if len(features_df) < window_size:
        raise ValueError(
            f"Not enough rows to build a prediction window: need at least {window_size}, got {len(features_df)} after feature engineering."
        )

    latest_window = features_df[FEATURE_COLUMNS].tail(window_size).to_numpy(dtype=np.float32)
    latest_window = scaler.transform(latest_window)
    latest_timestamp = features_df["timestamp"].iloc[-1]
    return latest_window[None, :, :], latest_timestamp, source


def predict_latest(
    asset: str,
    config: dict,
    checkpoint_path: str | None = None,
    csv_path: str | None = None,
) -> PredictionResult:
    asset = normalize_asset_name(asset)
    resolved_checkpoint = resolve_checkpoint_path(asset, checkpoint_path)
    metadata = load_metadata(asset)
    scaler = load_scaler(asset)
    window, latest_timestamp, source = _prepare_latest_window(
        asset, metadata, scaler, csv_path=csv_path
    )

    set_global_seed(config["evaluation"]["random_seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy_from_checkpoint(
        asset=asset,
        checkpoint_path=resolved_checkpoint,
        input_dim=window.shape[2],
        config=config,
        device=device,
    )
    model.eval()

    with torch.no_grad():
        window_tensor = torch.tensor(window, dtype=torch.float32).to(device)
        mean, std, value = model(window_tensor)
        action = float(np.clip(mean.cpu().numpy()[0][0], -1.0, 1.0))
        action_std = float(std.cpu().numpy()[0][0])
        value_estimate = float(value.cpu().numpy()[0][0])

    return PredictionResult(
        asset=asset,
        timestamp=pd.Timestamp(latest_timestamp).isoformat(),
        window_size=int(metadata["window_size"]),
        model_action=action,
        position_label=position_label(action),
        action_std=action_std,
        value_estimate=value_estimate,
        checkpoint=display_path(resolved_checkpoint),
        source=source,
        binance_symbol=asset_to_symbol(asset),
    )


def format_prediction_text(result: PredictionResult) -> str:
    timestamp = pd.Timestamp(result.timestamp)
    return "\n".join(
        [
            f"Asset: {result.asset}",
            f"Window size: {result.window_size}",
            f"Latest timestamp: {timestamp}",
            f"Model action: {result.model_action:.4f}",
            f"Position interpretation: {result.position_label}",
            f"Action std: {result.action_std:.4f}",
            f"Value estimate: {result.value_estimate:.4f}",
            f"Checkpoint: {result.checkpoint}",
            f"Source: {result.source}",
            "Note: This is an experimental model inference output, not an execution instruction.",
        ]
    )


def format_prediction_json(result: PredictionResult | dict[str, Any]) -> str:
    payload = asdict(result) if isinstance(result, PredictionResult) else result
    return json.dumps(payload, indent=2)


def format_prediction_table(results: list[PredictionResult], errors: list[PredictionError]) -> str:
    headers = ["Asset", "Action", "Position Label", "Std", "Value", "Timestamp"]
    rows = []
    for result in results:
        rows.append(
            [
                result.asset,
                f"{result.model_action:.4f}",
                result.position_label,
                f"{result.action_std:.4f}",
                f"{result.value_estimate:.4f}",
                str(pd.Timestamp(result.timestamp)),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    lines = [
        "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    ]
    for row in rows:
        lines.append("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))

    if errors:
        if rows:
            lines.append("")
        lines.append("Errors:")
        for error in errors:
            lines.append(f"- {error.asset}: {error.error}")

    lines.append("")
    lines.append(
        "Note: These are experimental model inference outputs, not execution instructions."
    )
    return "\n".join(lines)


def save_prediction(result: PredictionResult) -> Path:
    ensure_dir(PREDICTIONS_DIR)
    path = PREDICTIONS_DIR / f"{utc_timestamp_slug()}_{result.asset}_prediction.json"
    write_json(path, asdict(result))
    return path


def predict_many(
    assets: list[str],
    *,
    config: dict,
    checkpoint_path: str | None = None,
) -> tuple[list[PredictionResult], list[PredictionError]]:
    results: list[PredictionResult] = []
    errors: list[PredictionError] = []

    for asset in assets:
        normalized = normalize_asset_name(asset)
        try:
            result = predict_latest(
                asset=normalized,
                config=config,
                checkpoint_path=checkpoint_path,
            )
        except (FileNotFoundError, ValueError) as exc:
            errors.append(
                PredictionError(asset=normalized, status="error", error=str(exc))
            )
            continue
        results.append(result)

    return results, errors


def format_prediction_many_json(
    results: list[PredictionResult], errors: list[PredictionError]
) -> str:
    payload = {
        "mode": "predict_all",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "predictions": [asdict(result) for result in results],
        "errors": [asdict(error) for error in errors],
    }
    return json.dumps(payload, indent=2)


def save_predictions(results: list[PredictionResult], errors: list[PredictionError]) -> Path:
    ensure_dir(PREDICTIONS_DIR)
    path = PREDICTIONS_DIR / f"{utc_timestamp_slug()}_all_predictions.json"
    payload = {
        "mode": "predict_all",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "predictions": [asdict(result) for result in results],
        "errors": [asdict(error) for error in errors],
    }
    write_json(path, payload)
    return path
