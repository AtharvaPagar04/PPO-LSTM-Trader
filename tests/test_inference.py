import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from src.config.settings import load_config
from src.inference import (
    PredictionError,
    PredictionResult,
    display_path,
    format_prediction_many_json,
    format_prediction_table,
    position_label,
    predict_latest,
    predict_many,
)
from src.models.policy import LSTMPolicy


def make_csv(path: Path, rows=80):
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h")
    close = np.linspace(100.0, 140.0, rows)
    df = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.linspace(1000.0, 2000.0, rows),
        }
    )
    df.to_csv(path, index=False)


def test_position_label_mapping():
    assert position_label(-0.8) == "strong short"
    assert position_label(-0.5) == "mild short"
    assert position_label(0.0) == "flat / neutral"
    assert position_label(0.5) == "mild long"
    assert position_label(0.9) == "strong long"


def test_predict_latest_returns_required_keys(tmp_path):
    from sklearn.preprocessing import StandardScaler
    from src.config import paths as config_paths
    from src.data import dataset as dataset_module

    asset = "btc_usdt"
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    models_dir.mkdir()

    scaler = StandardScaler().fit(np.random.randn(200, 10))
    x = np.random.randn(12, 20, 10).astype(np.float32)
    prices = np.random.randn(12, 20, 5).astype(np.float32)
    np.save(processed_dir / f"{asset}_test_windows.npy", x)
    np.save(processed_dir / f"{asset}_test_price_windows.npy", prices)
    np.save(processed_dir / f"{asset}_train_windows.npy", x)
    np.save(processed_dir / f"{asset}_train_price_windows.npy", prices)
    torch.save(LSTMPolicy(input_dim=10).state_dict(), models_dir / f"{asset}_best.pt")

    with (processed_dir / f"{asset}_meta.json").open("w", encoding="utf-8") as handle:
        json.dump({"asset": asset, "window_size": 20, "features": [f"f{i}" for i in range(10)], "split_ratio": 0.8}, handle)
    import joblib
    joblib.dump(scaler, processed_dir / f"{asset}_scaler.pkl")

    csv_path = tmp_path / "sample.csv"
    make_csv(csv_path)

    dataset_module.resolve_processed_artifact_path = lambda a, suffix: processed_dir / f"{asset}_{suffix}"
    config_paths.resolve_processed_artifact_path = lambda a, suffix: processed_dir / f"{asset}_{suffix}"
    config_paths.resolve_checkpoint_path = lambda a, checkpoint=None: models_dir / f"{asset}_best.pt"

    result = predict_latest(
        asset=asset,
        config=load_config(),
        checkpoint_path=str(models_dir / f"{asset}_best.pt"),
        csv_path=str(csv_path),
    )
    payload = result.__dict__
    assert {"asset", "timestamp", "window_size", "model_action", "position_label", "action_std", "value_estimate", "checkpoint", "source", "binance_symbol"} <= set(payload)
    assert -1.0 <= payload["model_action"] <= 1.0


def test_predict_latest_missing_columns_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": [1], "open": [1]}).to_csv(csv_path, index=False)
    from sklearn.preprocessing import StandardScaler
    import joblib
    asset = "btc_usdt"
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    models_dir.mkdir()
    with (processed_dir / f"{asset}_meta.json").open("w", encoding="utf-8") as handle:
        json.dump({"asset": asset, "window_size": 20, "features": [f"f{i}" for i in range(10)], "split_ratio": 0.8}, handle)
    joblib.dump(StandardScaler().fit(np.random.randn(200, 10)), processed_dir / f"{asset}_scaler.pkl")
    torch.save(LSTMPolicy(input_dim=10).state_dict(), models_dir / f"{asset}_best.pt")

    from src.data import dataset as dataset_module
    from src.config import paths as config_paths
    dataset_module.resolve_processed_artifact_path = lambda a, suffix: processed_dir / f"{asset}_{suffix}"
    config_paths.resolve_processed_artifact_path = lambda a, suffix: processed_dir / f"{asset}_{suffix}"

    try:
        predict_latest(asset=asset, config=load_config(), checkpoint_path=str(models_dir / f"{asset}_best.pt"), csv_path=str(csv_path))
    except ValueError as exc:
        assert "missing required columns" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing CSV columns")


def test_predict_latest_is_deterministic_for_same_input(tmp_path):
    from sklearn.preprocessing import StandardScaler
    import joblib
    from src.data import dataset as dataset_module
    from src.config import paths as config_paths

    asset = "btc_usdt"
    processed_dir = tmp_path / "processed"
    models_dir = tmp_path / "models"
    processed_dir.mkdir()
    models_dir.mkdir()
    with (processed_dir / f"{asset}_meta.json").open("w", encoding="utf-8") as handle:
        json.dump({"asset": asset, "window_size": 20, "features": [f"f{i}" for i in range(10)], "split_ratio": 0.8}, handle)
    joblib.dump(StandardScaler().fit(np.random.randn(200, 10)), processed_dir / f"{asset}_scaler.pkl")
    torch.save(LSTMPolicy(input_dim=10).state_dict(), models_dir / f"{asset}_best.pt")

    dataset_module.resolve_processed_artifact_path = lambda a, suffix: processed_dir / f"{asset}_{suffix}"
    config_paths.resolve_processed_artifact_path = lambda a, suffix: processed_dir / f"{asset}_{suffix}"
    csv_path = tmp_path / "sample.csv"
    make_csv(csv_path)
    result1 = predict_latest(asset=asset, config=load_config(), checkpoint_path=str(models_dir / f"{asset}_best.pt"), csv_path=str(csv_path))
    result2 = predict_latest(asset=asset, config=load_config(), checkpoint_path=str(models_dir / f"{asset}_best.pt"), csv_path=str(csv_path))
    assert result1.model_action == result2.model_action
    assert result1.action_std == result2.action_std


def test_predict_many_returns_one_result_per_asset_and_continues_on_error():
    ok_result = PredictionResult(
        asset="btc_usdt",
        timestamp="2026-01-01T00:00:00",
        window_size=20,
        model_action=0.1,
        position_label="flat / neutral",
        action_std=0.2,
        value_estimate=0.3,
        checkpoint="models/btc_usdt_model.pth",
        source="raw_data",
        binance_symbol="BTCUSDT",
    )

    def fake_predict_latest(asset, config, checkpoint_path=None, csv_path=None):
        if asset == "eth_usdt":
            raise FileNotFoundError("Checkpoint not found")
        return ok_result if asset == "btc_usdt" else PredictionResult(
            asset="sol_usdt",
            timestamp="2026-01-01T00:00:00",
            window_size=20,
            model_action=-0.4,
            position_label="mild short",
            action_std=0.2,
            value_estimate=0.1,
            checkpoint="models/sol_usdt_model.pth",
            source="raw_data",
            binance_symbol="SOLUSDT",
        )

    with patch("src.inference.predict_latest", side_effect=fake_predict_latest):
        results, errors = predict_many(
            ["btc_usdt", "eth_usdt", "sol_usdt"], config=load_config()
        )

    assert len(results) == 2
    assert len(errors) == 1
    assert errors[0].status == "error"
    assert results[0].status == "ok"


def test_format_prediction_table_handles_multiple_assets():
    results = [
        PredictionResult(
            asset="btc_usdt",
            timestamp="2026-01-01T00:00:00",
            window_size=20,
            model_action=0.1,
            position_label="flat / neutral",
            action_std=0.2,
            value_estimate=0.3,
            checkpoint="models/btc_usdt_model.pth",
            source="raw_data",
            binance_symbol="BTCUSDT",
        ),
        PredictionResult(
            asset="eth_usdt",
            timestamp="2026-01-01T00:00:00",
            window_size=20,
            model_action=0.4,
            position_label="mild long",
            action_std=0.5,
            value_estimate=-0.2,
            checkpoint="models/eth_usdt_model.pth",
            source="raw_data",
            binance_symbol="ETHUSDT",
        ),
    ]
    errors = [PredictionError(asset="sol_usdt", status="error", error="Checkpoint not found")]
    table = format_prediction_table(results, errors)
    assert "btc_usdt" in table
    assert "eth_usdt" in table
    assert "Errors:" in table
    assert "sol_usdt" in table


def test_format_prediction_many_json_outputs_valid_json():
    results = [
        PredictionResult(
            asset="btc_usdt",
            timestamp="2026-01-01T00:00:00",
            window_size=20,
            model_action=0.1,
            position_label="flat / neutral",
            action_std=0.2,
            value_estimate=0.3,
            checkpoint="models/btc_usdt_model.pth",
            source="raw_data",
            binance_symbol="BTCUSDT",
        )
    ]
    errors = [PredictionError(asset="sol_usdt", status="error", error="missing checkpoint")]
    payload = json.loads(format_prediction_many_json(results, errors))
    assert payload["mode"] == "predict_all"
    assert payload["predictions"][0]["status"] == "ok"
    assert payload["errors"][0]["status"] == "error"


def test_display_path_prefers_repo_relative_paths():
    path = display_path(Path.cwd() / "models" / "btc_usdt_model.pth")
    assert path == "models/btc_usdt_model.pth"
