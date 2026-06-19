import json
import subprocess
from pathlib import Path


def run_cli(*args):
    return subprocess.run(
        ["./venv/bin/python", "-m", "src.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_help_works():
    result = run_cli("--help")
    assert result.returncode == 0
    assert "train" in result.stdout
    assert "evaluate" in result.stdout
    assert "predict" in result.stdout


def test_cli_subcommand_help_works():
    for command in ["train", "evaluate", "predict"]:
        result = run_cli(command, "--help")
        assert result.returncode == 0


def test_invalid_asset_gives_clear_error():
    result = run_cli("predict", "--asset", "doge_usdt")
    assert result.returncode != 0
    assert "Unsupported asset" in result.stderr


def test_missing_checkpoint_gives_clear_error():
    result = run_cli("predict", "--asset", "btc_usdt", "--checkpoint", "models/does_not_exist.pt")
    assert result.returncode != 0
    assert "Checkpoint not found" in result.stderr


def test_predict_json_outputs_valid_json():
    result = run_cli("predict", "--asset", "btc_usdt", "--format", "json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["asset"] == "btc_usdt"
    assert "model_action" in payload


def test_predict_help_shows_all_option():
    result = run_cli("predict", "--help")
    assert result.returncode == 0
    assert "--all" in result.stdout


def test_predict_all_json_outputs_valid_json():
    result = run_cli("predict", "--all", "--format", "json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "predict_all"
    assert isinstance(payload["predictions"], list)
    assert "errors" in payload


def test_predict_all_save_writes_combined_json():
    before = {path.name for path in Path("logs/predictions").glob("*_all_predictions.json")}
    result = run_cli("predict", "--all", "--save")
    assert result.returncode == 0
    after = {path.name for path in Path("logs/predictions").glob("*_all_predictions.json")}
    created = after - before
    assert created


def test_invalid_selector_combination_fails_clearly():
    result = run_cli("predict", "--asset", "btc_usdt", "--all")
    assert result.returncode != 0
    assert "Use only one of --asset, --assets, or --all." in result.stderr


def test_predict_all_with_csv_fails_clearly():
    result = run_cli("predict", "--all", "--csv", "data/raw/BTCUSDT_1h.csv")
    assert result.returncode != 0
    assert "--csv is only supported with --asset." in result.stderr


def test_predict_without_selector_fails_clearly():
    result = run_cli("predict")
    assert result.returncode != 0
    assert "Provide one of --asset, --assets, or --all." in result.stderr


def test_predict_assets_subset_works():
    result = run_cli("predict", "--assets", "btc_usdt", "eth_usdt", "--format", "json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    predicted_assets = {entry["asset"] for entry in payload["predictions"]}
    assert predicted_assets == {"btc_usdt", "eth_usdt"}
