import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import src.cli as cli


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
    for command in ["train", "evaluate", "predict", "diagnose"]:
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


def test_evaluate_help_shows_walk_forward_option():
    result = run_cli("evaluate", "--help")
    assert result.returncode == 0
    assert "--walk-forward" in result.stdout
    assert "--baselines" in result.stdout


def test_diagnose_help_works():
    result = run_cli("diagnose", "--help")
    assert result.returncode == 0
    assert "--format" in result.stdout


def test_cli_routes_single_asset_walk_forward():
    fake_result = {
        "asset": "btc_usdt",
        "fold_rows": [],
        "aggregate": {
            "mean_total_return": 0.0,
            "mean_sharpe": 0.0,
            "worst_max_drawdown": 0.0,
            "positive_fold_count": 0,
            "total_folds": 0,
            "robustness_score": 0.0,
        },
        "output_dir": "logs/walk_forward/fake",
    }
    with patch("src.cli.evaluate_walk_forward_asset", return_value=fake_result) as mock_eval:
        exit_code = cli.main(["evaluate", "--asset", "btc_usdt", "--walk-forward"])
    assert exit_code == 0
    mock_eval.assert_called_once()


def test_cli_routes_single_asset_walk_forward_with_baselines():
    fake_result = {
        "asset": "btc_usdt",
        "fold_rows": [],
        "baseline_aggregate": {
            "rl_beat_always_long_count": 0,
            "rl_beat_always_short_count": 0,
            "rl_beat_always_flat_count": 0,
            "rl_beat_random_count": 0,
            "rl_best_return_fold_count": 0,
            "total_folds": 0,
            "best_overall_strategy_by_mean_return": "always_flat",
        },
        "output_dir": "logs/walk_forward/fake",
    }
    with patch("src.cli.evaluate_walk_forward_asset", return_value=fake_result) as mock_eval:
        exit_code = cli.main(
            ["evaluate", "--asset", "btc_usdt", "--walk-forward", "--baselines"]
        )
    assert exit_code == 0
    assert mock_eval.call_args.kwargs["include_baselines"] is True


def test_cli_routes_all_asset_walk_forward():
    fake_result = {
        "summary_rows": [],
        "output_dir": "logs/walk_forward/fake_all",
    }
    with patch("src.cli.evaluate_walk_forward_all", return_value=fake_result) as mock_eval:
        exit_code = cli.main(["evaluate", "--all", "--walk-forward"])
    assert exit_code == 0
    mock_eval.assert_called_once()


def test_cli_routes_all_asset_walk_forward_with_baselines():
    fake_result = {
        "baseline_summary_rows": [],
        "output_dir": "logs/walk_forward/fake_all",
    }
    with patch("src.cli.evaluate_walk_forward_all", return_value=fake_result) as mock_eval:
        exit_code = cli.main(["evaluate", "--all", "--walk-forward", "--baselines"])
    assert exit_code == 0
    assert mock_eval.call_args.kwargs["include_baselines"] is True


def test_invalid_fold_count_fails_clearly():
    result = run_cli("evaluate", "--asset", "btc_usdt", "--walk-forward", "--folds", "0")
    assert result.returncode != 0
    assert "folds must be a positive integer" in result.stderr


def test_baselines_without_walk_forward_fail_clearly():
    result = run_cli("evaluate", "--asset", "btc_usdt", "--baselines")
    assert result.returncode != 0
    assert "--baselines is currently supported only with --walk-forward" in result.stderr


def test_cli_routes_single_asset_diagnose():
    fake_result = {
        "mode": "diagnose",
        "summary": {
            "asset": "btc_usdt",
            "num_steps": 10,
            "start_timestamp": "2026-01-01 00:00:00",
            "end_timestamp": "2026-01-01 10:00:00",
            "checkpoint": "models/btc_usdt_model.pth",
            "action_mean": 0.0,
            "action_std": 0.1,
            "action_min": -0.1,
            "action_max": 0.1,
            "action_median": 0.0,
            "average_abs_action": 0.05,
            "long_ratio": 0.1,
            "short_ratio": 0.1,
            "flat_ratio": 0.8,
            "strong_long_ratio": 0.0,
            "strong_short_ratio": 0.0,
            "average_position": 0.0,
            "average_abs_position": 0.05,
            "position_min": -0.1,
            "position_max": 0.1,
            "turnover": 0.2,
            "mean_position_change": 0.02,
            "max_position_change": 0.1,
            "policy_std_mean": 0.8,
            "policy_std_min": 0.8,
            "policy_std_max": 0.8,
            "policy_std_median": 0.8,
            "value_mean": 0.0,
            "value_min": -1.0,
            "value_max": 1.0,
            "value_std": 0.1,
            "reward_mean": 0.0,
            "reward_std": 0.1,
            "reward_min": -0.2,
            "reward_max": 0.2,
            "pnl_sum": 0.01,
            "pnl_mean": 0.001,
            "gross_pnl_sum": 0.02,
            "transaction_cost_sum": 0.001,
            "transaction_cost_mean": 0.0001,
            "transaction_cost_drag_ratio": 0.1,
            "position_penalty_sum": 0.1,
            "position_penalty_mean": 0.01,
            "drawdown_penalty_sum": 0.01,
            "drawdown_penalty_mean": 0.001,
            "action_change_penalty_sum": 0.01,
            "action_change_penalty_mean": 0.001,
            "reward_clip_ratio": 0.0,
            "final_equity": 1.01,
            "total_return": 0.01,
            "max_drawdown": 0.02,
            "sharpe": 0.5,
        },
        "output_dir": "logs/diagnostics/fake",
    }
    with patch("src.cli.collect_model_diagnostics", return_value=fake_result) as mock_diag:
        exit_code = cli.main(["diagnose", "--asset", "btc_usdt"])
    assert exit_code == 0
    mock_diag.assert_called_once()


def test_cli_routes_all_asset_diagnose():
    fake_result = {
        "mode": "diagnose_all",
        "summaries": [],
        "results": [],
        "output_dir": "logs/diagnostics/fake_all",
    }
    with patch("src.cli.collect_all_model_diagnostics", return_value=fake_result) as mock_diag:
        exit_code = cli.main(["diagnose", "--all"])
    assert exit_code == 0
    mock_diag.assert_called_once()


def test_diagnose_json_outputs_valid_json():
    result = run_cli("diagnose", "--asset", "btc_usdt", "--format", "json")
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "diagnose"
    assert payload["asset"] == "btc_usdt"
    assert "policy_std_mean" in payload


def test_experiment_help_works():
    result = run_cli("experiment", "--help")
    assert result.returncode == 0
    assert "reward" in result.stdout


def test_experiment_reward_help_works():
    result = run_cli("experiment", "reward", "--help")
    assert result.returncode == 0
    assert "--presets" in result.stdout


def test_experiment_reward_all_fails():
    result = run_cli("experiment", "reward", "--all")
    assert result.returncode != 0
    assert "Reward experiments currently support one asset at a time. Use --asset." in result.stderr


def test_experiment_reward_routes_correctly():
    fake_result = {
        "experiment_dir": "logs/experiments/fake",
        "best_by_score": "current",
        "best_by_sharpe": "current",
        "best_by_wins": "current",
        "summary": []
    }
    with patch("src.experiments.reward_tuning.run_reward_experiment", return_value=fake_result) as mock_exp:
        exit_code = cli.main(["experiment", "reward", "--asset", "btc_usdt", "--quick", "--presets", "current"])
    assert exit_code == 0
    mock_exp.assert_called_once()
    assert mock_exp.call_args[0][0] == "btc_usdt"
    assert mock_exp.call_args[0][2] == ["current"]
    assert mock_exp.call_args.kwargs["quick"] is True

def test_experiment_ppo_std_help_works():
    result = run_cli("experiment", "ppo-std", "--help")
    assert result.returncode == 0
    assert "--presets" in result.stdout

def test_experiment_ppo_std_all_fails():
    result = run_cli("experiment", "ppo-std", "--all")
    assert result.returncode != 0
    assert "PPO std experiments currently support one asset at a time." in result.stderr

def test_experiment_ppo_std_routes_correctly():
    fake_result = {
        "experiment_dir": "logs/experiments/fake",
        "best_by_score": "current",
        "best_by_policy_std_reduction": "current",
        "best_by_flat_ratio_reduction": "current",
        "best_by_walk_forward_sharpe": "current",
        "best_by_baseline_wins": "current",
        "summary": []
    }
    from unittest.mock import patch
    import src.cli as cli
    with patch("src.experiments.ppo_std_tuning.run_ppo_std_experiment", return_value=fake_result) as mock_exp:
        exit_code = cli.main(["experiment", "ppo-std", "--asset", "btc_usdt", "--quick", "--presets", "current", "--disable-early-stopping"])
    assert exit_code == 0
    mock_exp.assert_called_once()
    assert mock_exp.call_args[0][0] == "btc_usdt"
    assert mock_exp.call_args[0][2] == ["current"]
    assert mock_exp.call_args.kwargs["quick"] is True
    assert mock_exp.call_args.kwargs["disable_early_stopping"] is True

def test_experiment_training_signal_routes_correctly():
    fake_result = {
        "experiment_dir": "logs/experiments/fake",
        "summary": {}
    }
    from unittest.mock import patch
    import src.cli as cli
    with patch("src.experiments.training_signal.run_training_signal_experiment", return_value=fake_result) as mock_exp:
        exit_code = cli.main(["experiment", "training-signal", "--asset", "btc_usdt", "--quick"])
    assert exit_code == 0
    mock_exp.assert_called_once()
    assert mock_exp.call_args[0][0] == "btc_usdt"
    assert mock_exp.call_args.kwargs["quick"] is True
