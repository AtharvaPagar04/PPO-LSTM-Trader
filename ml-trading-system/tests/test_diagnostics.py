import numpy as np
import pandas as pd
import pytest

from src.evaluation.diagnostics import format_diagnostics_table, summarize_diagnostic_trace


def make_trace_df():
    return pd.DataFrame(
        [
            {
                "step": 0,
                "index": 1,
                "timestamp": "2026-01-01 00:00:00",
                "action": -0.8,
                "action_std": 0.8187,
                "value_estimate": -0.5,
                "position": -0.8,
                "position_change": 0.8,
                "reward": -0.2,
                "raw_trading_pnl": -0.01,
                "gross_pnl": -0.009,
                "pnl": -0.01,
                "scaled_pnl_reward": -0.5,
                "transaction_cost": 0.001,
                "drawdown": 0.01,
                "drawdown_penalty_value": 0.001,
                "position_penalty_value": 0.02,
                "action_change_penalty_value": 0.001,
                "unclipped_reward": -0.522,
                "clipped_reward": -0.2,
                "was_clipped": False,
                "equity": 0.99,
                "log_return": 0.01,
                "simple_return": 0.0101,
            },
            {
                "step": 1,
                "index": 2,
                "timestamp": "2026-01-01 01:00:00",
                "action": 0.0,
                "action_std": 0.8187,
                "value_estimate": -0.2,
                "position": 0.0,
                "position_change": 0.8,
                "reward": 0.0,
                "raw_trading_pnl": -0.001,
                "gross_pnl": 0.0,
                "pnl": -0.001,
                "scaled_pnl_reward": -0.05,
                "transaction_cost": 0.001,
                "drawdown": 0.01,
                "drawdown_penalty_value": 0.001,
                "position_penalty_value": 0.0,
                "action_change_penalty_value": 0.001,
                "unclipped_reward": -0.052,
                "clipped_reward": 0.0,
                "was_clipped": False,
                "equity": 0.989,
                "log_return": -0.01,
                "simple_return": -0.0099,
            },
            {
                "step": 2,
                "index": 3,
                "timestamp": "2026-01-01 02:00:00",
                "action": 0.9,
                "action_std": 0.8187,
                "value_estimate": 0.3,
                "position": 0.9,
                "position_change": 0.9,
                "reward": 0.4,
                "raw_trading_pnl": 0.02,
                "gross_pnl": 0.021,
                "pnl": 0.02,
                "scaled_pnl_reward": 1.0,
                "transaction_cost": 0.001,
                "drawdown": 0.0,
                "drawdown_penalty_value": 0.0,
                "position_penalty_value": 0.03,
                "action_change_penalty_value": 0.001,
                "unclipped_reward": 0.969,
                "clipped_reward": 0.4,
                "was_clipped": True,
                "equity": 1.009,
                "log_return": 0.022,
                "simple_return": 0.0222,
            },
        ]
    )


def test_diagnostics_summary_contains_required_keys():
    summary = summarize_diagnostic_trace(
        asset="btc_usdt",
        trace_df=make_trace_df(),
        metadata={"window_size": 20, "features": ["a", "b"]},
        checkpoint_path="models/btc_usdt_model.pth",
    )
    required = {
        "asset",
        "num_steps",
        "action_mean",
        "average_abs_action",
        "flat_ratio",
        "policy_std_mean",
        "pnl_sum",
        "transaction_cost_sum",
        "position_penalty_sum",
        "drawdown_penalty_sum",
        "action_change_penalty_sum",
        "reward_clip_ratio",
        "final_equity",
        "sharpe",
        "max_drawdown",
    }
    assert required <= set(summary)


def test_action_ratios_sum_to_one_and_non_negative_fields_hold():
    summary = summarize_diagnostic_trace(
        asset="btc_usdt",
        trace_df=make_trace_df(),
        metadata={"window_size": 20, "features": ["a"]},
        checkpoint_path="models/btc_usdt_model.pth",
    )
    ratio_sum = summary["long_ratio"] + summary["short_ratio"] + summary["flat_ratio"]
    assert ratio_sum == pytest.approx(1.0)
    assert summary["average_abs_action"] >= 0.0
    assert summary["turnover"] >= 0.0
    assert 0.0 <= summary["reward_clip_ratio"] <= 1.0


def test_policy_std_and_penalty_fields_are_present():
    summary = summarize_diagnostic_trace(
        asset="btc_usdt",
        trace_df=make_trace_df(),
        metadata={"window_size": 20, "features": ["a"]},
        checkpoint_path="models/btc_usdt_model.pth",
    )
    assert summary["policy_std_mean"] == pytest.approx(0.8187)
    assert summary["policy_std_min"] == pytest.approx(0.8187)
    assert summary["policy_std_max"] == pytest.approx(0.8187)
    assert summary["position_penalty_sum"] >= 0.0
    assert summary["drawdown_penalty_sum"] >= 0.0
    assert summary["action_change_penalty_sum"] >= 0.0


def test_diagnostics_table_formats_multiple_assets():
    summaries = [
        summarize_diagnostic_trace(
            asset="btc_usdt",
            trace_df=make_trace_df(),
            metadata={"window_size": 20, "features": ["a"]},
            checkpoint_path="models/btc_usdt_model.pth",
        ),
        summarize_diagnostic_trace(
            asset="eth_usdt",
            trace_df=make_trace_df(),
            metadata={"window_size": 20, "features": ["a"]},
            checkpoint_path="models/eth_usdt_model.pth",
        ),
    ]
    table = format_diagnostics_table(summaries)
    assert "Asset" in table
    assert "btc_usdt" in table
    assert "eth_usdt" in table
