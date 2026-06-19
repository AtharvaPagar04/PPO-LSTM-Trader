from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config.assets import SUPPORTED_ASSETS, asset_to_symbol, normalize_asset_name
from src.config.paths import DIAGNOSTICS_DIR, ensure_dir, resolve_checkpoint_path
from src.data.dataset import load_metadata, load_processed_data
from src.evaluation.benchmark import build_eval_env, load_policy_from_checkpoint
from src.evaluation.metrics import compute_performance_metrics
from src.features.pipeline import ProcessedDataset, engineer_features, load_raw_dataframe
from src.inference import display_path
from src.utils.logger import utc_timestamp_slug, write_json
from src.utils.seed import set_global_seed


TRACE_COLUMNS = [
    "step",
    "index",
    "timestamp",
    "action",
    "action_std",
    "value_estimate",
    "position",
    "position_change",
    "reward",
    "raw_trading_pnl",
    "gross_pnl",
    "pnl",
    "scaled_pnl_reward",
    "transaction_cost",
    "drawdown",
    "drawdown_penalty_value",
    "position_penalty_value",
    "action_change_penalty_value",
    "unclipped_reward",
    "clipped_reward",
    "was_clipped",
    "equity",
    "log_return",
    "simple_return",
    "pnl_component",
    "transaction_cost_component",
    "drawdown_penalty_component",
    "position_penalty_component",
    "turnover_penalty_component",
    "exposure_penalty_component",
    "directional_reward_component",
    "volatility_exposure_penalty_component",
    "total_reward",
]

DEFAULT_FLAT_THRESHOLD = 0.25
ACTION_THRESHOLD_GRID = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
REPORTING_THRESHOLDS = {
    "001": 0.01,
    "005": 0.05,
    "010": 0.10,
    "025": 0.25,
}
ACTION_HISTOGRAM_BUCKETS = [
    (-1.0, -0.5, "[-1.0, -0.5)"),
    (-0.5, -0.2, "[-0.5, -0.2)"),
    (-0.2, -0.1, "[-0.2, -0.1)"),
    (-0.1, -0.05, "[-0.1, -0.05)"),
    (-0.05, -0.01, "[-0.05, -0.01)"),
    (-0.01, 0.01, "[-0.01, 0.01]"),
    (0.01, 0.05, "(0.01, 0.05]"),
    (0.05, 0.1, "(0.05, 0.1]"),
    (0.1, 0.2, "(0.1, 0.2]"),
    (0.2, 0.5, "(0.2, 0.5]"),
    (0.5, 1.0, "(0.5, 1.0]"),
]


@dataclass
class DiagnosticSummary:
    asset: str
    num_steps: int
    start_timestamp: str
    end_timestamp: str
    checkpoint: str
    source: str
    window_size: int
    action_mean: float
    action_std: float
    action_min: float
    action_max: float
    action_median: float
    average_abs_action: float
    action_abs_mean: float
    action_abs_median: float
    action_abs_p75: float
    action_abs_p90: float
    action_abs_p95: float
    action_abs_p99: float
    positive_action_ratio: float
    negative_action_ratio: float
    near_zero_action_ratio_001: float
    near_zero_action_ratio_005: float
    near_zero_action_ratio_010: float
    dominant_action_side: str
    flat_ratio_001: float
    long_ratio_001: float
    short_ratio_001: float
    flat_ratio_005: float
    long_ratio_005: float
    short_ratio_005: float
    flat_ratio_010: float
    long_ratio_010: float
    short_ratio_010: float
    flat_ratio_025: float
    long_ratio_025: float
    short_ratio_025: float
    long_ratio: float
    short_ratio: float
    flat_ratio: float
    strong_long_ratio: float
    strong_short_ratio: float
    average_position: float
    average_abs_position: float
    position_min: float
    position_max: float
    turnover: float
    mean_position_change: float
    max_position_change: float
    policy_std_mean: float
    policy_std_min: float
    policy_std_max: float
    policy_std_median: float
    value_mean: float
    value_min: float
    value_max: float
    value_std: float
    reward_mean: float
    reward_std: float
    reward_min: float
    reward_max: float
    reward_clip_ratio: float
    pnl_mean: float
    pnl_sum: float
    gross_pnl_sum: float
    transaction_cost_sum: float
    transaction_cost_mean: float
    transaction_cost_drag_ratio: float
    position_penalty_sum: float
    position_penalty_mean: float
    drawdown_penalty_sum: float
    drawdown_penalty_mean: float
    action_change_penalty_sum: float
    action_change_penalty_mean: float
    final_equity: float
    total_return: float
    max_drawdown: float
    sharpe: float
    mean_pnl_component: float
    mean_transaction_cost_component: float
    mean_drawdown_penalty_component: float
    mean_position_penalty_component: float
    mean_turnover_penalty_component: float
    mean_exposure_penalty_component: float
    mean_directional_reward_component: float
    mean_volatility_exposure_penalty_component: float
    mean_total_reward: float


def compute_action_bucket_ratios(
    actions: np.ndarray | list[float], threshold: float
) -> dict:
    actions = np.asarray(actions, dtype=np.float64)
    nonflat_mask = np.abs(actions) > threshold
    flat_mask = ~nonflat_mask
    return {
        "threshold": float(threshold),
        "flat_ratio": float(np.mean(flat_mask)) if len(actions) else 0.0,
        "long_ratio": float(np.mean(actions > threshold)) if len(actions) else 0.0,
        "short_ratio": float(np.mean(actions < -threshold)) if len(actions) else 0.0,
        "avg_abs_action_nonflat": float(np.mean(np.abs(actions[nonflat_mask])))
        if np.any(nonflat_mask)
        else 0.0,
        "num_nonflat_steps": int(np.sum(nonflat_mask)),
    }


def compute_threshold_sensitivity(
    actions: np.ndarray | list[float],
    thresholds: list[float] | None = None,
) -> list[dict]:
    thresholds = thresholds or ACTION_THRESHOLD_GRID
    return [compute_action_bucket_ratios(actions, threshold) for threshold in thresholds]


def classify_dominant_action_side(
    *,
    positive_action_ratio: float,
    negative_action_ratio: float,
    action_abs_mean: float,
) -> str:
    if action_abs_mean < 0.01:
        return "near_zero"
    if positive_action_ratio >= 0.8:
        return "mostly_long"
    if negative_action_ratio >= 0.8:
        return "mostly_short"
    return "mixed"


def compute_reporting_threshold_metrics(actions: np.ndarray | list[float]) -> dict:
    rows = {
        suffix: compute_action_bucket_ratios(actions, threshold)
        for suffix, threshold in REPORTING_THRESHOLDS.items()
    }
    return {
        "flat_ratio_001": rows["001"]["flat_ratio"],
        "long_ratio_001": rows["001"]["long_ratio"],
        "short_ratio_001": rows["001"]["short_ratio"],
        "flat_ratio_005": rows["005"]["flat_ratio"],
        "long_ratio_005": rows["005"]["long_ratio"],
        "short_ratio_005": rows["005"]["short_ratio"],
        "flat_ratio_010": rows["010"]["flat_ratio"],
        "long_ratio_010": rows["010"]["long_ratio"],
        "short_ratio_010": rows["010"]["short_ratio"],
        "flat_ratio_025": rows["025"]["flat_ratio"],
        "long_ratio_025": rows["025"]["long_ratio"],
        "short_ratio_025": rows["025"]["short_ratio"],
    }


def compute_action_distribution(actions: np.ndarray | list[float]) -> dict:
    actions = np.asarray(actions, dtype=np.float64)
    abs_actions = np.abs(actions)
    histogram = []
    for lower, upper, label in ACTION_HISTOGRAM_BUCKETS:
        if label == "[-0.01, 0.01]":
            mask = (actions >= lower) & (actions <= upper)
        elif label.startswith("["):
            mask = (actions >= lower) & (actions < upper)
        else:
            mask = (actions > lower) & (actions <= upper)
        histogram.append({"bucket": label, "count": int(np.sum(mask))})
    return {
        "action_mean": float(np.mean(actions)),
        "action_std": float(np.std(actions)),
        "action_min": float(np.min(actions)),
        "action_max": float(np.max(actions)),
        "action_abs_mean": float(np.mean(abs_actions)),
        "action_abs_median": float(np.median(abs_actions)),
        "action_abs_p75": float(np.quantile(abs_actions, 0.75)),
        "action_abs_p90": float(np.quantile(abs_actions, 0.90)),
        "action_abs_p95": float(np.quantile(abs_actions, 0.95)),
        "action_abs_p99": float(np.quantile(abs_actions, 0.99)),
        "positive_action_ratio": float(np.mean(actions > 0.0)),
        "negative_action_ratio": float(np.mean(actions < 0.0)),
        "near_zero_action_ratio_001": float(np.mean(abs_actions <= 0.01)),
        "near_zero_action_ratio_005": float(np.mean(abs_actions <= 0.05)),
        "near_zero_action_ratio_010": float(np.mean(abs_actions <= 0.10)),
        "histogram_buckets": histogram,
    }


def compute_directional_signal_diagnostics(
    trace_df: pd.DataFrame, action_threshold: float = 0.01
) -> dict:
    actions = trace_df["action"].to_numpy(dtype=np.float64)
    next_returns = trace_df["simple_return"].to_numpy(dtype=np.float64)
    pnl = trace_df["pnl"].to_numpy(dtype=np.float64)
    abs_actions = np.abs(actions)
    nonzero_mask = abs_actions > action_threshold
    if np.any(nonzero_mask):
        sign_accuracy = float(
            np.mean(np.sign(actions[nonzero_mask]) == np.sign(next_returns[nonzero_mask]))
        )
    else:
        sign_accuracy = 0.0

    def _mean_where(mask, values):
        return float(np.mean(values[mask])) if np.any(mask) else 0.0

    def _safe_corr(a, b):
        if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    pnl_by_bucket = []
    for lower, upper, label in ACTION_HISTOGRAM_BUCKETS:
        if label == "[-0.01, 0.01]":
            mask = (actions >= lower) & (actions <= upper)
        elif label.startswith("["):
            mask = (actions >= lower) & (actions < upper)
        else:
            mask = (actions > lower) & (actions <= upper)
        pnl_by_bucket.append(
            {
                "bucket": label,
                "num_steps": int(np.sum(mask)),
                "pnl_sum": float(np.sum(pnl[mask])) if np.any(mask) else 0.0,
                "mean_next_return": float(np.mean(next_returns[mask])) if np.any(mask) else 0.0,
            }
        )

    return {
        "sign_accuracy_nonzero": sign_accuracy,
        "mean_next_return_when_long": _mean_where(actions > action_threshold, next_returns),
        "mean_next_return_when_short": _mean_where(actions < -action_threshold, next_returns),
        "mean_action_when_next_return_positive": _mean_where(next_returns > 0.0, actions),
        "mean_action_when_next_return_negative": _mean_where(next_returns < 0.0, actions),
        "action_next_return_correlation": _safe_corr(actions, next_returns),
        "abs_action_next_abs_return_correlation": _safe_corr(
            np.abs(actions), np.abs(next_returns)
        ),
        "pnl_by_action_bucket": pnl_by_bucket,
    }


def _window_end_timestamps(
    asset: str, metadata: dict, expected_rows: int | None = None
) -> list[pd.Timestamp]:
    raw_df = load_raw_dataframe(asset)
    feature_df = engineer_features(raw_df)
    window_size = int(metadata["window_size"])
    timestamps = [
        pd.Timestamp(feature_df["timestamp"].iloc[idx + window_size - 1])
        for idx in range(len(feature_df) - window_size + 1)
    ]
    if expected_rows is not None:
        if len(timestamps) >= expected_rows:
            return timestamps[-expected_rows:]
        raw_window_timestamps = [
            pd.Timestamp(raw_df["timestamp"].iloc[idx + window_size - 1])
            for idx in range(len(raw_df) - window_size + 1)
        ]
        if len(raw_window_timestamps) >= expected_rows:
            return raw_window_timestamps[-expected_rows:]
    split_ratio = float(metadata["split_ratio"])
    split_idx = int(len(timestamps) * split_ratio)
    return timestamps[split_idx:]


def run_diagnostic_trace(
    asset: str,
    *,
    config: dict,
    checkpoint: str | None = None,
    processed_dataset: ProcessedDataset | None = None,
) -> tuple[dict, pd.DataFrame]:
    asset = normalize_asset_name(asset)
    set_global_seed(config["evaluation"]["random_seed"])
    if processed_dataset is not None:
        test_x = processed_dataset.test_windows
        test_price = processed_dataset.test_price_windows
        metadata = processed_dataset.metadata
    else:
        test_x, test_price = load_processed_data(asset, "test")
        metadata = load_metadata(asset)
    timestamps = _window_end_timestamps(asset, metadata, expected_rows=len(test_x))
    checkpoint_path = resolve_checkpoint_path(asset, checkpoint)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy_from_checkpoint(
        asset=asset,
        checkpoint_path=checkpoint_path,
        input_dim=test_x.shape[2],
        config=config,
        device=device,
    )
    model.eval()

    env = build_eval_env(test_x, test_price, config)
    state = env.reset(mode="eval")
    rows: list[dict] = []
    done = False
    step = 0

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, std, value = model(state_t)
        action = float(np.clip(mean.cpu().numpy()[0][0], -1.0, 1.0))
        action_std = float(std.cpu().numpy()[0][0])
        value_estimate = float(value.cpu().numpy()[0][0])

        next_state, reward, done, info = env.step(action)
        if info:
            rows.append(
                {
                    "step": step,
                    "index": int(info["index"]),
                    "timestamp": pd.Timestamp(timestamps[step]).isoformat(sep=" "),
                    "action": action,
                    "action_std": action_std,
                    "value_estimate": value_estimate,
                    "position": float(info["position"]),
                    "position_change": float(info["position_change"]),
                    "reward": float(reward),
                    "raw_trading_pnl": float(info["raw_trading_pnl"]),
                    "gross_pnl": float(info["gross_pnl"]),
                    "pnl": float(info["pnl"]),
                    "scaled_pnl_reward": float(info["scaled_pnl_reward"]),
                    "transaction_cost": float(info["transaction_cost"]),
                    "drawdown": float(info["drawdown"]),
                    "drawdown_penalty_value": float(info["drawdown_penalty_value"]),
                    "position_penalty_value": float(info["position_penalty_value"]),
                    "action_change_penalty_value": float(info["action_change_penalty_value"]),
                    "unclipped_reward": float(info["unclipped_reward"]),
                    "clipped_reward": float(info["clipped_reward"]),
                    "was_clipped": bool(info["was_clipped"]),
                    "equity": float(info["equity"]),
                    "log_return": float(info["log_return"]),
                    "simple_return": float(info["simple_return"]),
                    "pnl_component": float(info.get("pnl_component", 0.0)),
                    "transaction_cost_component": float(info.get("transaction_cost_component", 0.0)),
                    "drawdown_penalty_component": float(info.get("drawdown_penalty_component", 0.0)),
                    "position_penalty_component": float(info.get("position_penalty_component", 0.0)),
                    "turnover_penalty_component": float(info.get("turnover_penalty_component", 0.0)),
                    "exposure_penalty_component": float(info.get("exposure_penalty_component", 0.0)),
                    "directional_reward_component": float(info.get("directional_reward_component", 0.0)),
                    "volatility_exposure_penalty_component": float(info.get("volatility_exposure_penalty_component", 0.0)),
                    "total_reward": float(info.get("total_reward", float(reward))),
                }
            )
        state = next_state
        step += 1

    trace_df = pd.DataFrame(rows, columns=TRACE_COLUMNS)
    summary = summarize_diagnostic_trace(
        asset=asset,
        trace_df=trace_df,
        metadata=metadata,
        checkpoint_path=checkpoint_path,
    )
    return summary, trace_df


def summarize_diagnostic_trace(
    *,
    asset: str,
    trace_df: pd.DataFrame,
    metadata: dict,
    checkpoint_path: str | Path,
) -> dict:
    if trace_df.empty:
        raise ValueError("Diagnostic trace is empty.")

    actions = trace_df["action"].to_numpy(dtype=np.float64)
    positions = trace_df["position"].to_numpy(dtype=np.float64)
    position_change = trace_df["position_change"].to_numpy(dtype=np.float64)
    policy_std = trace_df["action_std"].to_numpy(dtype=np.float64)
    values = trace_df["value_estimate"].to_numpy(dtype=np.float64)
    rewards = trace_df["reward"].to_numpy(dtype=np.float64)
    pnl = trace_df["pnl"].to_numpy(dtype=np.float64)
    gross_pnl = trace_df["gross_pnl"].to_numpy(dtype=np.float64)
    transaction_cost = trace_df["transaction_cost"].to_numpy(dtype=np.float64)
    position_penalty = trace_df["position_penalty_value"].to_numpy(dtype=np.float64)
    drawdown_penalty = trace_df["drawdown_penalty_value"].to_numpy(dtype=np.float64)
    action_change_penalty = trace_df["action_change_penalty_value"].to_numpy(dtype=np.float64)
    clipped = trace_df["was_clipped"].astype(bool).to_numpy()
    distribution = compute_action_distribution(actions)
    threshold_metrics = compute_reporting_threshold_metrics(actions)
    dominant_action_side = classify_dominant_action_side(
        positive_action_ratio=distribution["positive_action_ratio"],
        negative_action_ratio=distribution["negative_action_ratio"],
        action_abs_mean=distribution["action_abs_mean"],
    )

    perf = compute_performance_metrics(
        {
            "equity": np.concatenate([[1.0], trace_df["equity"].to_numpy(dtype=np.float64)]),
            "pnl": pnl,
            "position": positions,
            "transaction_cost": transaction_cost,
            "drawdown": trace_df["drawdown"].to_numpy(dtype=np.float64),
        }
    )

    gross_abs = float(np.sum(np.abs(gross_pnl)))
    transaction_cost_drag_ratio = (
        float(np.sum(transaction_cost) / gross_abs) if gross_abs > 0 else 0.0
    )

    summary = DiagnosticSummary(
        asset=asset,
        num_steps=int(len(trace_df)),
        start_timestamp=str(trace_df["timestamp"].iloc[0]),
        end_timestamp=str(trace_df["timestamp"].iloc[-1]),
        checkpoint=display_path(checkpoint_path),
        source="processed_test_data",
        window_size=int(metadata["window_size"]),
        action_mean=float(np.mean(actions)),
        action_std=float(np.std(actions)),
        action_min=float(np.min(actions)),
        action_max=float(np.max(actions)),
        action_median=float(np.median(actions)),
        average_abs_action=float(np.mean(np.abs(actions))),
        action_abs_mean=distribution["action_abs_mean"],
        action_abs_median=distribution["action_abs_median"],
        action_abs_p75=distribution["action_abs_p75"],
        action_abs_p90=distribution["action_abs_p90"],
        action_abs_p95=distribution["action_abs_p95"],
        action_abs_p99=distribution["action_abs_p99"],
        positive_action_ratio=distribution["positive_action_ratio"],
        negative_action_ratio=distribution["negative_action_ratio"],
        near_zero_action_ratio_001=distribution["near_zero_action_ratio_001"],
        near_zero_action_ratio_005=distribution["near_zero_action_ratio_005"],
        near_zero_action_ratio_010=distribution["near_zero_action_ratio_010"],
        dominant_action_side=dominant_action_side,
        flat_ratio_001=threshold_metrics["flat_ratio_001"],
        long_ratio_001=threshold_metrics["long_ratio_001"],
        short_ratio_001=threshold_metrics["short_ratio_001"],
        flat_ratio_005=threshold_metrics["flat_ratio_005"],
        long_ratio_005=threshold_metrics["long_ratio_005"],
        short_ratio_005=threshold_metrics["short_ratio_005"],
        flat_ratio_010=threshold_metrics["flat_ratio_010"],
        long_ratio_010=threshold_metrics["long_ratio_010"],
        short_ratio_010=threshold_metrics["short_ratio_010"],
        flat_ratio_025=threshold_metrics["flat_ratio_025"],
        long_ratio_025=threshold_metrics["long_ratio_025"],
        short_ratio_025=threshold_metrics["short_ratio_025"],
        long_ratio=threshold_metrics["long_ratio_025"],
        short_ratio=threshold_metrics["short_ratio_025"],
        flat_ratio=threshold_metrics["flat_ratio_025"],
        strong_long_ratio=float(np.mean(actions >= 0.75)),
        strong_short_ratio=float(np.mean(actions <= -0.75)),
        average_position=float(np.mean(positions)),
        average_abs_position=float(np.mean(np.abs(positions))),
        position_min=float(np.min(positions)),
        position_max=float(np.max(positions)),
        turnover=float(np.sum(position_change)),
        mean_position_change=float(np.mean(position_change)),
        max_position_change=float(np.max(position_change)),
        policy_std_mean=float(np.mean(policy_std)),
        policy_std_min=float(np.min(policy_std)),
        policy_std_max=float(np.max(policy_std)),
        policy_std_median=float(np.median(policy_std)),
        value_mean=float(np.mean(values)),
        value_min=float(np.min(values)),
        value_max=float(np.max(values)),
        value_std=float(np.std(values)),
        reward_mean=float(np.mean(rewards)),
        reward_std=float(np.std(rewards)),
        reward_min=float(np.min(rewards)),
        reward_max=float(np.max(rewards)),
        reward_clip_ratio=float(np.mean(clipped)),
        pnl_mean=float(np.mean(pnl)),
        pnl_sum=float(np.sum(pnl)),
        gross_pnl_sum=float(np.sum(gross_pnl)),
        transaction_cost_sum=float(np.sum(transaction_cost)),
        transaction_cost_mean=float(np.mean(transaction_cost)),
        transaction_cost_drag_ratio=transaction_cost_drag_ratio,
        position_penalty_sum=float(np.sum(position_penalty)),
        position_penalty_mean=float(np.mean(position_penalty)),
        drawdown_penalty_sum=float(np.sum(drawdown_penalty)),
        drawdown_penalty_mean=float(np.mean(drawdown_penalty)),
        action_change_penalty_sum=float(np.sum(action_change_penalty)),
        action_change_penalty_mean=float(np.mean(action_change_penalty)),
        final_equity=float(perf["final_equity"]),
        total_return=float(perf["total_return"]),
        max_drawdown=float(perf["max_drawdown"]),
        sharpe=float(perf["sharpe"]),
        mean_pnl_component=float(np.mean(trace_df["pnl_component"])) if "pnl_component" in trace_df else 0.0,
        mean_transaction_cost_component=float(np.mean(trace_df["transaction_cost_component"])) if "transaction_cost_component" in trace_df else 0.0,
        mean_drawdown_penalty_component=float(np.mean(trace_df["drawdown_penalty_component"])) if "drawdown_penalty_component" in trace_df else 0.0,
        mean_position_penalty_component=float(np.mean(trace_df["position_penalty_component"])) if "position_penalty_component" in trace_df else 0.0,
        mean_turnover_penalty_component=float(np.mean(trace_df["turnover_penalty_component"])) if "turnover_penalty_component" in trace_df else 0.0,
        mean_exposure_penalty_component=float(np.mean(trace_df["exposure_penalty_component"])) if "exposure_penalty_component" in trace_df else 0.0,
        mean_directional_reward_component=float(np.mean(trace_df["directional_reward_component"])) if "directional_reward_component" in trace_df else 0.0,
        mean_volatility_exposure_penalty_component=float(np.mean(trace_df["volatility_exposure_penalty_component"])) if "volatility_exposure_penalty_component" in trace_df else 0.0,
        mean_total_reward=float(np.mean(trace_df["total_reward"])) if "total_reward" in trace_df else float(np.mean(rewards)),
    )
    payload = asdict(summary)
    payload["binance_symbol"] = asset_to_symbol(asset)
    payload["feature_names"] = metadata.get("features", [])
    payload["flat_threshold_default"] = DEFAULT_FLAT_THRESHOLD
    return payload


def _format_interpretation(summary: dict) -> str:
    notes: list[str] = []
    if summary["flat_ratio_010"] >= 0.7:
        notes.append("The policy is low-exposure at the 10% threshold.")
    if summary["flat_ratio_025"] >= 0.7:
        notes.append("The policy appears flat only under the legacy 25% reporting threshold.")
    elif summary["average_abs_action"] >= 0.5:
        notes.append("The policy is taking relatively large directional exposure.")
    notes.append(f"Dominant action side: {summary['dominant_action_side']}.")

    if summary["policy_std_mean"] >= 0.8:
        notes.append("Policy std is near the configured upper bound.")

    penalty_sum = (
        summary["position_penalty_sum"]
        + summary["drawdown_penalty_sum"]
        + summary["action_change_penalty_sum"]
    )
    if penalty_sum > abs(summary["pnl_sum"]):
        notes.append("Reward penalties are larger than net PnL and should be inspected before tuning.")

    if summary["transaction_cost_drag_ratio"] >= 0.25:
        notes.append("Transaction-cost drag is materially large relative to gross PnL.")

    if not notes:
        notes.append("The policy behavior looks internally consistent, but reward decomposition should still be reviewed before tuning.")
    return " ".join(notes)


def format_diagnostics_text(summary: dict) -> str:
    return "\n".join(
        [
            f"Model diagnostics: {summary['asset']}",
            "",
            f"Steps: {summary['num_steps']}",
            f"Period: {summary['start_timestamp']} -> {summary['end_timestamp']}",
            f"Checkpoint: {summary['checkpoint']}",
            "",
            "Action behavior:",
            f"  Mean action: {summary['action_mean']:.4f}",
            f"  Action std: {summary['action_std']:.4f}",
            f"  Min / Max action: {summary['action_min']:.4f} / {summary['action_max']:.4f}",
            f"  Avg abs action: {summary['average_abs_action']:.4f}",
            f"  Side: {summary['dominant_action_side']}",
            f"  Flat@1% / 5% / 10% / 25%: {summary['flat_ratio_001']:.1%} / {summary['flat_ratio_005']:.1%} / {summary['flat_ratio_010']:.1%} / {summary['flat_ratio_025']:.1%}",
            f"  Long@10% / Short@10%: {summary['long_ratio_010']:.1%} / {summary['short_ratio_010']:.1%}",
            "",
            "Policy uncertainty:",
            f"  Policy std mean: {summary['policy_std_mean']:.4f}",
            f"  Policy std min/max: {summary['policy_std_min']:.4f} / {summary['policy_std_max']:.4f}",
            "",
            "Reward decomposition:",
            f"  PnL sum: {summary['pnl_sum']:.4f}",
            f"  Transaction cost sum: {summary['transaction_cost_sum']:.4f}",
            f"  Position penalty sum: {summary['position_penalty_sum']:.4f}",
            f"  Drawdown penalty sum: {summary['drawdown_penalty_sum']:.4f}",
            f"  Action-change penalty sum: {summary['action_change_penalty_sum']:.4f}",
            f"  Reward clip ratio: {summary['reward_clip_ratio']:.2%}",
            "",
            "Performance:",
            f"  Final equity: {summary['final_equity']:.4f}",
            f"  Total return: {summary['total_return']:.2%}",
            f"  Sharpe: {summary['sharpe']:.2f}",
            f"  Max drawdown: {summary['max_drawdown']:.2%}",
            "",
            "Interpretation:",
            f"  {_format_interpretation(summary)}",
            "",
            "Note: Diagnostics are offline model behavior measurements. They do not execute trades and do not imply live profitability.",
        ]
    )


def format_diagnostics_table(all_summaries: list[dict]) -> str:
    headers = [
        "Asset",
        "Steps",
        "Act Mean",
        "Avg |Act|",
        "Flat %",
        "Flat@1%",
        "Flat@5%",
        "Flat@10%",
        "Flat@25%",
        "Side",
        "Long %",
        "Short %",
        "PolStd",
        "PnL Sum",
        "Penalty Sum",
        "Return",
        "Sharpe",
    ]
    rows = []
    for summary in all_summaries:
        penalty_sum = (
            summary["position_penalty_sum"]
            + summary["drawdown_penalty_sum"]
            + summary["action_change_penalty_sum"]
        )
        rows.append(
            [
                summary["asset"],
                str(summary["num_steps"]),
                f"{summary['action_mean']:.4f}",
                f"{summary['average_abs_action']:.4f}",
                f"{summary['flat_ratio_001'] * 100:.1f}",
                f"{summary['flat_ratio_005'] * 100:.1f}",
                f"{summary['flat_ratio_010'] * 100:.1f}",
                f"{summary['flat_ratio_025'] * 100:.1f}",
                summary["dominant_action_side"],
                f"{summary['long_ratio_025'] * 100:.1f}",
                f"{summary['short_ratio_025'] * 100:.1f}",
                f"{summary['policy_std_mean']:.4f}",
                f"{summary['pnl_sum']:.4f}",
                f"{penalty_sum:.4f}",
                f"{summary['total_return'] * 100:.2f}%",
                f"{summary['sharpe']:.2f}",
            ]
        )
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    lines = ["  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))]
    lines.extend(
        "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))
        for row in rows
    )
    lines.append("")
    lines.append(
        "Note: Diagnostics are offline model behavior measurements. They do not execute trades and do not imply live profitability."
    )
    return "\n".join(lines)


def format_threshold_sensitivity_table(rows: list[dict]) -> str:
    headers = ["Threshold", "Flat %", "Long %", "Short %", "Avg |Nonflat|", "Nonflat Steps"]
    body = []
    for row in rows:
        body.append(
            [
                f"{row['threshold']:.3f}",
                f"{row['flat_ratio'] * 100:.1f}",
                f"{row['long_ratio'] * 100:.1f}",
                f"{row['short_ratio'] * 100:.1f}",
                f"{row['avg_abs_action_nonflat']:.4f}",
                str(row["num_nonflat_steps"]),
            ]
        )
    widths = [len(header) for header in headers]
    for row in body:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))
    lines = ["  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))]
    lines.extend(
        "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))
        for row in body
    )
    return "\n".join(lines)


def build_diagnostics_report(
    summary: dict,
    threshold_sensitivity: list[dict],
    action_distribution: dict,
    directional_signal: dict,
) -> str:
    histogram_lines = [
        f"- {bucket['bucket']}: {bucket['count']}"
        for bucket in action_distribution["histogram_buckets"]
    ]
    directional_lines = [
        f"- Sign accuracy on |action| > 0.01: {directional_signal['sign_accuracy_nonzero']:.3f}",
        f"- Mean next return when long: {directional_signal['mean_next_return_when_long']:.5f}",
        f"- Mean next return when short: {directional_signal['mean_next_return_when_short']:.5f}",
        f"- Action vs next return correlation: {directional_signal['action_next_return_correlation']:.4f}",
        f"- |Action| vs |next return| correlation: {directional_signal['abs_action_next_abs_return_correlation']:.4f}",
    ]
    return "\n".join(
        [
            f"# Model Diagnostics: {summary['asset']}",
            "",
            "## Action Distribution",
            f"- Mean: {summary['action_mean']:.4f}",
            f"- Avg abs action: {summary['average_abs_action']:.4f}",
            f"- Dominant side: {summary['dominant_action_side']}",
            f"- Abs action p90/p95/p99: {summary['action_abs_p90']:.4f} / {summary['action_abs_p95']:.4f} / {summary['action_abs_p99']:.4f}",
            f"- Near zero ratio <= 0.01: {summary['near_zero_action_ratio_001']:.1%}",
            f"- Near zero ratio <= 0.05: {summary['near_zero_action_ratio_005']:.1%}",
            f"- Near zero ratio <= 0.10: {summary['near_zero_action_ratio_010']:.1%}",
            "",
            "### Histogram",
            *histogram_lines,
            "",
            "## Threshold Sensitivity",
            "```text",
            format_threshold_sensitivity_table(threshold_sensitivity),
            "```",
            "",
            "## Directional Signal",
            *directional_lines,
            "",
            "## Interpretation",
            "## Continuous Action Interpretation",
            f"- Default legacy flat ratio currently uses threshold {summary.get('flat_threshold_default', DEFAULT_FLAT_THRESHOLD):.3f}.",
            f"- Flat@1% / 5% / 10% / 25%: {summary['flat_ratio_001']:.1%} / {summary['flat_ratio_005']:.1%} / {summary['flat_ratio_010']:.1%} / {summary['flat_ratio_025']:.1%}.",
            f"- {_format_interpretation(summary)}",
            "",
            "Note: Diagnostics are offline model behavior measurements. They do not execute trades and do not imply live profitability.",
        ]
    )


def save_diagnostics(
    summary: dict,
    trace_df: pd.DataFrame,
    threshold_sensitivity: list[dict],
    action_distribution: dict,
    directional_signal: dict,
    output_dir: Path,
) -> dict:
    ensure_dir(output_dir)
    write_json(output_dir / "summary.json", summary)
    trace_df.loc[:, ["timestamp", "action", "action_std", "value_estimate", "position"]].to_csv(
        output_dir / "actions.csv",
        index=False,
    )
    trace_df.to_csv(output_dir / "reward_components.csv", index=False)
    pd.DataFrame(threshold_sensitivity).to_csv(
        output_dir / "action_threshold_sensitivity.csv",
        index=False,
    )
    write_json(
        output_dir / "action_threshold_sensitivity.json",
        {"thresholds": threshold_sensitivity},
    )
    write_json(output_dir / "action_distribution.json", action_distribution)
    write_json(output_dir / "directional_signal.json", directional_signal)
    (output_dir / "diagnostics.txt").write_text(
        format_diagnostics_text(summary),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        build_diagnostics_report(
            summary, threshold_sensitivity, action_distribution, directional_signal
        ),
        encoding="utf-8",
    )
    return {
        "summary_path": str(output_dir / "summary.json"),
        "actions_path": str(output_dir / "actions.csv"),
        "reward_components_path": str(output_dir / "reward_components.csv"),
        "threshold_sensitivity_csv_path": str(
            output_dir / "action_threshold_sensitivity.csv"
        ),
        "threshold_sensitivity_json_path": str(
            output_dir / "action_threshold_sensitivity.json"
        ),
        "action_distribution_path": str(output_dir / "action_distribution.json"),
        "directional_signal_path": str(output_dir / "directional_signal.json"),
        "diagnostics_path": str(output_dir / "diagnostics.txt"),
        "report_path": str(output_dir / "report.md"),
    }


def collect_model_diagnostics(
    asset: str,
    *,
    config: dict,
    checkpoint: str | None = None,
    save: bool = True,
    output_dir: str | Path | None = None,
    processed_dataset: ProcessedDataset | None = None,
) -> dict:
    summary, trace_df = run_diagnostic_trace(
        asset,
        config=config,
        checkpoint=checkpoint,
        processed_dataset=processed_dataset,
    )
    threshold_sensitivity = compute_threshold_sensitivity(trace_df["action"].to_numpy())
    action_distribution = compute_action_distribution(trace_df["action"].to_numpy())
    directional_signal = compute_directional_signal_diagnostics(trace_df)
    if save:
        target_dir = (
            Path(output_dir)
            if output_dir
            else DIAGNOSTICS_DIR / f"{utc_timestamp_slug()}_{normalize_asset_name(asset)}"
        )
        paths = save_diagnostics(
            summary,
            trace_df,
            threshold_sensitivity,
            action_distribution,
            directional_signal,
            target_dir,
        )
    else:
        target_dir = None
        paths = {}
    return {
        "mode": "diagnose",
        "asset": normalize_asset_name(asset),
        "summary": summary,
        "trace_df": trace_df,
        "threshold_sensitivity": threshold_sensitivity,
        "action_distribution": action_distribution,
        "directional_signal": directional_signal,
        "output_dir": str(target_dir) if target_dir else None,
        "saved_files": paths,
    }


def collect_all_model_diagnostics(
    *,
    config: dict,
    save: bool = True,
    output_dir: str | Path | None = None,
) -> dict:
    root_dir = (
        Path(output_dir) if output_dir else DIAGNOSTICS_DIR / f"{utc_timestamp_slug()}_all"
    )
    summaries: list[dict] = []
    results: list[dict] = []

    if save:
        ensure_dir(root_dir)

    for asset in SUPPORTED_ASSETS:
        asset_dir = root_dir / asset if save else None
        result = collect_model_diagnostics(
            asset,
            config=config,
            save=save,
            output_dir=asset_dir,
        )
        summaries.append(result["summary"])
        results.append(
            {
                "asset": asset,
                "summary": result["summary"],
                "output_dir": result["output_dir"],
            }
        )

    if save:
        summary_csv = root_dir / "all_assets_summary.csv"
        with summary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            for summary in summaries:
                writer.writerow(summary)
        write_json(root_dir / "all_assets_summary.json", {"results": results})

    return {
        "mode": "diagnose_all",
        "summaries": summaries,
        "results": results,
        "output_dir": str(root_dir) if save else None,
    }
