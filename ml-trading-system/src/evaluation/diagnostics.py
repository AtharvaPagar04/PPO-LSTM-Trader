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
from src.features.pipeline import engineer_features, load_raw_dataframe
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


def _window_end_timestamps(asset: str, metadata: dict) -> list[pd.Timestamp]:
    raw_df = load_raw_dataframe(asset)
    feature_df = engineer_features(raw_df)
    window_size = int(metadata["window_size"])
    timestamps = [
        pd.Timestamp(feature_df["timestamp"].iloc[idx + window_size - 1])
        for idx in range(len(feature_df) - window_size + 1)
    ]
    split_ratio = float(metadata["split_ratio"])
    split_idx = int(len(timestamps) * split_ratio)
    return timestamps[split_idx:]


def run_diagnostic_trace(
    asset: str,
    *,
    config: dict,
    checkpoint: str | None = None,
) -> tuple[dict, pd.DataFrame]:
    asset = normalize_asset_name(asset)
    set_global_seed(config["evaluation"]["random_seed"])
    test_x, test_price = load_processed_data(asset, "test")
    metadata = load_metadata(asset)
    timestamps = _window_end_timestamps(asset, metadata)
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
        long_ratio=float(np.mean(actions > 0.25)),
        short_ratio=float(np.mean(actions < -0.25)),
        flat_ratio=float(np.mean((actions >= -0.25) & (actions <= 0.25))),
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
    )
    payload = asdict(summary)
    payload["binance_symbol"] = asset_to_symbol(asset)
    payload["feature_names"] = metadata.get("features", [])
    return payload


def _format_interpretation(summary: dict) -> str:
    notes: list[str] = []
    if summary["flat_ratio"] >= 0.7:
        notes.append("The policy is mostly flat/neutral.")
    elif summary["average_abs_action"] >= 0.5:
        notes.append("The policy is taking relatively large directional exposure.")

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
            f"  Long ratio: {summary['long_ratio']:.1%}",
            f"  Short ratio: {summary['short_ratio']:.1%}",
            f"  Flat ratio: {summary['flat_ratio']:.1%}",
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
                f"{summary['flat_ratio'] * 100:.1f}",
                f"{summary['long_ratio'] * 100:.1f}",
                f"{summary['short_ratio'] * 100:.1f}",
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


def save_diagnostics(
    summary: dict,
    trace_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    ensure_dir(output_dir)
    write_json(output_dir / "summary.json", summary)
    trace_df.loc[:, ["timestamp", "action", "action_std", "value_estimate", "position"]].to_csv(
        output_dir / "actions.csv",
        index=False,
    )
    trace_df.to_csv(output_dir / "reward_components.csv", index=False)
    (output_dir / "diagnostics.txt").write_text(
        format_diagnostics_text(summary),
        encoding="utf-8",
    )
    return {
        "summary_path": str(output_dir / "summary.json"),
        "actions_path": str(output_dir / "actions.csv"),
        "reward_components_path": str(output_dir / "reward_components.csv"),
        "diagnostics_path": str(output_dir / "diagnostics.txt"),
    }


def collect_model_diagnostics(
    asset: str,
    *,
    config: dict,
    checkpoint: str | None = None,
    save: bool = True,
    output_dir: str | Path | None = None,
) -> dict:
    summary, trace_df = run_diagnostic_trace(asset, config=config, checkpoint=checkpoint)
    if save:
        target_dir = (
            Path(output_dir)
            if output_dir
            else DIAGNOSTICS_DIR / f"{utc_timestamp_slug()}_{normalize_asset_name(asset)}"
        )
        paths = save_diagnostics(summary, trace_df, target_dir)
    else:
        target_dir = None
        paths = {}
    return {
        "mode": "diagnose",
        "asset": normalize_asset_name(asset),
        "summary": summary,
        "trace_df": trace_df,
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
