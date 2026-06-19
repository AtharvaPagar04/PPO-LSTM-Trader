from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

from src.config.assets import normalize_asset_name
from src.config.feature_ablation_presets import resolve_feature_ablation_preset
from src.config.paths import EXPERIMENTS_DIR, ensure_dir
from src.evaluation.baselines import run_baselines, simulate_positions
from src.evaluation.diagnostics import (
    classify_dominant_action_side,
    compute_reporting_threshold_metrics,
)
from src.evaluation.metrics import compute_performance_metrics
from src.experiments.signal_audit import build_labeled_dataset, walk_forward_split
from src.experiments.target_audit import build_threshold_labels, estimated_round_trip_cost
from src.utils.logger import get_git_commit, utc_timestamp_slug


DEFAULT_FEATURE_PRESET = "cross_asset_context_v1"
DEFAULT_HORIZON = 1
DEFAULT_TARGET_LABEL = "binary_up_threshold"
DEFAULT_THRESHOLD = 0.0
SUPPORTED_HORIZONS = {1, 3, 6, 12, 24}
SUPPORTED_TARGET_LABELS = {
    "binary_up_threshold",
    "binary_down_threshold",
}
PRICE_COLUMNS = ["open", "high", "low", "close", "volume"]


def target_direction_multiplier(target_label: str) -> float:
    if target_label == "binary_up_threshold":
        return 1.0
    if target_label == "binary_down_threshold":
        return -1.0
    raise ValueError(
        f"Unsupported target label for supervised signal strategy: {target_label}"
    )


def hard_sign_action(
    probabilities: np.ndarray, *, direction_multiplier: float = 1.0
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    raw_actions = np.where(probabilities > 0.5, 1.0, -1.0)
    return np.clip(direction_multiplier * raw_actions, -1.0, 1.0).astype(np.float32)


def confidence_scaled_action(
    probabilities: np.ndarray, *, direction_multiplier: float = 1.0
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    raw_confidence = 2.0 * (probabilities - 0.5)
    return np.clip(direction_multiplier * raw_confidence, -1.0, 1.0).astype(np.float32)


def confidence_scaled_2x_action(
    probabilities: np.ndarray, *, direction_multiplier: float = 1.0
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    raw_confidence = 4.0 * (probabilities - 0.5)
    return np.clip(direction_multiplier * raw_confidence, -1.0, 1.0).astype(np.float32)


def thresholded_confidence_action(
    probabilities: np.ndarray,
    threshold: float = 0.02,
    *,
    direction_multiplier: float = 1.0,
) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    edge = probabilities - 0.5
    raw_actions = np.where(np.abs(edge) < threshold, 0.0, 2.0 * edge)
    return np.clip(direction_multiplier * raw_actions, -1.0, 1.0).astype(np.float32)


def regression_scaled_action(
    predictions: np.ndarray, train_return_std: float
) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.float64)
    scale = max(float(train_return_std), 1e-8)
    return np.clip(predictions / scale, -1.0, 1.0).astype(np.float32)


def build_price_windows_from_frame(frame: pd.DataFrame) -> np.ndarray:
    return frame[PRICE_COLUMNS].to_numpy(dtype=np.float32)[:, None, :]


def evaluate_strategy_actions(
    *,
    price_windows: np.ndarray,
    actions: np.ndarray,
    transaction_cost: float,
    seed: int,
) -> dict:
    trace = simulate_positions(price_windows, actions, transaction_cost)
    metrics = compute_performance_metrics(trace)
    baselines = run_baselines(
        price_windows,
        transaction_cost=transaction_cost,
        seed=seed,
        reference_actions=actions,
    )
    baseline_metrics = {
        name: compute_performance_metrics(baseline_trace)
        for name, baseline_trace in baselines.items()
    }
    return {
        "trace": trace,
        "metrics": metrics,
        "baselines": baseline_metrics,
    }


def _safe_logistic_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, dict]:
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        constant_probability = float(unique_labels[0]) if len(unique_labels) else 0.5
        probabilities = np.full(len(X_test), constant_probability, dtype=np.float64)
        return probabilities, {
            "classification_accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "positive_prediction_ratio": constant_probability,
            "model_kind": "constant_classifier",
        }

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    probabilities = classifier.predict_proba(X_test)[:, 1]
    predictions = probabilities > 0.5
    return probabilities, {
        "classification_accuracy": None,
        "balanced_accuracy": None,
        "positive_prediction_ratio": float(np.mean(predictions)),
        "model_kind": "logistic_regression",
    }


def _evaluate_baseline_comparison(
    strategy_return: float,
    baseline_metrics: dict[str, dict],
) -> dict:
    return {
        "beat_always_flat": int(strategy_return > baseline_metrics["always_flat"]["total_return"]),
        "beat_always_long": int(strategy_return > baseline_metrics["always_long"]["total_return"]),
        "beat_always_short": int(strategy_return > baseline_metrics["always_short"]["total_return"]),
        "beat_random": int(strategy_return > baseline_metrics["random"]["total_return"]),
        "beat_constant_signed_mean_action": int(
            strategy_return > baseline_metrics["constant_signed_mean_action"]["total_return"]
        ),
        "beat_constant_abs_mean_long": int(
            strategy_return > baseline_metrics["constant_abs_mean_long"]["total_return"]
        ),
        "beat_constant_abs_mean_short": int(
            strategy_return > baseline_metrics["constant_abs_mean_short"]["total_return"]
        ),
    }


def _determine_best_strategy(
    strategy_return: float,
    baseline_metrics: dict[str, dict],
) -> int:
    candidates = [{"name": "strategy", "return": float(strategy_return)}]
    candidates.extend(
        {"name": name, "return": float(metrics["total_return"])}
        for name, metrics in baseline_metrics.items()
    )
    ranked = sorted(candidates, key=lambda item: (-item["return"], item["name"]))
    return int(ranked[0]["name"] == "strategy")


def _action_summary(actions: np.ndarray) -> dict:
    actions = np.asarray(actions, dtype=np.float64)
    positive_ratio = float(np.mean(actions > 0.0)) if len(actions) else 0.0
    negative_ratio = float(np.mean(actions < 0.0)) if len(actions) else 0.0
    action_abs_mean = float(np.mean(np.abs(actions))) if len(actions) else 0.0
    return {
        "action_mean": float(np.mean(actions)) if len(actions) else 0.0,
        "action_abs_mean": action_abs_mean,
        "positive_action_ratio": positive_ratio,
        "negative_action_ratio": negative_ratio,
        "dominant_action_side": classify_dominant_action_side(
            positive_action_ratio=positive_ratio,
            negative_action_ratio=negative_ratio,
            action_abs_mean=action_abs_mean,
        ),
        **compute_reporting_threshold_metrics(actions),
    }


def _make_strategy_row(
    *,
    model_name: str,
    action_mapping: str,
    metrics: dict,
    baseline_metrics: dict[str, dict],
    actions: np.ndarray,
    transaction_cost_sum: float,
    accuracy_metrics: dict,
) -> dict:
    row = {
        "model_name": model_name,
        "action_mapping": action_mapping,
        "return": metrics["total_return"],
        "final_equity": metrics["final_equity"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "turnover": metrics["turnover"],
        "transaction_cost_sum": transaction_cost_sum,
        **_action_summary(actions),
        "always_flat_return": baseline_metrics["always_flat"]["total_return"],
        "always_long_return": baseline_metrics["always_long"]["total_return"],
        "always_short_return": baseline_metrics["always_short"]["total_return"],
        "random_return": baseline_metrics["random"]["total_return"],
        "buy_and_hold_return": baseline_metrics["buy_and_hold"]["total_return"],
        "constant_signed_mean_action_return": baseline_metrics["constant_signed_mean_action"]["total_return"],
        "constant_abs_mean_long_return": baseline_metrics["constant_abs_mean_long"]["total_return"],
        "constant_abs_mean_short_return": baseline_metrics["constant_abs_mean_short"]["total_return"],
        **accuracy_metrics,
    }
    return row


def _evaluate_logistic_mapping(
    probabilities: np.ndarray,
    mapping_name: str,
    *,
    direction_multiplier: float,
) -> np.ndarray:
    if mapping_name == "hard_sign":
        return hard_sign_action(probabilities, direction_multiplier=direction_multiplier)
    if mapping_name == "confidence_scaled":
        return confidence_scaled_action(probabilities, direction_multiplier=direction_multiplier)
    if mapping_name == "confidence_scaled_2x":
        return confidence_scaled_2x_action(probabilities, direction_multiplier=direction_multiplier)
    if mapping_name == "thresholded_confidence":
        return thresholded_confidence_action(
            probabilities, direction_multiplier=direction_multiplier
        )
    raise ValueError(f"Unknown logistic action mapping: {mapping_name}")


def evaluate_supervised_holdout(
    labeled_df: pd.DataFrame,
    *,
    features: list[str],
    horizon: int,
    target_label: str,
    threshold: float,
    train_split: float,
    transaction_cost: float,
    seed: int,
) -> list[dict]:
    target_reg = f"future_return_{horizon}"
    direction_multiplier = target_direction_multiplier(target_label)
    split_idx = int(len(labeled_df) * train_split)
    train_df = labeled_df.iloc[:split_idx].copy()
    test_df = labeled_df.iloc[split_idx:].copy()
    if len(test_df) < 2:
        raise ValueError("Test split is too small for supervised signal strategy evaluation.")

    X_train = train_df[features].to_numpy(dtype=np.float64)
    X_test = test_df[features].to_numpy(dtype=np.float64)
    y_train_reg = train_df[target_reg].to_numpy(dtype=np.float64)
    y_test_reg = test_df[target_reg].to_numpy(dtype=np.float64)
    round_trip_cost = estimated_round_trip_cost(transaction_cost)
    y_train_bin = build_threshold_labels(
        y_train_reg,
        threshold=threshold,
        label_type=target_label,
        round_trip_cost=round_trip_cost,
    ).astype(int)
    y_test_bin = build_threshold_labels(
        y_test_reg,
        threshold=threshold,
        label_type=target_label,
        round_trip_cost=round_trip_cost,
    ).astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    evaluation_slice = slice(0, len(test_df) - 1)
    price_windows = build_price_windows_from_frame(test_df.iloc[: len(test_df)])
    y_eval_bin = y_test_bin[evaluation_slice]
    y_eval_reg = y_test_reg[evaluation_slice]

    probabilities, logistic_meta = _safe_logistic_fit_predict(
        X_train_scaled,
        y_train_bin,
        X_test_scaled[evaluation_slice],
    )
    logistic_rows = []
    logistic_predictions = probabilities > 0.5
    logistic_accuracy = {
        "classification_accuracy": float(accuracy_score(y_eval_bin, logistic_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval_bin, logistic_predictions)),
        "regression_r2": 0.0,
        "directional_accuracy": float(accuracy_score(y_eval_bin, logistic_predictions)),
        "positive_label_ratio": float(np.mean(y_eval_bin)) if len(y_eval_bin) else 0.0,
        "positive_prediction_ratio": float(np.mean(logistic_predictions)) if len(logistic_predictions) else 0.0,
        "model_kind": logistic_meta["model_kind"],
    }
    for mapping_name in (
        "hard_sign",
        "confidence_scaled",
        "confidence_scaled_2x",
        "thresholded_confidence",
    ):
        actions = _evaluate_logistic_mapping(
            probabilities,
            mapping_name,
            direction_multiplier=direction_multiplier,
        )
        evaluation = evaluate_strategy_actions(
            price_windows=price_windows,
            actions=actions,
            transaction_cost=transaction_cost,
            seed=seed,
        )
        logistic_rows.append(
            _make_strategy_row(
                model_name="logistic_regression",
                action_mapping=mapping_name,
                metrics=evaluation["metrics"],
                baseline_metrics=evaluation["baselines"],
                actions=actions,
                transaction_cost_sum=float(np.sum(evaluation["trace"]["transaction_cost"])),
                accuracy_metrics={
                    **logistic_accuracy,
                    "target_label": target_label,
                    "threshold": float(threshold),
                    "direction_multiplier": float(direction_multiplier),
                },
            )
        )

    regressor = Ridge()
    regressor.fit(X_train_scaled, y_train_reg)
    reg_predictions = regressor.predict(X_test_scaled[evaluation_slice])
    reg_actions = regression_scaled_action(reg_predictions, float(np.std(y_train_reg)))
    reg_eval = evaluate_strategy_actions(
        price_windows=price_windows,
        actions=reg_actions,
        transaction_cost=transaction_cost,
        seed=seed,
    )
    reg_directional = reg_predictions > 0.0
    regression_row = _make_strategy_row(
        model_name="ridge_regression",
        action_mapping="regression_scaled",
        metrics=reg_eval["metrics"],
        baseline_metrics=reg_eval["baselines"],
        actions=reg_actions,
        transaction_cost_sum=float(np.sum(reg_eval["trace"]["transaction_cost"])),
        accuracy_metrics={
            "classification_accuracy": float(accuracy_score(y_eval_bin, reg_directional)),
            "balanced_accuracy": float(balanced_accuracy_score(y_eval_bin, reg_directional)),
            "regression_r2": float(r2_score(y_eval_reg, reg_predictions)),
            "directional_accuracy": float(accuracy_score(y_eval_bin, reg_directional)),
            "positive_label_ratio": float(np.mean(y_eval_bin)) if len(y_eval_bin) else 0.0,
            "positive_prediction_ratio": float(np.mean(reg_directional)) if len(reg_directional) else 0.0,
            "model_kind": "ridge_regression",
            "target_label": target_label,
            "threshold": float(threshold),
            "direction_multiplier": float(direction_multiplier),
        },
    )
    return logistic_rows + [regression_row]


def evaluate_supervised_walk_forward(
    labeled_df: pd.DataFrame,
    *,
    features: list[str],
    horizon: int,
    target_label: str,
    threshold: float,
    transaction_cost: float,
    seed: int,
    folds: int = 5,
) -> list[dict]:
    target_reg = f"future_return_{horizon}"
    direction_multiplier = target_direction_multiplier(target_label)
    round_trip_cost = estimated_round_trip_cost(transaction_cost)
    results_by_strategy: dict[tuple[str, str], list[dict]] = {}

    for fold_index, (train_df, test_df) in enumerate(walk_forward_split(labeled_df, folds=folds), start=1):
        if len(test_df) < 2:
            continue
        X_train = train_df[features].to_numpy(dtype=np.float64)
        X_test = test_df[features].to_numpy(dtype=np.float64)
        y_train_reg = train_df[target_reg].to_numpy(dtype=np.float64)
        y_test_reg = test_df[target_reg].to_numpy(dtype=np.float64)
        y_train_bin = build_threshold_labels(
            y_train_reg,
            threshold=threshold,
            label_type=target_label,
            round_trip_cost=round_trip_cost,
        ).astype(int)
        y_test_bin = build_threshold_labels(
            y_test_reg,
            threshold=threshold,
            label_type=target_label,
            round_trip_cost=round_trip_cost,
        ).astype(int)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        evaluation_slice = slice(0, len(test_df) - 1)
        price_windows = build_price_windows_from_frame(test_df.iloc[: len(test_df)])
        y_eval_bin = y_test_bin[evaluation_slice]
        y_eval_reg = y_test_reg[evaluation_slice]

        probabilities, _ = _safe_logistic_fit_predict(
            X_train_scaled,
            y_train_bin,
            X_test_scaled[evaluation_slice],
        )
        for mapping_name in (
            "hard_sign",
            "confidence_scaled",
            "confidence_scaled_2x",
            "thresholded_confidence",
        ):
            actions = _evaluate_logistic_mapping(
                probabilities,
                mapping_name,
                direction_multiplier=direction_multiplier,
            )
            evaluation = evaluate_strategy_actions(
                price_windows=price_windows,
                actions=actions,
                transaction_cost=transaction_cost,
                seed=seed + fold_index,
            )
            comparison = _evaluate_baseline_comparison(
                evaluation["metrics"]["total_return"],
                evaluation["baselines"],
            )
            key = ("logistic_regression", mapping_name)
            results_by_strategy.setdefault(key, []).append(
                {
                    "fold_index": fold_index,
                    "return": evaluation["metrics"]["total_return"],
                    "sharpe": evaluation["metrics"]["sharpe"],
                    "max_drawdown": evaluation["metrics"]["max_drawdown"],
                    "classification_accuracy": float(accuracy_score(y_eval_bin, probabilities > 0.5)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_eval_bin, probabilities > 0.5)),
                    "directional_accuracy": float(accuracy_score(y_eval_bin, probabilities > 0.5)),
                    "positive_prediction_ratio": float(np.mean(probabilities > 0.5)),
                    "transaction_cost_sum": float(np.sum(evaluation["trace"]["transaction_cost"])),
                    "constant_signed_mean_action_return": evaluation["baselines"]["constant_signed_mean_action"]["total_return"],
                    "constant_abs_mean_short_return": evaluation["baselines"]["constant_abs_mean_short"]["total_return"],
                    "constant_abs_mean_long_return": evaluation["baselines"]["constant_abs_mean_long"]["total_return"],
                    "best_fold": _determine_best_strategy(
                        evaluation["metrics"]["total_return"],
                        evaluation["baselines"],
                    ),
                    **comparison,
                }
            )

        regressor = Ridge()
        regressor.fit(X_train_scaled, y_train_reg)
        reg_predictions = regressor.predict(X_test_scaled[evaluation_slice])
        reg_actions = regression_scaled_action(reg_predictions, float(np.std(y_train_reg)))
        reg_eval = evaluate_strategy_actions(
            price_windows=price_windows,
            actions=reg_actions,
            transaction_cost=transaction_cost,
            seed=seed + 100 + fold_index,
        )
        reg_comparison = _evaluate_baseline_comparison(
            reg_eval["metrics"]["total_return"],
            reg_eval["baselines"],
        )
        reg_key = ("ridge_regression", "regression_scaled")
        reg_directional = reg_predictions > 0.0
        results_by_strategy.setdefault(reg_key, []).append(
            {
                "fold_index": fold_index,
                "return": reg_eval["metrics"]["total_return"],
                "sharpe": reg_eval["metrics"]["sharpe"],
                "max_drawdown": reg_eval["metrics"]["max_drawdown"],
                "classification_accuracy": float(accuracy_score(y_eval_bin, reg_directional)),
                "balanced_accuracy": float(balanced_accuracy_score(y_eval_bin, reg_directional)),
                "directional_accuracy": float(accuracy_score(y_eval_bin, reg_directional)),
                "regression_r2": float(r2_score(y_eval_reg, reg_predictions)),
                "positive_prediction_ratio": float(np.mean(reg_directional)),
                "transaction_cost_sum": float(np.sum(reg_eval["trace"]["transaction_cost"])),
                "constant_signed_mean_action_return": reg_eval["baselines"]["constant_signed_mean_action"]["total_return"],
                "constant_abs_mean_short_return": reg_eval["baselines"]["constant_abs_mean_short"]["total_return"],
                "constant_abs_mean_long_return": reg_eval["baselines"]["constant_abs_mean_long"]["total_return"],
                "best_fold": _determine_best_strategy(
                    reg_eval["metrics"]["total_return"],
                    reg_eval["baselines"],
                ),
                **reg_comparison,
            }
        )

    aggregate_rows = []
    for (model_name, action_mapping), fold_rows in results_by_strategy.items():
        returns = np.asarray([row["return"] for row in fold_rows], dtype=np.float64)
        sharpes = np.asarray([row["sharpe"] for row in fold_rows], dtype=np.float64)
        drawdowns = np.asarray([row["max_drawdown"] for row in fold_rows], dtype=np.float64)
        accuracy = np.asarray([row["classification_accuracy"] for row in fold_rows], dtype=np.float64)
        balance = np.asarray([row["balanced_accuracy"] for row in fold_rows], dtype=np.float64)
        directional = np.asarray([row["directional_accuracy"] for row in fold_rows], dtype=np.float64)
        r2_values = np.asarray([row.get("regression_r2", 0.0) for row in fold_rows], dtype=np.float64)
        aggregate_rows.append(
            {
                "model_name": model_name,
                "action_mapping": action_mapping,
                "mean_return": float(np.mean(returns)),
                "mean_sharpe": float(np.mean(sharpes)),
                "positive_folds": int(np.sum(returns > 0.0)),
                "best_fold_count": int(sum(row["best_fold"] for row in fold_rows)),
                "beat_constant_signed_count": int(sum(row["beat_constant_signed_mean_action"] for row in fold_rows)),
                "beat_constant_abs_short_count": int(sum(row["beat_constant_abs_mean_short"] for row in fold_rows)),
                "beat_constant_abs_long_count": int(sum(row["beat_constant_abs_mean_long"] for row in fold_rows)),
                "mean_max_drawdown": float(np.mean(drawdowns)),
                "mean_classification_accuracy": float(np.mean(accuracy)),
                "mean_balanced_accuracy": float(np.mean(balance)),
                "mean_directional_accuracy": float(np.mean(directional)),
                "mean_regression_r2": float(np.mean(r2_values)),
                "constant_signed_mean_action_mean_return": float(
                    np.mean([row["constant_signed_mean_action_return"] for row in fold_rows])
                ),
                "constant_abs_mean_short_mean_return": float(
                    np.mean([row["constant_abs_mean_short_return"] for row in fold_rows])
                ),
                "constant_abs_mean_long_mean_return": float(
                    np.mean([row["constant_abs_mean_long_return"] for row in fold_rows])
                ),
                "fold_rows": fold_rows,
            }
        )
    return aggregate_rows


def _git_dirty_status():
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or ""


def _build_report(
    *,
    asset: str,
    feature_preset: str,
    horizon: int,
    target_label: str,
    threshold: float,
    direction_multiplier: float,
    summary_rows: list[dict],
    experiment_dir: Path,
) -> str:
    sorted_rows = sorted(summary_rows, key=lambda row: row["walk_forward_mean_sharpe"], reverse=True)
    table = [
        "| Model | Mapping | Test Acc | WF Mean Return | WF Mean Sharpe | Pos Folds | Best Folds | Beat Const Mean | Beat Const Short | Beat Const Long | Side | Flat@1 | Flat@5 | Flat@10 | Flat@25 | Avg |Act| |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted_rows:
        table.append(
            f"| {row['model_name']} | {row['action_mapping']} | {row['classification_accuracy']:.3f} | "
            f"{row['walk_forward_mean_return']:.4f} | {row['walk_forward_mean_sharpe']:.3f} | "
            f"{row['walk_forward_positive_folds']}/{row['walk_forward_total_folds']} | {row['best_fold_count']}/{row['walk_forward_total_folds']} | "
            f"{row['beat_constant_signed_count']}/{row['walk_forward_total_folds']} | {row['beat_constant_abs_short_count']}/{row['walk_forward_total_folds']} | "
            f"{row['beat_constant_abs_long_count']}/{row['walk_forward_total_folds']} | {row['dominant_action_side']} | "
            f"{row['flat_ratio_001']:.1%} | {row['flat_ratio_005']:.1%} | {row['flat_ratio_010']:.1%} | {row['flat_ratio_025']:.1%} | {row['action_abs_mean']:.4f} |"
        )

    best_mapping = sorted_rows[0]
    return "\n".join(
        [
            "# Supervised Signal Strategy Baseline",
            "",
            "This is an offline diagnostic baseline. It does not execute trades and does not prove live profitability.",
            "",
            "## Setup",
            f"- Asset: {asset}",
            f"- Feature preset: {feature_preset}",
            f"- Horizon: {horizon}",
            f"- Target label: {target_label}",
            f"- Threshold: {threshold:.4f}",
            f"- Direction multiplier: {direction_multiplier:+.1f}",
            f"- Output: {experiment_dir}",
            "",
            "## Target Label",
            f"- This target predicts {'upside' if direction_multiplier > 0 else 'downside'} events, so high model probability maps to {'long' if direction_multiplier > 0 else 'short'} exposure.",
            "",
            "## Model Accuracy",
            *[
                f"- {row['model_name']} / {row['action_mapping']}: test_acc={row['classification_accuracy']:.3f}, "
                f"wf_acc={row['walk_forward_mean_accuracy']:.3f}, wf_bal_acc={row['walk_forward_mean_balanced_accuracy']:.3f}"
                for row in sorted_rows
            ],
            "",
            "## Action Mapping Results",
            *table,
            "",
            "## Walk-Forward Trading Results",
            *[
                f"- {row['model_name']} / {row['action_mapping']}: mean_return={row['walk_forward_mean_return']:.4f}, "
                f"mean_sharpe={row['walk_forward_mean_sharpe']:.3f}, positive_folds={row['walk_forward_positive_folds']}/{row['walk_forward_total_folds']}"
                for row in sorted_rows
            ],
            "",
            "## Exposure-Equivalent Baseline Comparison",
            *[
                f"- {row['model_name']} / {row['action_mapping']}: "
                f"beat_const_mean={row['beat_constant_signed_count']}/{row['walk_forward_total_folds']}, "
                f"beat_const_short={row['beat_constant_abs_short_count']}/{row['walk_forward_total_folds']}, "
                f"beat_const_long={row['beat_constant_abs_long_count']}/{row['walk_forward_total_folds']}"
                for row in sorted_rows
            ],
            "",
            "## Interpretation",
            "- If a strategy beats always-flat but not same-exposure baselines, the signal is not adding much timing value.",
            "- If confidence-based mappings outperform hard sign, calibrated sizing is more promising than discrete classification.",
            "",
            "## Recommendation",
            f"- Best mapping in this run: {best_mapping['model_name']} / {best_mapping['action_mapping']}.",
            "- Only treat it as PPO-guidance candidate if it beats exposure-equivalent baselines consistently.",
        ]
    )


def run_supervised_signal_strategy_experiment(
    asset: str,
    config: dict,
    *,
    feature_preset: str = DEFAULT_FEATURE_PRESET,
    horizon: int = DEFAULT_HORIZON,
    target_label: str = DEFAULT_TARGET_LABEL,
    threshold: float = DEFAULT_THRESHOLD,
    quick: bool = False,
):
    asset = normalize_asset_name(asset)
    if horizon not in SUPPORTED_HORIZONS:
        supported = ", ".join(str(value) for value in sorted(SUPPORTED_HORIZONS))
        raise ValueError(f"Unsupported horizon: {horizon}. Supported horizons: {supported}")
    if target_label not in SUPPORTED_TARGET_LABELS:
        supported = ", ".join(sorted(SUPPORTED_TARGET_LABELS))
        raise ValueError(
            f"Unsupported target label: {target_label}. Supported target labels: {supported}"
        )

    preset = resolve_feature_ablation_preset(feature_preset)
    features = preset["features"]
    labeled_df = build_labeled_dataset(asset, features)
    if quick:
        labeled_df = labeled_df.iloc[-1000:].reset_index(drop=True)
    if len(labeled_df) < 50:
        raise ValueError("Supervised signal strategy requires at least 50 labeled rows.")

    transaction_cost = float(config["environment"]["transaction_cost"])
    train_split = float(config["data"]["train_split"])
    seed = int(config["evaluation"]["random_seed"])
    direction_multiplier = target_direction_multiplier(target_label)

    timestamp = utc_timestamp_slug()
    experiment_dir = EXPERIMENTS_DIR / "supervised_signal_strategy" / f"{timestamp}_{asset}"
    ensure_dir(experiment_dir)

    holdout_rows = evaluate_supervised_holdout(
        labeled_df,
        features=features,
        horizon=horizon,
        target_label=target_label,
        threshold=threshold,
        train_split=train_split,
        transaction_cost=transaction_cost,
        seed=seed,
    )
    walk_forward_rows = evaluate_supervised_walk_forward(
        labeled_df,
        features=features,
        horizon=horizon,
        target_label=target_label,
        threshold=threshold,
        transaction_cost=transaction_cost,
        seed=seed,
        folds=5,
    )
    wf_lookup = {
        (row["model_name"], row["action_mapping"]): row for row in walk_forward_rows
    }

    summary_rows = []
    for holdout_row in holdout_rows:
        key = (holdout_row["model_name"], holdout_row["action_mapping"])
        wf_row = wf_lookup[key]
        summary_rows.append(
            {
                "asset": asset,
                "feature_preset": feature_preset,
                "horizon": horizon,
                "target_label": target_label,
                "threshold": float(threshold),
                "direction_multiplier": float(direction_multiplier),
                **holdout_row,
                "walk_forward_mean_return": wf_row["mean_return"],
                "walk_forward_mean_sharpe": wf_row["mean_sharpe"],
                "walk_forward_positive_folds": wf_row["positive_folds"],
                "walk_forward_total_folds": len(wf_row["fold_rows"]),
                "best_fold_count": wf_row["best_fold_count"],
                "beat_constant_signed_count": wf_row["beat_constant_signed_count"],
                "beat_constant_abs_short_count": wf_row["beat_constant_abs_short_count"],
                "beat_constant_abs_long_count": wf_row["beat_constant_abs_long_count"],
                "walk_forward_mean_accuracy": wf_row["mean_classification_accuracy"],
                "walk_forward_mean_balanced_accuracy": wf_row["mean_balanced_accuracy"],
                "walk_forward_mean_directional_accuracy": wf_row["mean_directional_accuracy"],
                "walk_forward_mean_regression_r2": wf_row["mean_regression_r2"],
                "constant_signed_mean_action_mean_return": wf_row["constant_signed_mean_action_mean_return"],
                "constant_abs_mean_short_mean_return": wf_row["constant_abs_mean_short_mean_return"],
                "constant_abs_mean_long_mean_return": wf_row["constant_abs_mean_long_mean_return"],
            }
        )

    with (experiment_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (experiment_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    report = _build_report(
        asset=asset,
        feature_preset=feature_preset,
        horizon=horizon,
        target_label=target_label,
        threshold=threshold,
        direction_multiplier=direction_multiplier,
        summary_rows=summary_rows,
        experiment_dir=experiment_dir,
    )
    (experiment_dir / "report.md").write_text(report, encoding="utf-8")

    manifest = {
        "asset": asset,
        "experiment_type": "supervised-signal-strategy",
        "feature_preset": feature_preset,
        "horizon": horizon,
        "target_label": target_label,
        "threshold": float(threshold),
        "direction_multiplier": float(direction_multiplier),
        "quick": quick,
        "selected_features": features,
        "feature_count": len(features),
        "python_command": " ".join(sys.argv),
        "git_commit_if_available": get_git_commit(),
        "git_dirty_status_if_available": _git_dirty_status(),
        "models_tested": ["logistic_regression", "ridge_regression"],
        "action_mappings_tested": [
            "hard_sign",
            "confidence_scaled",
            "confidence_scaled_2x",
            "thresholded_confidence",
            "regression_scaled",
        ],
    }
    with (experiment_dir / "audit_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    best_row = max(summary_rows, key=lambda row: row["walk_forward_mean_sharpe"])
    print(f"Supervised signal strategy complete. Output saved to {experiment_dir}")
    print(
        f"Best mapping: {best_row['model_name']} / {best_row['action_mapping']} | "
        f"wf_sharpe={best_row['walk_forward_mean_sharpe']:.3f} | "
        f"beat_const_mean={best_row['beat_constant_signed_count']}/{best_row['walk_forward_total_folds']}"
    )
    return {
        "experiment_dir": str(experiment_dir),
        "summary_rows": summary_rows,
        "best_row": best_row,
    }
