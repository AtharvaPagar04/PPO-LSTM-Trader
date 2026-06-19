from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from src.config.assets import normalize_asset_name
from src.config.feature_ablation_presets import resolve_feature_ablation_preset
from src.config.paths import EXPERIMENTS_DIR, ensure_dir
from src.experiments.signal_audit import build_labeled_dataset, walk_forward_split
from src.utils.logger import get_git_commit, utc_timestamp_slug


DEFAULT_FEATURE_PRESET = "cross_asset_context_v1"
DEFAULT_HORIZONS = [1, 3, 6]
DEFAULT_THRESHOLDS = [0.0, 0.0005, 0.001]
SUPPORTED_HORIZONS = {1, 3, 6, 12, 24}
LABEL_TYPES = (
    "binary_up_threshold",
    "binary_down_threshold",
    "ternary_direction",
    "cost_aware_direction",
)


def estimated_round_trip_cost(transaction_cost: float) -> float:
    return float(transaction_cost) * 2.0


def build_threshold_labels(
    future_returns: np.ndarray | list[float],
    *,
    threshold: float,
    label_type: str,
    round_trip_cost: float = 0.0,
) -> np.ndarray:
    returns = np.asarray(future_returns, dtype=np.float64)
    effective_threshold = float(threshold)
    if label_type == "cost_aware_direction":
        effective_threshold += float(round_trip_cost)

    if label_type == "binary_up_threshold":
        return (returns > threshold).astype(np.int8)
    if label_type == "binary_down_threshold":
        return (returns < -threshold).astype(np.int8)
    if label_type in {"ternary_direction", "cost_aware_direction"}:
        labels = np.zeros(len(returns), dtype=np.int8)
        labels[returns > effective_threshold] = 1
        labels[returns < -effective_threshold] = -1
        return labels
    raise ValueError(f"Unknown label type: {label_type}")


def compute_label_balance(labels: np.ndarray, *, label_type: str) -> dict:
    labels = np.asarray(labels)
    if label_type == "binary_up_threshold":
        positive_ratio = float(np.mean(labels == 1)) if len(labels) else 0.0
        negative_ratio = float(np.mean(labels == 0)) if len(labels) else 0.0
        neutral_ratio = 0.0
        signal_coverage = 1.0
    elif label_type == "binary_down_threshold":
        negative_ratio = float(np.mean(labels == 1)) if len(labels) else 0.0
        positive_ratio = float(np.mean(labels == 0)) if len(labels) else 0.0
        neutral_ratio = 0.0
        signal_coverage = 1.0
    else:
        positive_ratio = float(np.mean(labels == 1)) if len(labels) else 0.0
        negative_ratio = float(np.mean(labels == -1)) if len(labels) else 0.0
        neutral_ratio = float(np.mean(labels == 0)) if len(labels) else 0.0
        signal_coverage = 1.0 - neutral_ratio
    return {
        "positive_ratio": positive_ratio,
        "negative_ratio": negative_ratio,
        "neutral_ratio": neutral_ratio,
        "signal_coverage": signal_coverage,
    }


def compute_target_relevance(
    future_returns: np.ndarray | list[float],
    labels: np.ndarray,
    *,
    label_type: str,
    threshold: float,
    round_trip_cost: float,
) -> dict:
    returns = np.asarray(future_returns, dtype=np.float64)
    labels = np.asarray(labels)
    if label_type == "binary_up_threshold":
        long_mask = labels == 1
        short_mask = labels == 0
        effective_threshold = float(threshold)
    elif label_type == "binary_down_threshold":
        long_mask = labels == 0
        short_mask = labels == 1
        effective_threshold = float(threshold)
    else:
        long_mask = labels == 1
        short_mask = labels == -1
        effective_threshold = float(threshold)
        if label_type == "cost_aware_direction":
            effective_threshold += float(round_trip_cost)
    non_neutral = long_mask | short_mask
    return {
        "mean_future_return_when_long": float(np.mean(returns[long_mask])) if np.any(long_mask) else 0.0,
        "mean_future_return_when_short": float(np.mean(returns[short_mask])) if np.any(short_mask) else 0.0,
        "mean_abs_future_return_for_non_neutral": float(np.mean(np.abs(returns[non_neutral]))) if np.any(non_neutral) else 0.0,
        "estimated_cost_threshold": effective_threshold,
    }


def majority_baseline_accuracy(labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    if len(labels) == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    return float(np.max(counts) / len(labels))


def _safe_logistic_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    unique_labels = np.unique(y_train)
    if len(unique_labels) < 2:
        return np.full(len(X_test), unique_labels[0], dtype=np.int64)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test)


def _ridge_directional_predictions(
    predicted_returns: np.ndarray,
    *,
    threshold: float,
    label_type: str,
    round_trip_cost: float,
) -> np.ndarray:
    if label_type == "binary_up_threshold":
        return (predicted_returns > threshold).astype(np.int8)
    if label_type == "binary_down_threshold":
        return (predicted_returns < -threshold).astype(np.int8)
    effective_threshold = float(threshold)
    if label_type == "cost_aware_direction":
        effective_threshold += float(round_trip_cost)
    labels = np.zeros(len(predicted_returns), dtype=np.int8)
    labels[predicted_returns > effective_threshold] = 1
    labels[predicted_returns < -effective_threshold] = -1
    return labels


def evaluate_target_configuration(
    labeled_df: pd.DataFrame,
    *,
    features: list[str],
    horizon: int,
    threshold: float,
    label_type: str,
    train_split: float,
    round_trip_cost: float,
    folds: int = 5,
) -> dict:
    target_col = f"future_return_{horizon}"
    future_returns = labeled_df[target_col].to_numpy(dtype=np.float64)
    labels = build_threshold_labels(
        future_returns,
        threshold=threshold,
        label_type=label_type,
        round_trip_cost=round_trip_cost,
    )
    split_idx = int(len(labeled_df) * train_split)
    train_df = labeled_df.iloc[:split_idx].copy()
    test_df = labeled_df.iloc[split_idx:].copy()
    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError("Chronological train/test split produced an empty partition.")

    X_train = train_df[features].to_numpy(dtype=np.float64)
    X_test = test_df[features].to_numpy(dtype=np.float64)
    y_train = labels[:split_idx]
    y_test = labels[split_idx:]
    y_train_reg = train_df[target_col].to_numpy(dtype=np.float64)
    y_test_reg = test_df[target_col].to_numpy(dtype=np.float64)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic_predictions = _safe_logistic_predict(X_train_scaled, y_train, X_test_scaled)
    ridge = Ridge()
    ridge.fit(X_train_scaled, y_train_reg)
    ridge_predictions = ridge.predict(X_test_scaled)
    ridge_directional = _ridge_directional_predictions(
        ridge_predictions,
        threshold=threshold,
        label_type=label_type,
        round_trip_cost=round_trip_cost,
    )

    wf_accuracy = []
    wf_balanced_accuracy = []
    wf_positive_folds = 0
    for train_fold, test_fold in walk_forward_split(labeled_df, folds=folds):
        fold_returns = test_fold[target_col].to_numpy(dtype=np.float64)
        fold_train_returns = train_fold[target_col].to_numpy(dtype=np.float64)
        y_fold_train = build_threshold_labels(
            train_fold[target_col].to_numpy(dtype=np.float64),
            threshold=threshold,
            label_type=label_type,
            round_trip_cost=round_trip_cost,
        )
        y_fold_test = build_threshold_labels(
            fold_returns,
            threshold=threshold,
            label_type=label_type,
            round_trip_cost=round_trip_cost,
        )
        fold_scaler = StandardScaler()
        X_fold_train = fold_scaler.fit_transform(train_fold[features].to_numpy(dtype=np.float64))
        X_fold_test = fold_scaler.transform(test_fold[features].to_numpy(dtype=np.float64))
        fold_predictions = _safe_logistic_predict(X_fold_train, y_fold_train, X_fold_test)
        fold_acc = accuracy_score(y_fold_test, fold_predictions)
        fold_bacc = balanced_accuracy_score(y_fold_test, fold_predictions)
        wf_accuracy.append(fold_acc)
        wf_balanced_accuracy.append(fold_bacc)
        if fold_acc > majority_baseline_accuracy(y_fold_test):
            wf_positive_folds += 1

    label_balance = compute_label_balance(labels, label_type=label_type)
    target_relevance = compute_target_relevance(
        future_returns,
        labels,
        label_type=label_type,
        threshold=threshold,
        round_trip_cost=round_trip_cost,
    )
    result = {
        "horizon": horizon,
        "threshold": float(threshold),
        "label_type": label_type,
        **label_balance,
        "majority_baseline_accuracy": majority_baseline_accuracy(y_test),
        "logistic_accuracy": float(accuracy_score(y_test, logistic_predictions)),
        "logistic_balanced_accuracy": float(balanced_accuracy_score(y_test, logistic_predictions)),
        "ridge_directional_accuracy": float(accuracy_score(y_test, ridge_directional)),
        "walk_forward_accuracy": float(np.mean(wf_accuracy)) if wf_accuracy else 0.0,
        "walk_forward_balanced_accuracy": float(np.mean(wf_balanced_accuracy)) if wf_balanced_accuracy else 0.0,
        "positive_folds": int(wf_positive_folds),
        **target_relevance,
    }
    return result


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
    horizons: list[int],
    thresholds: list[float],
    summary_rows: list[dict],
    output_dir: Path,
) -> str:
    best = max(summary_rows, key=lambda row: row["walk_forward_balanced_accuracy"])
    lines = [
        "# Cost-Aware Target / Label Audit",
        "",
        "This is a supervised target diagnostic. It does not prove trading profitability.",
        "",
        "## Setup",
        f"- Asset: {asset}",
        f"- Feature preset: {feature_preset}",
        f"- Horizons: {', '.join(str(h) for h in horizons)}",
        f"- Thresholds: {', '.join(f'{t:.4f}' for t in thresholds)}",
        f"- Output: {output_dir}",
        "",
        "## Label Balance",
    ]
    for row in summary_rows:
        lines.append(
            f"- h={row['horizon']} thr={row['threshold']:.4f} {row['label_type']}: "
            f"coverage={row['signal_coverage']:.1%}, pos={row['positive_ratio']:.1%}, "
            f"neg={row['negative_ratio']:.1%}, neutral={row['neutral_ratio']:.1%}"
        )
    lines.extend(
        [
            "",
            "## Horizon Comparison",
            *[
                f"- h={row['horizon']} thr={row['threshold']:.4f} {row['label_type']}: "
                f"log_acc={row['logistic_accuracy']:.3f}, log_bal_acc={row['logistic_balanced_accuracy']:.3f}, "
                f"wf_bal_acc={row['walk_forward_balanced_accuracy']:.3f}"
                for row in summary_rows
            ],
            "",
            "## Threshold Comparison",
            *[
                f"- h={row['horizon']} thr={row['threshold']:.4f} {row['label_type']}: "
                f"maj={row['majority_baseline_accuracy']:.3f}, ridge_dir={row['ridge_directional_accuracy']:.3f}, "
                f"cost_thr={row['estimated_cost_threshold']:.4f}"
                for row in summary_rows
            ],
            "",
            "## Walk-Forward Results",
            *[
                f"- h={row['horizon']} thr={row['threshold']:.4f} {row['label_type']}: "
                f"wf_acc={row['walk_forward_accuracy']:.3f}, wf_bal_acc={row['walk_forward_balanced_accuracy']:.3f}, "
                f"positive_folds={row['positive_folds']}/5"
                for row in summary_rows
            ],
            "",
            "## Best Candidate Targets",
            f"- Best by walk-forward balanced accuracy: h={best['horizon']} thr={best['threshold']:.4f} {best['label_type']} "
            f"(wf_bal_acc={best['walk_forward_balanced_accuracy']:.3f}, coverage={best['signal_coverage']:.1%})",
            "",
            "## Interpretation",
            "- If thresholding improves balanced accuracy and coverage stays reasonable, that target is a candidate.",
            "- If balanced accuracy rises but coverage collapses, the signal is too sparse to be useful.",
            "- If cost-aware targets beat raw next-up labels, use them in the next supervised strategy pass.",
            "",
            "## Recommendation",
        ]
    )
    raw_best = max(
        (row for row in summary_rows if row["threshold"] == 0.0 and row["label_type"] == "binary_up_threshold"),
        key=lambda row: row["walk_forward_balanced_accuracy"],
        default=None,
    )
    if raw_best and best["walk_forward_balanced_accuracy"] > raw_best["walk_forward_balanced_accuracy"]:
        lines.append(
            "- Cost-aware or thresholded labels outperformed raw next-up labels. Rerun supervised-signal-strategy with the best target."
        )
    else:
        lines.append(
            "- Current features still do not show a clearly stronger cost-aware target. Prioritize stronger feature/data research before PPO changes."
        )
    return "\n".join(lines)


def run_target_audit_experiment(
    asset: str,
    config: dict,
    *,
    feature_preset: str = DEFAULT_FEATURE_PRESET,
    horizons: list[int] | None = None,
    thresholds: list[float] | None = None,
    quick: bool = False,
):
    asset = normalize_asset_name(asset)
    preset = resolve_feature_ablation_preset(feature_preset)
    horizons = list(horizons or DEFAULT_HORIZONS)
    thresholds = [float(value) for value in (thresholds or DEFAULT_THRESHOLDS)]
    unsupported = [horizon for horizon in horizons if horizon not in SUPPORTED_HORIZONS]
    if unsupported:
        raise ValueError(
            "Unsupported horizons: " + ", ".join(str(value) for value in unsupported)
        )

    labeled_df = build_labeled_dataset(asset, preset["features"])
    if quick:
        labeled_df = labeled_df.iloc[-1000:].reset_index(drop=True)
    transaction_cost = float(config["environment"]["transaction_cost"])
    round_trip_cost = estimated_round_trip_cost(transaction_cost)
    train_split = float(config["data"]["train_split"])

    timestamp = utc_timestamp_slug()
    output_dir = EXPERIMENTS_DIR / "target_audit" / f"{timestamp}_{asset}"
    ensure_dir(output_dir)

    summary_rows = []
    for horizon in horizons:
        for threshold in thresholds:
            for label_type in LABEL_TYPES:
                summary_rows.append(
                    {
                        "asset": asset,
                        "feature_preset": feature_preset,
                        **evaluate_target_configuration(
                            labeled_df,
                            features=preset["features"],
                            horizon=horizon,
                            threshold=threshold,
                            label_type=label_type,
                            train_split=train_split,
                            round_trip_cost=round_trip_cost,
                        ),
                    }
                )

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    report = _build_report(
        asset=asset,
        feature_preset=feature_preset,
        horizons=horizons,
        thresholds=thresholds,
        summary_rows=summary_rows,
        output_dir=output_dir,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    best = max(summary_rows, key=lambda row: row["walk_forward_balanced_accuracy"])
    raw_best = max(
        (row for row in summary_rows if row["threshold"] == 0.0 and row["label_type"] == "binary_up_threshold"),
        key=lambda row: row["walk_forward_balanced_accuracy"],
        default=None,
    )
    manifest = {
        "asset": asset,
        "experiment_type": "target-audit",
        "feature_preset": feature_preset,
        "horizons": horizons,
        "thresholds": thresholds,
        "quick": quick,
        "selected_features": preset["features"],
        "python_command": " ".join(sys.argv),
        "git_commit_if_available": get_git_commit(),
        "git_dirty_status_if_available": _git_dirty_status(),
        "transaction_cost": transaction_cost,
        "estimated_round_trip_cost": round_trip_cost,
        "best_candidate": best,
        "raw_next_up_best": raw_best,
    }
    with (output_dir / "audit_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Target audit complete. Output saved to {output_dir}")
    print(
        f"Best candidate: h={best['horizon']} thr={best['threshold']:.4f} {best['label_type']} | "
        f"wf_bal_acc={best['walk_forward_balanced_accuracy']:.3f} | coverage={best['signal_coverage']:.1%}"
    )
    return {
        "experiment_dir": str(output_dir),
        "summary_rows": summary_rows,
        "best_row": best,
        "raw_next_up_best": raw_best,
    }
