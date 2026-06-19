from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config.assets import normalize_asset_name
from src.config.feature_ablation_presets import resolve_feature_ablation_preset
from src.config.paths import EXPERIMENTS_DIR, ensure_dir
from src.experiments.signal_audit import walk_forward_split
from src.features.pipeline import (
    ALL_FEATURE_COLUMNS,
    CROSS_ASSET_FEATURE_COLUMNS,
    add_cross_asset_features,
    engineer_features,
    engineer_labels,
    load_raw_dataframe,
)
from src.utils.logger import get_git_commit, utc_timestamp_slug


DEFAULT_FEATURE_PRESET = "cross_asset_context_v1"
DEFAULT_HORIZONS = [1, 3, 6, 12, 24]
SUPPORTED_HORIZONS = {1, 3, 6, 12, 24}
POSITIVE_SIGNAL_SPEARMAN = 0.03
POSITIVE_SIGNAL_AUC = 0.53
STRONG_SPEARMAN = 0.03
STRONG_AUC = 0.53
STRONG_MI = 0.003
WEAK_SPEARMAN = 0.015
WEAK_AUC = 0.515
WEAK_MI = 0.001


LEAKAGE_AUDIT_TEMPLATES = {
    "log_return": {
        "construction": "log(close_t / close_t-1)",
        "uses_current_close": True,
        "uses_past_rolling_values": False,
        "uses_future_shifted_values": False,
        "uses_cross_asset_aligned_current_timestamp": False,
    },
    "momentum_10": {
        "construction": "close_t.pct_change(10)",
        "uses_current_close": True,
        "uses_past_rolling_values": True,
        "uses_future_shifted_values": False,
        "uses_cross_asset_aligned_current_timestamp": False,
    },
    "trend": {
        "construction": "ma_10_t - ma_30_t",
        "uses_current_close": True,
        "uses_past_rolling_values": True,
        "uses_future_shifted_values": False,
        "uses_cross_asset_aligned_current_timestamp": False,
    },
    "rsi": {
        "construction": "14-period RSI at timestamp t",
        "uses_current_close": True,
        "uses_past_rolling_values": True,
        "uses_future_shifted_values": False,
        "uses_cross_asset_aligned_current_timestamp": False,
    },
    "eth_return_24": {
        "construction": "aligned ETH log(close_t / close_t-24)",
        "uses_current_close": True,
        "uses_past_rolling_values": True,
        "uses_future_shifted_values": False,
        "uses_cross_asset_aligned_current_timestamp": True,
    },
    "sol_return_24": {
        "construction": "aligned SOL log(close_t / close_t-24)",
        "uses_current_close": True,
        "uses_past_rolling_values": True,
        "uses_future_shifted_values": False,
        "uses_cross_asset_aligned_current_timestamp": True,
    },
    "eth_return_72": {
        "construction": "aligned ETH log(close_t / close_t-72)",
        "uses_current_close": True,
        "uses_past_rolling_values": True,
        "uses_future_shifted_values": False,
        "uses_cross_asset_aligned_current_timestamp": True,
    },
    "sol_return_72": {
        "construction": "aligned SOL log(close_t / close_t-72)",
        "uses_current_close": True,
        "uses_past_rolling_values": True,
        "uses_future_shifted_values": False,
        "uses_cross_asset_aligned_current_timestamp": True,
    },
    "eth_btc_return_spread_24": {
        "construction": "btc_return_24_t - eth_return_24_t",
        "uses_current_close": True,
        "uses_past_rolling_values": True,
        "uses_future_shifted_values": False,
        "uses_cross_asset_aligned_current_timestamp": True,
    },
    "sol_btc_return_spread_24": {
        "construction": "btc_return_24_t - sol_return_24_t",
        "uses_current_close": True,
        "uses_past_rolling_values": True,
        "uses_future_shifted_values": False,
        "uses_cross_asset_aligned_current_timestamp": True,
    },
    "market_avg_return_24": {
        "construction": "mean(btc_return_24_t, eth_return_24_t, sol_return_24_t)",
        "uses_current_close": True,
        "uses_past_rolling_values": True,
        "uses_future_shifted_values": False,
        "uses_cross_asset_aligned_current_timestamp": True,
    },
    "btc_relative_strength_24": {
        "construction": "btc_return_24_t - market_avg_return_24_t",
        "uses_current_close": True,
        "uses_past_rolling_values": True,
        "uses_future_shifted_values": False,
        "uses_cross_asset_aligned_current_timestamp": True,
    },
}


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


def safe_corr(method: str, x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    if method == "pearson":
        value = np.corrcoef(x, y)[0, 1]
    elif method == "spearman":
        value = pd.Series(x).corr(pd.Series(y), method="spearman")
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    return 0.0 if np.isnan(value) else float(value)


def safe_mutual_information(feature_values: np.ndarray, labels: np.ndarray) -> float:
    feature_values = np.asarray(feature_values, dtype=np.float64).reshape(-1, 1)
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) < 3 or np.unique(labels).size < 2 or np.std(feature_values[:, 0]) < 1e-12:
        return 0.0
    value = mutual_info_classif(
        feature_values,
        labels,
        discrete_features=False,
        random_state=42,
    )[0]
    return 0.0 if np.isnan(value) else float(value)


def safe_single_feature_auc(
    train_feature: np.ndarray,
    train_labels: np.ndarray,
    test_feature: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[float, float]:
    train_feature = np.asarray(train_feature, dtype=np.float64).reshape(-1, 1)
    test_feature = np.asarray(test_feature, dtype=np.float64).reshape(-1, 1)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    if np.unique(train_labels).size < 2 or np.unique(test_labels).size < 2:
        return 0.5, 0.5
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_feature)
    x_test = scaler.transform(test_feature)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, train_labels)
    train_scores = model.predict_proba(x_train)[:, 1]
    test_scores = model.predict_proba(x_test)[:, 1]
    return float(roc_auc_score(train_labels, train_scores)), float(
        roc_auc_score(test_labels, test_scores)
    )


def shuffle_train_feature(train_feature: np.ndarray, seed: int) -> np.ndarray:
    shuffled = np.asarray(train_feature, dtype=np.float64).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    return shuffled


def compute_signal_strength_bucket(
    *, mean_abs_spearman_test: float, mean_mi_test: float, mean_auc_test: float
) -> str:
    if (
        mean_abs_spearman_test >= STRONG_SPEARMAN
        or mean_auc_test >= STRONG_AUC
        or mean_mi_test >= STRONG_MI
    ):
        return "strong"
    if (
        mean_abs_spearman_test >= WEAK_SPEARMAN
        or mean_auc_test >= WEAK_AUC
        or mean_mi_test >= WEAK_MI
    ):
        return "weak"
    return "no_signal"


def prepare_feature_signal_dataset(asset: str, feature_preset: str) -> dict:
    preset = resolve_feature_ablation_preset(feature_preset)
    configured_features = list(preset["features"])

    raw_df = load_raw_dataframe(asset)
    engineered_df = engineer_features(raw_df)
    rows_after_engineering = len(engineered_df)
    if any(feature in CROSS_ASSET_FEATURE_COLUMNS for feature in configured_features):
        generated_df = add_cross_asset_features(engineered_df, asset=asset)
    else:
        generated_df = engineered_df.copy()

    generated_feature_columns = [
        feature for feature in ALL_FEATURE_COLUMNS if feature in generated_df.columns
    ]
    missing_configured = sorted(set(configured_features) - set(generated_feature_columns))
    if missing_configured:
        raise ValueError(
            "Configured preset features are missing from generated dataset: "
            + ", ".join(missing_configured)
        )
    extra_generated = sorted(set(generated_feature_columns) - set(configured_features))

    labeled_df = engineer_labels(generated_df)
    feature_matrix = labeled_df[configured_features].copy()
    if any(column.startswith("future_return_") or column.startswith("next_up_") for column in feature_matrix.columns):
        raise ValueError("Feature matrix unexpectedly includes future labels.")

    return {
        "preset": preset,
        "configured_features": configured_features,
        "generated_feature_columns": generated_feature_columns,
        "missing_configured_features": missing_configured,
        "extra_generated_features": extra_generated,
        "raw_rows": len(raw_df),
        "engineered_rows": rows_after_engineering,
        "aligned_rows": len(generated_df),
        "labeled_rows": len(labeled_df),
        "rows_dropped_due_to_rolling_cross_asset_alignment": len(raw_df) - len(generated_df),
        "rows_dropped_due_to_future_labels": len(generated_df) - len(labeled_df),
        "timestamp_start": str(labeled_df["timestamp"].min()),
        "timestamp_end": str(labeled_df["timestamp"].max()),
        "x_shape": list(feature_matrix.shape),
        "generated_df": generated_df,
        "labeled_df": labeled_df,
    }


def build_leakage_audit(feature_names: list[str]) -> list[dict]:
    rows = []
    for feature in feature_names:
        template = LEAKAGE_AUDIT_TEMPLATES.get(
            feature,
            {
                "construction": "Feature not explicitly documented in leakage template.",
                "uses_current_close": None,
                "uses_past_rolling_values": None,
                "uses_future_shifted_values": None,
                "uses_cross_asset_aligned_current_timestamp": None,
            },
        )
        rows.append({"feature": feature, **template})
    return rows


def compute_feature_fold_metrics(
    labeled_df: pd.DataFrame,
    *,
    feature_name: str,
    horizon: int,
    folds: int = 5,
) -> list[dict]:
    target_col = f"future_return_{horizon}"
    full_spearman = safe_corr(
        "spearman",
        labeled_df[feature_name].to_numpy(),
        labeled_df[target_col].to_numpy(),
    )
    fold_rows = []
    for fold_index, (train_df, test_df) in enumerate(
        walk_forward_split(labeled_df, folds=folds), start=1
    ):
        x_train = train_df[feature_name].to_numpy(dtype=np.float64)
        x_test = test_df[feature_name].to_numpy(dtype=np.float64)
        y_train_return = train_df[target_col].to_numpy(dtype=np.float64)
        y_test_return = test_df[target_col].to_numpy(dtype=np.float64)
        y_train_up = (y_train_return > 0.0).astype(int)
        y_test_up = (y_test_return > 0.0).astype(int)
        y_train_down = (y_train_return < 0.0).astype(int)
        y_test_down = (y_test_return < 0.0).astype(int)

        auc_up_train, auc_up_test = safe_single_feature_auc(
            x_train, y_train_up, x_test, y_test_up
        )
        auc_down_train, auc_down_test = safe_single_feature_auc(
            x_train, y_train_down, x_test, y_test_down
        )

        shuffled_train = shuffle_train_feature(x_train, seed=42 + fold_index)
        _, shuffled_auc_up_test = safe_single_feature_auc(
            shuffled_train, y_train_up, x_test, y_test_up
        )
        _, shuffled_auc_down_test = safe_single_feature_auc(
            shuffled_train, y_train_down, x_test, y_test_down
        )

        mi_up_train = safe_mutual_information(x_train, y_train_up)
        mi_down_train = safe_mutual_information(x_train, y_train_down)
        mi_up_test = safe_mutual_information(x_test, y_test_up)
        mi_down_test = safe_mutual_information(x_test, y_test_down)
        shuffled_mi_up_train = safe_mutual_information(shuffled_train, y_train_up)
        shuffled_mi_down_train = safe_mutual_information(shuffled_train, y_train_down)

        fold_rows.append(
            {
                "horizon": horizon,
                "feature": feature_name,
                "fold_index": fold_index,
                "train_start": str(train_df["timestamp"].min()),
                "train_end": str(train_df["timestamp"].max()),
                "test_start": str(test_df["timestamp"].min()),
                "test_end": str(test_df["timestamp"].max()),
                "spearman_train": safe_corr("spearman", x_train, y_train_return),
                "spearman_test": safe_corr("spearman", x_test, y_test_return),
                "mi_up_train": mi_up_train,
                "mi_down_train": mi_down_train,
                "mi_train": max(mi_up_train, mi_down_train),
                "mi_up_test": mi_up_test,
                "mi_down_test": mi_down_test,
                "mi_test": max(mi_up_test, mi_down_test),
                "auc_up_train": auc_up_train,
                "auc_up_test": auc_up_test,
                "auc_down_train": auc_down_train,
                "auc_down_test": auc_down_test,
                "auc_train": max(auc_up_train, auc_down_train),
                "auc_test": max(auc_up_test, auc_down_test),
                "shuffled_auc_up_test": shuffled_auc_up_test,
                "shuffled_auc_down_test": shuffled_auc_down_test,
                "shuffled_auc_test": max(shuffled_auc_up_test, shuffled_auc_down_test),
                "shuffled_mi_up_train": shuffled_mi_up_train,
                "shuffled_mi_down_train": shuffled_mi_down_train,
                "shuffled_mi_train": max(shuffled_mi_up_train, shuffled_mi_down_train),
                "full_period_spearman": full_spearman,
            }
        )
    return fold_rows


def aggregate_feature_metrics(
    labeled_df: pd.DataFrame,
    *,
    feature_name: str,
    horizon: int,
    fold_rows: list[dict],
) -> dict:
    target_col = f"future_return_{horizon}"
    full_x = labeled_df[feature_name].to_numpy(dtype=np.float64)
    full_y = labeled_df[target_col].to_numpy(dtype=np.float64)
    full_up = (full_y > 0.0).astype(int)
    full_down = (full_y < 0.0).astype(int)
    test_abs_spearman = np.asarray(
        [abs(row["spearman_test"]) for row in fold_rows], dtype=np.float64
    )
    test_mi = np.asarray([row["mi_test"] for row in fold_rows], dtype=np.float64)
    test_auc = np.asarray([row["auc_test"] for row in fold_rows], dtype=np.float64)
    full_sign = np.sign(safe_corr("spearman", full_x, full_y))

    mean_auc_up_test = float(np.mean([row["auc_up_test"] for row in fold_rows]))
    mean_auc_down_test = float(np.mean([row["auc_down_test"] for row in fold_rows]))
    real_mean_mi = float(np.mean([row["mi_train"] for row in fold_rows]))
    shuffled_mean_mi = float(np.mean([row["shuffled_mi_train"] for row in fold_rows]))
    real_mean_auc = float(np.mean([row["auc_test"] for row in fold_rows]))
    shuffled_mean_auc = float(np.mean([row["shuffled_auc_test"] for row in fold_rows]))

    mean_abs_spearman_test = float(np.mean(test_abs_spearman))
    mean_mi_test = float(np.mean(test_mi))
    mean_auc_test = float(np.mean(test_auc))
    summary = {
        "feature": feature_name,
        "horizon": horizon,
        "pearson_corr_vs_future_return": safe_corr("pearson", full_x, full_y),
        "spearman_corr_vs_future_return": safe_corr("spearman", full_x, full_y),
        "abs_spearman_corr": abs(safe_corr("spearman", full_x, full_y)),
        "mutual_information_vs_future_up": safe_mutual_information(full_x, full_up),
        "mutual_information_vs_future_down": safe_mutual_information(full_x, full_down),
        "auc_single_feature_up": float(
            roc_auc_score(full_up, full_x) if np.unique(full_up).size > 1 else 0.5
        ),
        "auc_single_feature_down": float(
            roc_auc_score(full_down, full_x) if np.unique(full_down).size > 1 else 0.5
        ),
        "mean_abs_spearman_test": mean_abs_spearman_test,
        "std_abs_spearman_test": float(np.std(test_abs_spearman)),
        "mean_mi_test": mean_mi_test,
        "std_mi_test": float(np.std(test_mi)),
        "mean_auc_test": mean_auc_test,
        "std_auc_test": float(np.std(test_auc)),
        "mean_auc_up_test": mean_auc_up_test,
        "mean_auc_down_test": mean_auc_down_test,
        "positive_signal_folds": int(
            sum(
                abs(row["spearman_test"]) >= POSITIVE_SIGNAL_SPEARMAN
                or row["auc_test"] >= POSITIVE_SIGNAL_AUC
                for row in fold_rows
            )
        ),
        "sign_stability": int(
            sum(np.sign(row["spearman_test"]) == full_sign for row in fold_rows)
        ),
        "real_mean_auc": real_mean_auc,
        "shuffled_mean_auc": shuffled_mean_auc,
        "real_minus_shuffled_auc": real_mean_auc - shuffled_mean_auc,
        "real_mean_mi": real_mean_mi,
        "shuffled_mean_mi": shuffled_mean_mi,
        "real_minus_shuffled_mi": real_mean_mi - shuffled_mean_mi,
    }
    summary["signal_bucket"] = compute_signal_strength_bucket(
        mean_abs_spearman_test=summary["mean_abs_spearman_test"],
        mean_mi_test=summary["mean_mi_test"],
        mean_auc_test=summary["mean_auc_test"],
    )
    return summary


def _write_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _build_report(
    *,
    asset: str,
    feature_preset: str,
    horizons: list[int],
    dataset_info: dict,
    leakage_rows: list[dict],
    ranking_rows: list[dict],
    output_dir: Path,
) -> str:
    strong = [row for row in ranking_rows if row["signal_bucket"] == "strong"]
    weak = [row for row in ranking_rows if row["signal_bucket"] == "weak"]
    none = [row for row in ranking_rows if row["signal_bucket"] == "no_signal"]
    top_spearman = sorted(
        ranking_rows, key=lambda row: row["mean_abs_spearman_test"], reverse=True
    )[:5]
    top_mi = sorted(ranking_rows, key=lambda row: row["mean_mi_test"], reverse=True)[:5]
    top_auc = sorted(ranking_rows, key=lambda row: row["mean_auc_test"], reverse=True)[:5]
    strong_names = ", ".join(f"{row['feature']}@h{row['horizon']}" for row in strong) or "none"
    weak_names = ", ".join(f"{row['feature']}@h{row['horizon']}" for row in weak) or "none"
    none_names = ", ".join(f"{row['feature']}@h{row['horizon']}" for row in none[:10]) or "none"
    lines = [
        "# Feature Signal Audit",
        "",
        "This is a validation-only diagnostic. It does not prove trading profitability.",
        "",
        "## Setup",
        f"- Asset: {asset}",
        f"- Feature preset: {feature_preset}",
        f"- Horizons: {', '.join(str(h) for h in horizons)}",
        f"- Output: {output_dir}",
        "",
        "## Actual Feature Set",
        f"- Configured features: {', '.join(dataset_info['configured_features'])}",
        f"- Generated feature columns: {', '.join(dataset_info['generated_feature_columns'])}",
        f"- Missing configured features: {', '.join(dataset_info['missing_configured_features']) or 'none'}",
        f"- Extra generated features: {', '.join(dataset_info['extra_generated_features']) or 'none'}",
        f"- Final X shape: {tuple(dataset_info['x_shape'])}",
        f"- Timestamp range: {dataset_info['timestamp_start']} -> {dataset_info['timestamp_end']}",
        f"- Rows dropped due to rolling/cross-asset alignment: {dataset_info['rows_dropped_due_to_rolling_cross_asset_alignment']}",
        f"- Rows dropped due to future-label shifting: {dataset_info['rows_dropped_due_to_future_labels']}",
        "",
        "## Leakage / Lag Audit",
        '- execution_timing_assumption: "features are known at bar close; action applies after feature timestamp"',
        "- backtest_timing: evaluation uses current window close to next window close, so no same-bar return is consumed if the close-to-next-bar assumption holds.",
    ]
    for row in leakage_rows:
        lines.append(
            f"- {row['feature']}: {row['construction']} | current_close={row['uses_current_close']} | "
            f"past_rolling={row['uses_past_rolling_values']} | future_shifted={row['uses_future_shifted_values']} | "
            f"cross_asset_current_ts={row['uses_cross_asset_aligned_current_timestamp']}"
        )
    lines.extend(
        [
            "",
            "## Full-Sample Feature Ranking",
            *[
                f"- {row['feature']} h={row['horizon']}: spearman={row['spearman_corr_vs_future_return']:.4f}, "
                f"mi_up={row['mutual_information_vs_future_up']:.4f}, mi_down={row['mutual_information_vs_future_down']:.4f}, "
                f"auc_up={row['auc_single_feature_up']:.4f}, auc_down={row['auc_single_feature_down']:.4f}"
                for row in ranking_rows[:10]
            ],
            "",
            "## Walk-Forward Feature Stability",
            *[
                f"- {row['feature']} h={row['horizon']}: mean|spearman|={row['mean_abs_spearman_test']:.4f}, "
                f"mean_auc={row['mean_auc_test']:.4f}, mean_mi={row['mean_mi_test']:.4f}, "
                f"positive_folds={row['positive_signal_folds']}/5, sign_stability={row['sign_stability']}/5"
                for row in ranking_rows[:10]
            ],
            "",
            "## Permutation Sanity Check",
            *[
                f"- {row['feature']} h={row['horizon']}: real_auc={row['real_mean_auc']:.4f}, shuffled_auc={row['shuffled_mean_auc']:.4f}, "
                f"real-shuffled_auc={row['real_minus_shuffled_auc']:.4f}, real_mi={row['real_mean_mi']:.4f}, "
                f"shuffled_mi={row['shuffled_mean_mi']:.4f}, real-shuffled_mi={row['real_minus_shuffled_mi']:.4f}"
                for row in ranking_rows[:10]
            ],
            "",
            "## Strong / Weak / No-Signal Features",
            f"- Strong: {strong_names}",
            f"- Weak: {weak_names}",
            f"- No signal: {none_names}",
            "",
            "## Interpretation",
            "- Signal that appears only in train metrics but not test metrics should be treated as unstable or overfit.",
            "- If shuffled train features match real-train MI or real-test AUC, the apparent feature signal is likely noise.",
            "",
            "## Recommendation",
        ]
    )
    if not strong and len(weak) < 3:
        lines.append("- cross_asset_context_v1 shows no useful signal. Stop model tuning and redesign stronger features/data.")
    elif strong or len(weak) >= 3:
        lines.append("- cross_asset_context_v1 may contain weak signal. Use only the strongest stable candidates for further feature research, not PPO tuning yet.")
    else:
        lines.append("- Signal is inconclusive. Validate the few stable features individually before more model work.")
    lines.extend(
        [
            "",
            "### Top 5 by Test Spearman",
            *[
                f"- {row['feature']} h={row['horizon']}: {row['mean_abs_spearman_test']:.4f}"
                for row in top_spearman
            ],
            "",
            "### Top 5 by Test MI",
            *[
                f"- {row['feature']} h={row['horizon']}: {row['mean_mi_test']:.4f}"
                for row in top_mi
            ],
            "",
            "### Top 5 by Test AUC",
            *[
                f"- {row['feature']} h={row['horizon']}: {row['mean_auc_test']:.4f}"
                for row in top_auc
            ],
        ]
    )
    return "\n".join(lines)


def run_feature_signal_audit_experiment(
    asset: str,
    config: dict,
    *,
    feature_preset: str = DEFAULT_FEATURE_PRESET,
    horizons: list[int] | None = None,
    quick: bool = False,
):
    asset = normalize_asset_name(asset)
    horizons = list(horizons or DEFAULT_HORIZONS)
    unsupported = [h for h in horizons if h not in SUPPORTED_HORIZONS]
    if unsupported:
        raise ValueError(
            "Unsupported horizons: " + ", ".join(str(value) for value in unsupported)
        )

    dataset_info = prepare_feature_signal_dataset(asset, feature_preset)
    leakage_rows = build_leakage_audit(dataset_info["configured_features"])
    labeled_df = dataset_info["labeled_df"]
    if quick:
        labeled_df = labeled_df.iloc[-1000:].reset_index(drop=True)
        dataset_info["timestamp_start"] = str(labeled_df["timestamp"].min())
        dataset_info["timestamp_end"] = str(labeled_df["timestamp"].max())
        dataset_info["x_shape"] = [len(labeled_df), len(dataset_info["configured_features"])]
        dataset_info["labeled_rows"] = len(labeled_df)

    fold_rows: list[dict] = []
    summary_rows: list[dict] = []
    for horizon in horizons:
        for feature_name in dataset_info["configured_features"]:
            feature_fold_rows = compute_feature_fold_metrics(
                labeled_df,
                feature_name=feature_name,
                horizon=horizon,
            )
            fold_rows.extend(feature_fold_rows)
            summary_rows.append(
                {
                    "asset": asset,
                    "feature_preset": feature_preset,
                    **aggregate_feature_metrics(
                        labeled_df,
                        feature_name=feature_name,
                        horizon=horizon,
                        fold_rows=feature_fold_rows,
                    ),
                }
            )

    ranking_rows = sorted(
        summary_rows,
        key=lambda row: (
            row["signal_bucket"] != "strong",
            row["signal_bucket"] != "weak",
            -row["mean_auc_test"],
            -row["mean_abs_spearman_test"],
        ),
    )

    timestamp = utc_timestamp_slug()
    output_dir = EXPERIMENTS_DIR / "feature_signal_audit" / f"{timestamp}_{asset}"
    ensure_dir(output_dir)

    _write_csv(output_dir / "summary.csv", summary_rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
    _write_csv(output_dir / "feature_rankings.csv", ranking_rows)
    _write_csv(output_dir / "fold_level_metrics.csv", fold_rows)

    report = _build_report(
        asset=asset,
        feature_preset=feature_preset,
        horizons=horizons,
        dataset_info=dataset_info,
        leakage_rows=leakage_rows,
        ranking_rows=ranking_rows,
        output_dir=output_dir,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    strong = [
        f"{row['feature']}@h{row['horizon']}"
        for row in ranking_rows
        if row["signal_bucket"] == "strong"
    ]
    weak = [
        f"{row['feature']}@h{row['horizon']}"
        for row in ranking_rows
        if row["signal_bucket"] == "weak"
    ]
    no_signal = [
        f"{row['feature']}@h{row['horizon']}"
        for row in ranking_rows
        if row["signal_bucket"] == "no_signal"
    ]
    verdict = (
        "cross_asset_context_v1 has no useful signal"
        if not strong and len(weak) < 3
        else "cross_asset_context_v1 may contain weak signal"
    )
    manifest = {
        "asset": asset,
        "experiment_type": "feature-signal-audit",
        "feature_preset": feature_preset,
        "horizons": horizons,
        "quick": quick,
        "configured_features": dataset_info["configured_features"],
        "generated_feature_columns": dataset_info["generated_feature_columns"],
        "rows_used": dataset_info["labeled_rows"],
        "timestamp_range": [dataset_info["timestamp_start"], dataset_info["timestamp_end"]],
        "execution_timing_assumption": "features are known at bar close; action applies after feature timestamp",
        "git_commit_if_available": get_git_commit(),
        "git_dirty_status_if_available": _git_dirty_status(),
        "python_command": " ".join(sys.argv),
        "strong_features": strong,
        "weak_features": weak,
        "no_signal_features": no_signal,
        "overall_verdict": verdict,
    }
    with (output_dir / "audit_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Feature signal audit complete. Output saved to {output_dir}")
    print(f"Overall verdict: {verdict}")
    return {
        "experiment_dir": str(output_dir),
        "summary_rows": summary_rows,
        "ranking_rows": ranking_rows,
        "manifest": manifest,
    }
