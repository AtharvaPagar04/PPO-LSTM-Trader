import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

from src.config.paths import EXPERIMENTS_DIR, ensure_dir
from src.config.feature_ablation_presets import resolve_feature_ablation_preset
from src.features.pipeline import (
    load_raw_dataframe, engineer_features, add_cross_asset_features,
    engineer_labels, CROSS_ASSET_FEATURE_COLUMNS
)

def build_labeled_dataset(asset, preset_features):
    df = engineer_features(load_raw_dataframe(asset))
    if any(feat in CROSS_ASSET_FEATURE_COLUMNS for feat in preset_features):
        df = add_cross_asset_features(df, asset=asset)
    
    labeled_df = engineer_labels(df)
    return labeled_df


def walk_forward_split(df, folds=5):
    # simple expanding window walk-forward
    n = len(df)
    fold_size = n // (folds + 1)
    splits = []
    for i in range(folds):
        train_end = (i + 1) * fold_size
        test_end = train_end + fold_size
        splits.append((df.iloc[:train_end], df.iloc[train_end:test_end]))
    return splits


def evaluate_horizon(labeled_df, features, horizon):
    target = f"future_return_{horizon}"
    target_bin = f"next_up_{horizon}"
    
    X = labeled_df[features].values
    y_reg = labeled_df[target].values
    y_bin = labeled_df[target_bin].values.astype(int)
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_reg_train, y_reg_test = y_reg[:split_idx], y_reg[split_idx:]
    y_bin_train, y_bin_test = y_bin[:split_idx], y_bin[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Regression
    reg = Ridge()
    reg.fit(X_train_scaled, y_reg_train)
    y_reg_pred = reg.predict(X_test_scaled)
    
    r2 = r2_score(y_reg_test, y_reg_pred)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    dir_acc_reg = accuracy_score(y_bin_test, y_reg_pred > 0)
    
    # Classification
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_scaled, y_bin_train)
    y_bin_pred = clf.predict(X_test_scaled)
    
    acc = accuracy_score(y_bin_test, y_bin_pred)
    bacc = balanced_accuracy_score(y_bin_test, y_bin_pred)
    prec = precision_score(y_bin_test, y_bin_pred, zero_division=0)
    rec = recall_score(y_bin_test, y_bin_pred, zero_division=0)
    
    # Baselines
    acc_always_up = accuracy_score(y_bin_test, np.ones_like(y_bin_test))
    acc_always_down = accuracy_score(y_bin_test, np.zeros_like(y_bin_test))
    acc_random = accuracy_score(y_bin_test, np.random.randint(0, 2, size=len(y_bin_test)))
    
    # Correlation & IC
    corrs = []
    ics = []
    for f in features:
        val = labeled_df[f].values
        # Pearson for basic corr
        corr = np.corrcoef(val, y_reg)[0, 1]
        corrs.append(corr if not np.isnan(corr) else 0.0)
        # Spearman for IC
        ic, _ = spearmanr(val, y_reg)
        ics.append(ic if not np.isnan(ic) else 0.0)
        
    corrs = np.array(corrs)
    ics = np.array(ics)
    
    top_feature_idx = np.argmax(np.abs(corrs))
    
    res = {
        "label_horizon": horizon,
        "target_mean": float(np.mean(y_reg)),
        "target_std": float(np.std(y_reg)),
        "positive_label_ratio": float(np.mean(y_bin)),
        "feature_return_correlation_mean": float(np.mean(corrs)),
        "feature_return_correlation_max_abs": float(np.max(np.abs(corrs))),
        "top_correlated_feature": features[top_feature_idx],
        "information_coefficient_mean": float(np.mean(ics)),
        "information_coefficient_std": float(np.std(ics)),
        "regression_r2": float(r2),
        "regression_mae": float(mae),
        "directional_accuracy_from_regression": float(dir_acc_reg),
        "classification_accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "precision": float(prec),
        "recall": float(rec),
        "baseline_always_up": float(acc_always_up),
        "baseline_always_down": float(acc_always_down),
        "baseline_random": float(acc_random),
    }
    
    # Walk-forward
    splits = walk_forward_split(labeled_df, folds=5)
    wf_accs = []
    wf_baccs = []
    wf_pred_ratios = []
    for train_df, test_df in splits:
        X_tr = scaler.fit_transform(train_df[features].values)
        X_te = scaler.transform(test_df[features].values)
        y_tr = train_df[target_bin].values.astype(int)
        y_te = test_df[target_bin].values.astype(int)
        
        clf_wf = LogisticRegression(max_iter=1000)
        clf_wf.fit(X_tr, y_tr)
        preds = clf_wf.predict(X_te)
        wf_accs.append(accuracy_score(y_te, preds))
        wf_baccs.append(balanced_accuracy_score(y_te, preds))
        wf_pred_ratios.append(np.mean(preds))
        
    res["wf_mean_accuracy"] = float(np.mean(wf_accs))
    res["wf_mean_balanced_accuracy"] = float(np.mean(wf_baccs))
    res["wf_positive_folds"] = int(sum(a > 0.5 for a in wf_accs))
    res["wf_mean_pred_ratio"] = float(np.mean(wf_pred_ratios))
    
    return res


def run_signal_audit_experiment(asset: str, config: dict, presets: list[str], quick: bool = False):
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = EXPERIMENTS_DIR / "signal_audit" / f"{timestamp}_{asset}"
    ensure_dir(run_dir)
    
    all_results = []
    horizons = [1, 3] if quick else [1, 3, 6, 12, 24]
    
    for preset_name in presets:
        preset = resolve_feature_ablation_preset(preset_name)
        features = preset["features"]
        
        labeled_df = build_labeled_dataset(asset, features)
        if quick:
            labeled_df = labeled_df.iloc[-1000:]
            
        for h in horizons:
            res = evaluate_horizon(labeled_df, features, h)
            res["preset"] = preset_name
            res["feature_count"] = len(features)
            
            # Simple directional rules
            if "momentum_10" in features:
                res["momentum_10_acc"] = float(accuracy_score(labeled_df[f"next_up_{h}"], labeled_df["momentum_10"] > 0))
            if "trend" in features:
                res["trend_acc"] = float(accuracy_score(labeled_df[f"next_up_{h}"], labeled_df["trend"] > 0))
            if "rsi" in features:
                res["rsi_low_acc"] = float(np.mean(labeled_df[labeled_df["rsi"] < 30][f"next_up_{h}"]))
                res["rsi_high_acc"] = float(np.mean(~labeled_df[labeled_df["rsi"] > 70][f"next_up_{h}"]))
                
            all_results.append(res)
            
    df_res = pd.DataFrame(all_results)
    df_res.to_csv(run_dir / "summary.csv", index=False)
    
    with (run_dir / "summary.json").open("w") as f:
        json.dump(all_results, f, indent=2)
        
    best_horizon = df_res.loc[df_res["wf_mean_accuracy"].idxmax()]["label_horizon"]
    best_preset = df_res.loc[df_res["wf_mean_accuracy"].idxmax()]["preset"]
    best_acc = df_res["wf_mean_accuracy"].max()
    
    report = f"""# Predictive Signal Audit

## Setup
- Asset: {asset}
- Quick Mode: {quick}

## Feature Presets Tested
{chr(10).join(f"- {p}" for p in presets)}

## Label Horizons
{horizons}

## Supervised Baseline Results
Best Walk-Forward Accuracy: {best_acc:.4f} (Preset: {best_preset}, Horizon: {best_horizon})

"""
    report += "## Interpretation\n"
    if best_acc < 0.51:
        report += "Current features likely do not contain enough directional signal.\nDo not continue PPO tuning yet.\nNext step should be stronger feature research or different target objective.\n"
    else:
        report += "Use that preset/horizon as candidate for RL reward/action redesign.\n"
        
    with (run_dir / "report.md").open("w") as f:
        f.write(report)
        
    manifest = {
        "asset": asset,
        "presets": presets,
        "quick": quick,
        "horizons": horizons,
        "best_preset": str(best_preset),
        "best_horizon": int(best_horizon)
    }
    with (run_dir / "audit_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Signal audit complete. Output saved to {run_dir}")
    return run_dir
