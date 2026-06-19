import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.assets import SUPPORTED_ASSETS, normalize_asset_name
from src.config.paths import resolve_checkpoint_path
from src.config.settings import load_config
from src.evaluate import (
    evaluate_assets,
    print_result,
    print_walk_forward_all,
    print_walk_forward_all_baselines,
    print_walk_forward_asset,
    print_walk_forward_asset_baselines,
)
from src.evaluation.diagnostics import (
    collect_all_model_diagnostics,
    collect_model_diagnostics,
    format_diagnostics_table,
    format_diagnostics_text,
)
from src.evaluation.walk_forward import (
    evaluate_walk_forward_all,
    evaluate_walk_forward_asset,
)
from src.inference import (
    format_prediction_json,
    format_prediction_many_json,
    format_prediction_table,
    format_prediction_text,
    predict_many,
    predict_latest,
    save_prediction,
    save_predictions,
)
from src.train import train_asset


def build_parser():
    parser = argparse.ArgumentParser(description="CLI for the ML trading research model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train one or all supported assets.")
    train_parser.add_argument("--asset", type=str, default=None)
    train_parser.add_argument("--all", action="store_true")
    train_parser.add_argument("--config", type=str, default=None)
    train_parser.add_argument("--checkpoint", type=str, default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate one or all supported assets.")
    eval_parser.add_argument("--asset", type=str, default=None)
    eval_parser.add_argument("--all", action="store_true")
    eval_parser.add_argument("--config", type=str, default=None)
    eval_parser.add_argument("--checkpoint", type=str, default=None)
    eval_parser.add_argument("--walk-forward", action="store_true")
    eval_parser.add_argument("--baselines", action="store_true")
    eval_parser.add_argument("--folds", type=int, default=5)
    eval_parser.add_argument("--fold-size", type=int, default=None)
    eval_parser.add_argument("--output-dir", type=str, default=None)

    diagnose_parser = subparsers.add_parser(
        "diagnose", help="Inspect deterministic 1h model behavior and reward decomposition."
    )
    diagnose_parser.add_argument("--asset", type=str, default=None)
    diagnose_parser.add_argument("--all", action="store_true")
    diagnose_parser.add_argument("--config", type=str, default=None)
    diagnose_parser.add_argument("--checkpoint", type=str, default=None)
    diagnose_parser.add_argument("--walk-forward", action="store_true")
    diagnose_parser.add_argument("--save", action="store_true")
    diagnose_parser.add_argument("--format", choices=["text", "json"], default="text")
    diagnose_parser.add_argument("--output-dir", type=str, default=None)

    predict_parser = subparsers.add_parser("predict", help="Run deterministic model inference.")
    predict_parser.add_argument("--asset", type=str, default=None)
    predict_parser.add_argument("--assets", nargs="+", default=None)
    predict_parser.add_argument("--all", action="store_true")
    predict_parser.add_argument("--config", type=str, default=None)
    predict_parser.add_argument("--checkpoint", type=str, default=None)
    predict_parser.add_argument("--csv", type=str, default=None)
    predict_parser.add_argument("--save", action="store_true")
    predict_parser.add_argument("--format", choices=["text", "json"], default="text")

    experiment_parser = subparsers.add_parser("experiment", help="Run model experiments.")
    experiment_subparsers = experiment_parser.add_subparsers(dest="experiment_type", required=True)
    
    reward_parser = experiment_subparsers.add_parser("reward", help="Run reward tuning experiments.")
    reward_parser.add_argument("--asset", type=str, default=None)
    reward_parser.add_argument("--all", action="store_true")
    reward_parser.add_argument("--config", type=str, default=None)
    reward_parser.add_argument("--presets", nargs="+", default=None)
    reward_parser.add_argument("--all-presets", action="store_true")
    reward_parser.add_argument("--quick", action="store_true")

    ppo_std_parser = experiment_subparsers.add_parser("ppo-std", help="Run PPO std tuning experiments.")
    ppo_std_parser.add_argument("--asset", type=str, default=None)
    ppo_std_parser.add_argument("--all", action="store_true")
    ppo_std_parser.add_argument("--config", type=str, default=None)
    ppo_std_parser.add_argument("--presets", nargs="+", default=None)
    ppo_std_parser.add_argument("--all-presets", action="store_true")
    ppo_std_parser.add_argument("--quick", action="store_true")
    ppo_std_parser.add_argument("--disable-early-stopping", action="store_true")

    ts_parser = experiment_subparsers.add_parser("training-signal")
    ts_parser.add_argument("--asset", required=True)
    ts_parser.add_argument("--config", default=None)
    ts_parser.add_argument("--quick", action="store_true")
    ts_parser.add_argument("--disable-early-stopping", action="store_true")

    feature_ablation_parser = experiment_subparsers.add_parser(
        "feature-ablation", help="Run feature ablation experiments."
    )
    feature_ablation_parser.add_argument("--asset", type=str, default=None)
    feature_ablation_parser.add_argument("--all", action="store_true")
    feature_ablation_parser.add_argument("--config", type=str, default=None)
    feature_ablation_parser.add_argument("--presets", nargs="+", default=None)
    feature_ablation_parser.add_argument("--all-presets", action="store_true")
    feature_ablation_parser.add_argument("--quick", action="store_true")

    seed_validation_parser = experiment_subparsers.add_parser(
        "seed-validation", help="Run repeated seed validation across feature presets."
    )
    seed_validation_parser.add_argument("--asset", type=str, default=None)
    seed_validation_parser.add_argument("--all", action="store_true")
    seed_validation_parser.add_argument("--config", type=str, default=None)
    seed_validation_parser.add_argument("--feature-presets", nargs="+", default=None)
    seed_validation_parser.add_argument("--seeds", nargs="+", type=int, default=None)
    seed_validation_parser.add_argument("--quick", action="store_true")

    action_mapping_parser = experiment_subparsers.add_parser(
        "action-mapping",
        help="Audit flat thresholds and evaluation-only action scaling.",
    )
    action_mapping_parser.add_argument("--asset", type=str, default=None)
    action_mapping_parser.add_argument("--all", action="store_true")
    action_mapping_parser.add_argument("--config", type=str, default=None)
    action_mapping_parser.add_argument("--feature-preset", type=str, default="price_action_minimal")
    action_mapping_parser.add_argument("--scales", nargs="+", type=float, default=None)
    action_mapping_parser.add_argument("--quick", action="store_true")

    objective_calibration_parser = experiment_subparsers.add_parser(
        "objective-calibration", help="Run objective/action calibration experiments."
    )
    objective_calibration_parser.add_argument("--asset", type=str, default=None)
    objective_calibration_parser.add_argument("--config", type=str, default=None)
    objective_calibration_parser.add_argument("--presets", nargs="+", default=None)
    objective_calibration_parser.add_argument("--feature-preset", type=str, default="price_action_minimal")
    objective_calibration_parser.add_argument("--quick", action="store_true")

    return parser


def _load_config(path):
    return load_config(*([path] if path else []))


def _resolve_assets(asset, all_assets):
    if all_assets:
        return list(SUPPORTED_ASSETS)
    if not asset:
        raise ValueError("You must provide --asset unless --all is used.")
    return [normalize_asset_name(asset)]


def _resolve_prediction_assets(asset, assets, all_assets, csv_path):
    selected = int(bool(asset)) + int(bool(assets)) + int(bool(all_assets))
    if selected == 0:
        raise ValueError("Provide one of --asset, --assets, or --all.")
    if selected > 1:
        raise ValueError("Use only one of --asset, --assets, or --all.")
    if csv_path and (all_assets or assets):
        raise ValueError("--csv is only supported with --asset.")
    if all_assets:
        return list(SUPPORTED_ASSETS)
    if assets:
        return [normalize_asset_name(item) for item in assets]
    return [normalize_asset_name(asset)]


def cmd_train(args):
    config = _load_config(args.config)
    results = []
    for asset in _resolve_assets(args.asset, args.all):
        summary = train_asset(asset, config, best_checkpoint=args.checkpoint)
        results.append(summary)
        print(
            f"{summary['asset']:>8} | best_reward={summary['training_metrics']['best_reward']:.4f} | "
            f"equity={summary['evaluation']['final_equity']:.4f} | run_dir={summary['run_dir']}"
        )
    return 0


def cmd_evaluate(args):
    config = _load_config(args.config)
    assets = _resolve_assets(args.asset, args.all)
    if args.walk_forward:
        if args.folds <= 0:
            print("[ERROR] folds must be a positive integer.", file=sys.stderr)
            return 1
        if args.fold_size is not None and args.fold_size <= 1:
            print("[ERROR] fold-size must be greater than 1.", file=sys.stderr)
            return 1
        if args.all:
            result = evaluate_walk_forward_all(
                assets=assets,
                config=config,
                checkpoint=args.checkpoint,
                folds=args.folds,
                fold_size=args.fold_size,
                output_dir=args.output_dir,
                include_baselines=args.baselines,
            )
            if args.baselines:
                print_walk_forward_all_baselines(
                    result["baseline_summary_rows"], result["output_dir"]
                )
            else:
                print_walk_forward_all(result["summary_rows"], result["output_dir"])
            return 0

        result = evaluate_walk_forward_asset(
            assets[0],
            config=config,
            checkpoint=args.checkpoint,
            folds=args.folds,
            fold_size=args.fold_size,
            output_dir=args.output_dir,
            include_baselines=args.baselines,
        )
        if args.baselines:
            print_walk_forward_asset_baselines(result)
        else:
            print_walk_forward_asset(result)
        return 0
    if args.baselines:
        print("[ERROR] --baselines is currently supported only with --walk-forward", file=sys.stderr)
        return 1

    try:
        results = evaluate_assets(
            assets=assets,
            config=config,
            checkpoint=args.checkpoint if not args.all else None,
        )
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    for result in results:
        print_result(result)
    if args.all:
        from src.evaluation.benchmark import write_summary

        write_summary(results)
        print(f"Saved evaluation summary for {len(results)} assets.")
    return 0


def cmd_predict(args):
    config = _load_config(args.config)
    try:
        assets = _resolve_prediction_assets(args.asset, args.assets, args.all, args.csv)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if len(assets) == 1:
        asset = assets[0]
        try:
            checkpoint = str(resolve_checkpoint_path(asset, args.checkpoint))
            result = predict_latest(
                asset=asset,
                config=config,
                checkpoint_path=checkpoint,
                csv_path=args.csv,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

        if args.save:
            path = save_prediction(result)
        else:
            path = None

        if args.format == "json":
            payload = json.loads(format_prediction_json(result))
            if path is not None:
                payload["saved_to"] = str(path)
            print(json.dumps(payload, indent=2))
        else:
            print(format_prediction_text(result))
            if path is not None:
                print(f"Saved prediction: {path}")
        return 0

    try:
        results, errors = predict_many(
            assets=assets,
            config=config,
            checkpoint_path=args.checkpoint,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.save:
        path = save_predictions(results, errors)
    else:
        path = None

    if args.format == "json":
        payload = json.loads(format_prediction_many_json(results, errors))
        if path is not None:
            payload["saved_to"] = str(path)
        print(json.dumps(payload, indent=2))
    else:
        print(format_prediction_table(results, errors))
        if path is not None:
            print(f"Saved prediction: {path}")
    return 0 if results else 1


def cmd_diagnose(args):
    config = _load_config(args.config)
    if args.walk_forward:
        print(
            "[ERROR] Walk-forward diagnostics are not implemented in v1 yet. Future work: fold-level diagnostics for walk-forward behavior.",
            file=sys.stderr,
        )
        return 1

    try:
        assets = _resolve_assets(args.asset, args.all)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.all:
        result = collect_all_model_diagnostics(
            config=config,
            save=True if args.save or args.output_dir else True,
            output_dir=args.output_dir,
        )
        if args.format == "json":
            payload = {
                "mode": result["mode"],
                "results": result["results"],
                "output_dir": result["output_dir"],
            }
            print(json.dumps(payload, indent=2))
        else:
            print(format_diagnostics_table(result["summaries"]))
            if result["output_dir"]:
                print(f"Output: {result['output_dir']}")
        return 0

    asset = assets[0]
    try:
        result = collect_model_diagnostics(
            asset,
            config=config,
            checkpoint=args.checkpoint,
            save=True if args.save or args.output_dir else True,
            output_dir=args.output_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        payload = {
            "mode": result["mode"],
            **result["summary"],
        }
        if result["output_dir"]:
            payload["output_dir"] = result["output_dir"]
        print(json.dumps(payload, indent=2))
    else:
        print(format_diagnostics_text(result["summary"]))
        if result["output_dir"]:
            print(f"Output: {result['output_dir']}")
    return 0


def cmd_experiment(args):
    config = _load_config(args.config)
    if args.experiment_type == "reward":
        if args.all:
            print("[ERROR] Reward experiments currently support one asset at a time. Use --asset.", file=sys.stderr)
            return 1
        if not args.asset:
            print("[ERROR] You must provide --asset.", file=sys.stderr)
            return 1
            
        asset = normalize_asset_name(args.asset)
        
        if args.all_presets:
            from src.config.reward_presets import load_reward_presets
            presets = list(load_reward_presets().keys())
        elif args.presets:
            presets = args.presets
        else:
            presets = ["current", "no_action_change_penalty", "low_position_penalty", "reduced_penalty_combo"]
            
        from src.experiments.reward_tuning import run_reward_experiment
        run_reward_experiment(asset, config, presets, quick=args.quick)
        return 0
    elif args.experiment_type == "ppo-std":
        if args.all:
            print("[ERROR] PPO std experiments currently support one asset at a time. Use --asset.", file=sys.stderr)
            return 1
        if not args.asset:
            print("[ERROR] You must provide --asset.", file=sys.stderr)
            return 1
            
        asset = normalize_asset_name(args.asset)
        
        if args.all_presets:
            from src.config.ppo_std_presets import load_ppo_std_presets
            presets = list(load_ppo_std_presets().keys())
        elif args.presets:
            presets = args.presets
        else:
            presets = ["current", "low_entropy", "lower_std_ceiling", "combined_std_control"]
            
        from src.experiments.ppo_std_tuning import run_ppo_std_experiment
        run_ppo_std_experiment(asset, config, presets, quick=args.quick, disable_early_stopping=args.disable_early_stopping)
        return 0
    elif args.experiment_type == "training-signal":
        asset = normalize_asset_name(args.asset)
        from src.experiments.training_signal import run_training_signal_experiment
        run_training_signal_experiment(asset, config, quick=args.quick, disable_early_stopping=getattr(args, "disable_early_stopping", False))
        return 0
    elif args.experiment_type == "feature-ablation":
        if args.all:
            print("[ERROR] Feature ablation experiments currently support one asset at a time. Use --asset.", file=sys.stderr)
            return 1
        if not args.asset:
            print("[ERROR] You must provide --asset.", file=sys.stderr)
            return 1
        asset = normalize_asset_name(args.asset)
        if args.all_presets:
            from src.config.feature_ablation_presets import load_feature_ablation_presets
            presets = list(load_feature_ablation_presets().keys())
        elif args.presets:
            presets = args.presets
        else:
            presets = ["full_features", "no_candle_features", "price_action_minimal"]
        from src.experiments.feature_ablation import run_feature_ablation_experiment
        run_feature_ablation_experiment(asset, config, presets, quick=args.quick)
        return 0
    elif args.experiment_type == "seed-validation":
        if args.all:
            print(
                "[ERROR] Repeated seed validation currently supports one asset at a time. Use --asset.",
                file=sys.stderr,
            )
            return 1
        asset = normalize_asset_name(args.asset or "btc_usdt")
        presets = args.feature_presets or ["full_features", "price_action_minimal"]
        seeds = args.seeds or [42, 43, 44]
        from src.experiments.seed_validation import run_seed_validation_experiment

        run_seed_validation_experiment(
            asset,
            config,
            feature_presets=presets,
            seeds=seeds,
            quick=args.quick,
        )
        return 0
    elif args.experiment_type == "action-mapping":
        if args.all:
            print(
                "[ERROR] Action-mapping experiments currently support one asset at a time. Use --asset.",
                file=sys.stderr,
            )
            return 1
        if not args.asset:
            print("[ERROR] You must provide --asset.", file=sys.stderr)
            return 1
        asset = normalize_asset_name(args.asset)
        from src.experiments.action_mapping import run_action_mapping_experiment

        run_action_mapping_experiment(
            asset,
            config,
            feature_preset=args.feature_preset,
            scales=args.scales or [1, 2, 3, 5],
            quick=args.quick,
        )
        return 0
    elif args.experiment_type == "objective-calibration":
        if not args.asset:
            print("[ERROR] You must provide --asset.", file=sys.stderr)
            return 1
        asset = normalize_asset_name(args.asset)
        from src.experiments.objective_calibration import run_objective_calibration_experiment

        run_objective_calibration_experiment(
            asset,
            config,
            presets=args.presets or ["current", "exposure_penalty_light", "directional_edge_reward", "timing_calibration_combo"],
            feature_preset=args.feature_preset,
            quick=args.quick,
        )
        return 0
    return 1


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            return cmd_train(args)
        if args.command == "evaluate":
            return cmd_evaluate(args)
        if args.command == "diagnose":
            return cmd_diagnose(args)
        if args.command == "predict":
            return cmd_predict(args)
        if args.command == "experiment":
            return cmd_experiment(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
