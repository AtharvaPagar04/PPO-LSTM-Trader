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
from src.evaluate import evaluate_assets, print_result
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

    predict_parser = subparsers.add_parser("predict", help="Run deterministic model inference.")
    predict_parser.add_argument("--asset", type=str, default=None)
    predict_parser.add_argument("--assets", nargs="+", default=None)
    predict_parser.add_argument("--all", action="store_true")
    predict_parser.add_argument("--config", type=str, default=None)
    predict_parser.add_argument("--checkpoint", type=str, default=None)
    predict_parser.add_argument("--csv", type=str, default=None)
    predict_parser.add_argument("--save", action="store_true")
    predict_parser.add_argument("--format", choices=["text", "json"], default="text")

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


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            return cmd_train(args)
        if args.command == "evaluate":
            return cmd_evaluate(args)
        if args.command == "predict":
            return cmd_predict(args)
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
