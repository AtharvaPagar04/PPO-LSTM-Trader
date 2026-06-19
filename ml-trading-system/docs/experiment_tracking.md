# Experiment Tracking

## Purpose

Training runs now save structured metadata so experiments are easier to inspect and compare.

## Run Directory Layout

Each training run creates:

```text
logs/runs/{timestamp}_{asset}/
├── run_config.json
├── metrics.json
├── training_log.csv
├── evaluation_metrics.json
├── equity_curve.png
└── evaluation/
```

## File Meanings

### `run_config.json`

Contains:

- canonical asset name
- Binance symbol
- training seed
- git commit when available
- full merged config
- referenced data paths
- checkpoint paths
- train/test dataset shapes
- feature names
- window size
- split timestamps when available

### `metrics.json`

Training summary:

- iterations completed
- best reward observed
- best checkpoint path
- final checkpoint path

### `training_log.csv`

Per-iteration training metrics:

- iteration
- total reward
- rolling average reward
- average position
- policy std mean
- whether the best checkpoint improved

### `evaluation_metrics.json`

Deterministic full-period evaluation result for the selected checkpoint at the end of training.

### `equity_curve.png`

Run-specific evaluation curve plot copied from the evaluation artifacts.

## Standalone Evaluation vs Training Run Outputs

Training run outputs live under:

```text
logs/runs/
```

Official standalone evaluation outputs live under:

```text
logs/evaluation/
```

This separation is intentional:

- `logs/runs/` captures one training experiment
- `logs/evaluation/` captures reusable asset-level evaluation reports

## Recommended Workflow

1. Process data for the asset.
2. Train with `src/train.py`.
3. Inspect `logs/runs/{timestamp}_{asset}/`.
4. Re-run `src/evaluate.py` if you want a fresh standalone report.
5. Compare `logs/evaluation/summary.csv` across assets.

## Current Limitations

- There is no experiment database or dashboard yet.
- There is no automatic best-run comparison across multiple seeds.
- Historical prototype logs in `logs/*.log` are still present and may not match the new deterministic evaluation format.
