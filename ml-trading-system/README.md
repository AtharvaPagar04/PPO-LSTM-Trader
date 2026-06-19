# ML Trading System

An experimental reinforcement-learning trading research project for hourly crypto markets using engineered OHLCV features, rolling LSTM windows, a custom trading environment, and a shared LSTM actor-critic policy trained with PPO.

The project demonstrates a functioning RL trading research pipeline. Current results are experimental and should not be interpreted as evidence of live trading profitability.

## Current Scope

This repository is intentionally limited to research and offline evaluation.

It does:

- fetch hourly OHLCV market data from Binance
- engineer time-series features
- build rolling train/test datasets
- train an LSTM actor-critic policy with PPO
- run deterministic full-period evaluation against baselines
- save checkpoints, metrics, traces, and plots

It does not:

- place live orders
- manage exchange credentials for trading
- deploy a live trading service
- claim production readiness

## Supported Assets

The canonical asset naming convention is now:

- `btc_usdt`
- `eth_usdt`
- `sol_usdt`

Legacy forms such as `btcusdt`, `ethusdt`, `solusdt`, and uppercase Binance symbols are normalized automatically.

## Repository Structure

```text
ml-trading-system/
├── configs/
│   ├── default.yaml
│   ├── ppo_default.yaml
│   └── assets.yaml
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── architecture.md
│   ├── current_status.md
│   ├── evaluation.md
│   └── experiment_tracking.md
├── logs/
│   ├── evaluation/
│   └── runs/
├── models/
├── src/
│   ├── config/
│   ├── data/
│   ├── env/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   ├── ppo/
│   ├── utils/
│   ├── evaluate.py
│   ├── train.py
│   └── train_multi.py
└── tests/
```

## Pipeline

1. Fetch hourly OHLCV candles from Binance.
2. Engineer 10 derived features from raw candles.
3. Build rolling windows of size `20`.
4. Split data into `80/20` train/test partitions.
5. Train a continuous-action PPO policy on randomized training episodes.
6. Evaluate the saved checkpoint deterministically over the full held-out test period.
7. Compare against baseline strategies and save metrics and plots.

## Feature Set

The current preprocessing pipeline produces 10 features:

- `log_return`
- `volatility_10`
- `volatility_20`
- `momentum_5`
- `momentum_10`
- `trend`
- `rsi`
- `body_ratio`
- `range_pct`
- `vol_z`

## Model and PPO Design

The policy is a shared LSTM actor-critic:

- 2-layer LSTM encoder
- shared MLP trunk
- actor mean head
- actor standard deviation head
- critic value head

The action is a continuous scalar in `[-1, 1]`:

- `-1` = fully short
- `0` = flat
- `1` = fully long

The PPO implementation includes:

- Gaussian policy sampling during training
- GAE advantage estimation
- clipped policy objective
- clipped value objective
- entropy regularization
- gradient clipping

## Configuration

The project now uses explicit config files in `configs/`:

- `configs/default.yaml`
- `configs/ppo_default.yaml`
- `configs/assets.yaml`

These files store the current default values for:

- window size
- train/test split
- episode length
- seed
- PPO hyperparameters
- environment penalties
- model dimensions

## Commands

### Process one asset

```bash
./venv/bin/python src/features/feature_engineering.py --asset btc_usdt
```

### Process all supported assets

```bash
./venv/bin/python src/features/process_multi.py
```

### Train one asset

```bash
./venv/bin/python src/train.py --asset btc_usdt
```

### Train all supported assets

```bash
./venv/bin/python src/train_multi.py
```

### Evaluate one asset

```bash
./venv/bin/python src/evaluate.py --asset btc_usdt
```

### Evaluate all supported assets

```bash
./venv/bin/python src/evaluate.py --all
```

## CLI Usage

The project now includes a CLI entrypoint:

```bash
./venv/bin/python -m src.cli <command>
```

### Train one asset

```bash
./venv/bin/python -m src.cli train --asset btc_usdt
```

### Train all assets

```bash
./venv/bin/python -m src.cli train --all
```

### Evaluate one asset

```bash
./venv/bin/python -m src.cli evaluate --asset btc_usdt
```

### Evaluate all assets

```bash
./venv/bin/python -m src.cli evaluate --all
```

### Walk-forward evaluation for one asset

```bash
./venv/bin/python -m src.cli evaluate --asset btc_usdt --walk-forward
```

### Walk-forward evaluation for all assets

```bash
./venv/bin/python -m src.cli evaluate --all --walk-forward
```

### Walk-forward baseline comparison for one asset

```bash
./venv/bin/python -m src.cli evaluate --asset btc_usdt --walk-forward --baselines
```

### Walk-forward baseline comparison for all assets

```bash
./venv/bin/python -m src.cli evaluate --all --walk-forward --baselines
```

### Diagnose one asset

```bash
./venv/bin/python -m src.cli diagnose --asset btc_usdt
```

### Diagnose all supported assets

```bash
./venv/bin/python -m src.cli diagnose --all
```

### Predict from the latest available window

```bash
./venv/bin/python -m src.cli predict --asset btc_usdt
```

### Run Reward Tuning Experiments

```bash
./venv/bin/python -m src.cli experiment reward --asset btc_usdt
```

### Run PPO Std Tuning Experiments

```bash
./venv/bin/python -m src.cli experiment ppo-std --asset btc_usdt
```

### Run Training Signal Diagnostics

```bash
./venv/bin/python -m src.cli experiment training-signal --asset btc_usdt
```

### Predict for all supported assets

```bash
./venv/bin/python -m src.cli predict --all
```

### Predict for a selected subset of assets

```bash
./venv/bin/python -m src.cli predict --assets btc_usdt eth_usdt
```

### Predict from a custom CSV

```bash
./venv/bin/python -m src.cli predict --asset btc_usdt --csv data/raw/BTCUSDT_1h.csv
```

### Predict with JSON output

```bash
./venv/bin/python -m src.cli predict --asset btc_usdt --format json
./venv/bin/python -m src.cli predict --all --format json
```

### Save prediction output

```bash
./venv/bin/python -m src.cli predict --asset btc_usdt --save
./venv/bin/python -m src.cli predict --all --save
```

### Use a custom checkpoint

```bash
./venv/bin/python -m src.cli predict --asset btc_usdt --checkpoint models/btc_usdt_best.pt
./venv/bin/python -m src.cli evaluate --asset btc_usdt --checkpoint models/btc_usdt_best.pt
```

The CLI prediction command returns the model's inferred target exposure from the latest feature window. It is not an order execution system and should not be interpreted as financial advice.

The `predict --all` command runs latest-window deterministic inference for every supported asset using existing checkpoints and data. It does not execute trades or connect to an exchange.

## Deterministic Evaluation

Training and evaluation now use different execution modes:

- training resets at random start indices and uses capped episodes for PPO data collection
- evaluation starts at the beginning of the test split and runs sequentially through the full held-out period

This makes standalone evaluation reproducible and directly comparable across assets and baselines.

## Walk-Forward Evaluation

Walk-forward v1 splits the existing held-out test period into chronological folds and evaluates the existing trained checkpoint on each fold. This helps reveal whether performance is stable across different periods.

It does not retrain per fold. It is a lightweight offline robustness diagnostic over the existing saved model.

Example commands:

```bash
./venv/bin/python -m src.cli evaluate --asset btc_usdt --walk-forward
./venv/bin/python -m src.cli evaluate --all --walk-forward
./venv/bin/python -m src.cli evaluate --asset btc_usdt --walk-forward --folds 5
```

Walk-forward outputs are saved under timestamped directories such as:

```text
logs/walk_forward/{timestamp}_{asset}/
logs/walk_forward/{timestamp}_all/
```

Walk-forward metrics are experimental research diagnostics and should not be interpreted as evidence of live trading profitability.

## Walk-Forward Baseline Comparison

The `--baselines` flag compares the PPO/LSTM policy against simple deterministic baseline strategies inside each chronological walk-forward fold.

Example commands:

```bash
./venv/bin/python -m src.cli evaluate --asset btc_usdt --walk-forward --baselines
./venv/bin/python -m src.cli evaluate --all --walk-forward --baselines
```

This mode evaluates the existing checkpoint only. It does not retrain a new model for each fold.

In the current spot-style setup, `buy_and_hold` is equivalent to `always_long`, so it is kept in detailed baseline traces but excluded from comparison rankings to avoid duplicating the same strategy.

Baseline comparison is an offline research diagnostic. It does not execute trades and should not be interpreted as evidence of live trading profitability.

## 1h Model Diagnostics

Diagnostics measure what the trained policy is doing during deterministic offline evaluation. They do not modify the model, do not retrain, and do not execute trades.

Example commands:

```bash
./venv/bin/python -m src.cli diagnose --asset btc_usdt
./venv/bin/python -m src.cli diagnose --all
./venv/bin/python -m src.cli diagnose --asset btc_usdt --format json
```

The diagnostics report focuses on:

- action neutrality vs directional bias
- turnover and position-change behavior
- policy standard deviation behavior
- reward decomposition into PnL and penalties
- transaction-cost drag
- reward clipping frequency

Current diagnostic evidence on the saved 1h checkpoints shows that the policy is mostly flat across assets, policy std is effectively constant at `0.8187`, and cumulative penalty terms are materially larger than final net PnL.

## Official Current Results

The current official metrics are the deterministic full-period outputs saved under `logs/evaluation/`.

| Asset | RL Final Equity | RL Sharpe | RL Max Drawdown | Steps |
|---|---:|---:|---:|---:|
| `btc_usdt` | 1.0252 | 0.65 | 2.98% | 7600 |
| `eth_usdt` | 0.9773 | -0.42 | 8.40% | 7600 |
| `sol_usdt` | 0.8983 | -0.86 | 19.45% | 7602 |

These numbers come from the existing saved checkpoints evaluated over the full test period, not from retraining in this change.

## Outputs

Standalone evaluation saves:

```text
logs/evaluation/{asset}/metrics.json
logs/evaluation/{asset}/equity_curve.png
logs/evaluation/{asset}/actions.csv
logs/evaluation/{asset}/positions.csv
logs/evaluation/summary.csv
logs/evaluation/summary.json
```

Training runs save structured metadata under:

```text
logs/runs/{timestamp}_{asset}/
```

including config, training log, evaluation metrics, and a run-specific equity plot.

Prediction outputs can be saved under:

```text
logs/predictions/{timestamp}_{asset}_prediction.json
logs/predictions/{timestamp}_all_predictions.json
```

Walk-forward evaluation saves:

```text
logs/walk_forward/{timestamp}_{asset}/fold_metrics.csv
logs/walk_forward/{timestamp}_{asset}/fold_metrics.json
logs/walk_forward/{timestamp}_{asset}/aggregate_metrics.json
logs/walk_forward/{timestamp}_{asset}/summary.txt
logs/walk_forward/{timestamp}_all/all_assets_summary.csv
logs/walk_forward/{timestamp}_all/all_assets_summary.json
```

Walk-forward baseline comparison additionally saves:

```text
logs/walk_forward/{timestamp}_{asset}_baselines/baseline_comparison.csv
logs/walk_forward/{timestamp}_{asset}_baselines/baseline_comparison.json
logs/walk_forward/{timestamp}_{asset}_baselines/baseline_aggregate.json
logs/walk_forward/{timestamp}_all_baselines/all_assets_baseline_summary.csv
logs/walk_forward/{timestamp}_all_baselines/all_assets_baseline_summary.json
```

Diagnostics save:

```text
logs/diagnostics/{timestamp}_{asset}/summary.json
logs/diagnostics/{timestamp}_{asset}/actions.csv
logs/diagnostics/{timestamp}_{asset}/reward_components.csv
logs/diagnostics/{timestamp}_{asset}/diagnostics.txt
logs/diagnostics/{timestamp}_all/all_assets_summary.csv
logs/diagnostics/{timestamp}_all/all_assets_summary.json
```

Experiments save:

```text
logs/experiments/reward_tuning/{timestamp}_{asset}/
logs/experiments/ppo_std_tuning/{timestamp}_{asset}/
logs/experiments/training_signal/{timestamp}_{asset}/
```

## Test Coverage

Focused pytest coverage now exists for:

- asset normalization
- feature engineering
- rolling window generation
- environment behavior
- deterministic evaluation
- PPO utility math

## Known Limitations

- Results still depend on the quality of the saved checkpoints.
- Walk-forward v1 evaluates existing checkpoints only.
- Walk-forward v1 does not retrain per fold.
- Walk-forward baseline comparison v1 also evaluates existing checkpoints only.
- Diagnostics evaluate existing checkpoints only.
- Diagnostics are limited to the current 1h model.
- There is no hyperparameter sweep or repeated-seed benchmark report yet.
- Baselines are intentionally simple.
- This is still a research prototype, not a production trading system.

## Additional Documentation

- `docs/architecture.md`
- `docs/current_status.md`
- `docs/evaluation.md`
- `docs/experiment_tracking.md`
- `docs/model_diagnostics.md`
- `docs/ppo_std_tuning.md`
- `docs/reward_tuning.md`
- `docs/training_signal_diagnostics.md`
