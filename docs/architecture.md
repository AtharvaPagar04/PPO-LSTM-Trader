# Architecture

## Overview

This repository implements an offline RL trading research pipeline for hourly crypto time series.

The high-level flow is:

```text
Binance OHLCV
    ->
raw CSV files
    ->
feature engineering
    ->
rolling train/test datasets
    ->
TradingEnv
    ->
LSTM actor-critic policy
    ->
PPO training
    ->
deterministic full-period evaluation
    ->
metrics, traces, and plots
```

## Core Modules

### `src/config/`

Centralized configuration and naming:

- `assets.py`
  Canonical asset registry and normalization helpers.
- `paths.py`
  Shared filesystem paths and artifact resolution with legacy filename fallback.
- `settings.py`
  Config loader for `configs/*.yaml`.

This is the layer that removes hardcoded asset strings from the rest of the repo.

### `src/data/`

- `fetch_data.py`
  Downloads hourly Binance candles for a selected asset and saves them under the canonical asset name.
- `dataset.py`
  Loads processed train/test arrays, metadata, and scalers from disk.

### `src/features/`

- `pipeline.py`
  Main preprocessing implementation.
- `feature_engineering.py`
  CLI entrypoint for one asset.
- `process_multi.py`
  Multi-asset preprocessing runner.

The preprocessing pipeline:

1. Loads raw candles.
2. Sorts by timestamp.
3. Engineers 10 features.
4. Builds rolling windows of size `20`.
5. Splits into `80/20` train/test.
6. Fits `StandardScaler` on train features only.
7. Saves canonical processed artifacts.

### `src/env/`

- `trading_env.py`

`TradingEnv` supports two distinct modes:

- `mode="train"`
  random episode starts with capped length
- `mode="eval"`
  deterministic start at the beginning of the test split with no random reset

The environment is still a continuous-position simulator rather than an order-book or execution engine.

State:

- one `[window_size, num_features]` feature window

Action:

- continuous scalar in `[-1, 1]`

Reward:

- log-return PnL
- transaction cost
- drawdown penalty
- position penalty
- action-change penalty
- clipping

### `src/models/`

- `lstm_encoder.py`
- `actor.py`
- `critic.py`
- `policy.py`

The model is now structured as modular components while preserving the original shared LSTM actor-critic idea.

Architecture:

- 2-layer LSTM encoder
- shared MLP trunk
- actor mean head
- actor std head
- critic value head

### `src/ppo/`

- `ppo_trainer.py`
- `losses.py`
- `rollout_buffer.py`

These modules now separate:

- rollout collection
- GAE computation
- clipped PPO losses
- advantage normalization
- gradient-clipped optimization

### `src/evaluation/`

- `backtest.py`
  Deterministic policy backtest over the full held-out period.
- `baselines.py`
  Deterministic baseline simulations using the same test period and transaction cost assumption.
- `metrics.py`
  Performance metric calculations.
- `plot.py`
  Equity curve plotting.
- `benchmark.py`
  Official evaluation orchestration and output writing.

This is the main reliability upgrade over the prototype.

## Training vs Evaluation Behavior

### Training

Training remains PPO-friendly:

- randomized start index
- capped episode length
- stochastic action sampling from `Normal(mean, std)`

This preserves the core research idea and the original optimization behavior.

### Evaluation

Evaluation is now explicitly different:

- starts at the beginning of the test dataset
- uses policy mean actions
- runs one sequential pass through the full test period
- records full action/position/equity traces
- saves structured artifacts

This separation is critical for reproducibility.

## Artifact Layout

### Processed data

Canonical naming:

```text
data/processed/{asset}_train_windows.npy
data/processed/{asset}_test_windows.npy
data/processed/{asset}_train_price_windows.npy
data/processed/{asset}_test_price_windows.npy
data/processed/{asset}_scaler.pkl
data/processed/{asset}_meta.json
```

### Checkpoints

Canonical naming for new runs:

```text
models/{asset}_best.pt
models/{asset}_final.pt
```

Legacy checkpoints are still resolved automatically for backward compatibility.

### Evaluation outputs

```text
logs/evaluation/{asset}/metrics.json
logs/evaluation/{asset}/equity_curve.png
logs/evaluation/{asset}/actions.csv
logs/evaluation/{asset}/positions.csv
logs/evaluation/summary.csv
logs/evaluation/summary.json
```

### Training run outputs

```text
logs/runs/{timestamp}_{asset}/
├── run_config.json
├── metrics.json
├── training_log.csv
├── evaluation_metrics.json
├── equity_curve.png
└── evaluation/
```

## Reliability Improvements Introduced

- canonical asset registry with legacy normalization
- deterministic full-period evaluation
- official standalone evaluation CLI
- structured config files
- structured run metadata and outputs
- modularized PPO utilities
- modularized actor/critic/encoder implementation
- focused test coverage for critical logic

## Remaining Gaps

- no walk-forward validation
- no repeated-seed experiment aggregation
- no hyperparameter sweep tooling
- no stronger benchmark family beyond simple positional baselines
- still not intended for production trading
