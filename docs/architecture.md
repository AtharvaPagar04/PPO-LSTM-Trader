# Architecture

## System Overview

This repository implements an offline research pipeline for continuous crypto position sizing on 1h data. The project turns raw market data into rolling feature windows, trains an LSTM actor-critic policy with PPO, and evaluates saved checkpoints with deterministic backtests and walk-forward diagnostics.

High-level flow:

```text
raw OHLCV CSV
  -> feature engineering
  -> rolling windows + scaling
  -> TradingEnv
  -> LSTMPolicy
  -> PPO training
  -> deterministic evaluation
  -> walk-forward and diagnostic analysis
```

## Directory Structure

```text
configs/
  Static configuration, reward presets, PPO std presets, feature presets.

docs/
  Project documentation and closure record.

src/
  Main application code.

tests/
  Focused regression coverage for environment, evaluation, CLI, and experiments.
```

Key source modules:

```text
src/
├── cli.py
├── train.py
├── train_multi.py
├── evaluate.py
├── inference.py
├── config/
├── data/
├── env/
├── evaluation/
├── experiments/
├── features/
├── models/
├── ppo/
└── utils/
```

## Data Flow

The data path is implemented around processed rolling windows rather than online candles:

1. Raw CSV files are resolved per asset through `src.config.paths`.
2. `src.features.pipeline` loads and cleans OHLCV data.
3. Feature engineering creates technical, regime, and optional cross-asset features.
4. `create_windows()` converts feature rows and price rows into aligned rolling windows.
5. Windows are split chronologically into train and test sets.
6. `StandardScaler` is fit on train windows only and applied to both splits.
7. Processed arrays, scaler artifacts, and metadata are stored under `data/processed/`.

The metadata records feature names, split ratio, window size, and timestamp boundaries so training and evaluation use the same processed layout.

## Feature Pipeline

Default base features include:

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

The pipeline also supports:

- regime features such as `return_24`, `return_72`, `volatility_regime`, `trend_strength_24`
- cross-asset features such as `eth_return_24`, `sol_return_72`, `market_avg_return_24`, `btc_relative_strength_24`

Cross-asset features are computed only when a selected preset requires them. ETH and SOL raw data must be available for those presets.

## Environment Design

`src.env.trading_env.TradingEnv` is a continuous-position simulator, not an execution engine.

State:

- one scaled feature window with shape `[window_size, num_features]`

Action:

- one continuous scalar clipped to `[-1.0, 1.0]`

Position semantics:

```text
-1.0 = fully short
 0.0 = flat
+1.0 = fully long
```

Training behavior:

- random episode starts when enough data exists
- capped episode length from config
- stochastic policy sampling

Evaluation behavior:

- deterministic reset at test index `0`
- full sequential pass through the held-out split
- policy mean action only

Reward components are computed in `step()` from:

- log-return PnL
- transaction cost on position change
- drawdown penalty
- position penalty
- action-change penalty
- optional exposure, turnover, directional, and volatility-exposure terms
- reward scaling and clipping

The environment also exposes detailed reward components in `info` for diagnostics.

## Model Architecture

The implemented policy is `src.models.policy.LSTMPolicy`.

Architecture:

1. `LSTMEncoder`
2. shared MLP trunk
3. `ActorHead`
4. `CriticHead`

Concrete structure:

- LSTM encoder with configurable hidden size, layer count, and dropout
- shared MLP: `Linear(hidden_dim, 128) -> ReLU -> Linear(128, 128) -> ReLU`
- actor mean head with `tanh` output
- actor std head with bounded log-std
- critic head with scalar value output

The actor supports two std parameterizations:

- `hard_clamp`
- `smooth_bound`

## PPO Training Loop

Training is orchestrated by `src.train.train_asset()` and `src.ppo.PPOTrainer`.

Loop outline:

1. Build processed dataset and train environment.
2. Build `LSTMPolicy`.
3. Collect rollouts with sampled Gaussian actions.
4. Compute GAE advantages and returns.
5. Normalize advantages.
6. Optimize PPO losses across minibatches and epochs.
7. Track update diagnostics such as KL, clip fraction, actor/critic gradients, and policy std.
8. Save best and final checkpoints plus run artifacts.

Loss terms include:

- clipped policy loss
- clipped value loss
- entropy bonus
- std stability penalty

## CLI Entrypoints

Main public entrypoint:

```bash
./venv/bin/python -m src.cli <command>
```

Important commands:

- `train`
- `evaluate`
- `diagnose`
- `predict`
- `experiment reward`
- `experiment ppo-std`
- `experiment training-signal`
- `experiment feature-ablation`
- `experiment seed-validation`
- `experiment objective-calibration`
- `experiment signal-audit`
- `experiment target-audit`
- `experiment feature-signal-audit`
- `experiment supervised-signal-strategy`

Supporting scripts:

- `src/train.py`
- `src/evaluate.py`
- `src/inference.py`
- `src/train_multi.py`

## Artifact and Log Structure

Processed data:

```text
data/processed/{asset}_train_windows.npy
data/processed/{asset}_test_windows.npy
data/processed/{asset}_train_price_windows.npy
data/processed/{asset}_test_price_windows.npy
data/processed/{asset}_scaler.pkl
data/processed/{asset}_meta.json
```

Checkpoints:

```text
models/{asset}_best.pt
models/{asset}_final.pt
models/{asset}_model.pth
models/{asset}_model_final.pth
```

Standalone evaluation:

```text
logs/evaluation/{asset}/
logs/evaluation/summary.csv
logs/evaluation/summary.json
```

Training runs:

```text
logs/runs/{timestamp}_{asset}/
```

Diagnostics:

```text
logs/diagnostics/{timestamp}_{asset}/
logs/diagnostics/{timestamp}_all/
```

Walk-forward outputs:

```text
logs/walk_forward/{timestamp}_{asset}/
logs/walk_forward/{timestamp}_{asset}_baselines/
logs/walk_forward/{timestamp}_all/
logs/walk_forward/{timestamp}_all_baselines/
```

Experiment outputs:

```text
logs/experiments/<experiment_type>/{timestamp}_{asset}/
```

## Design Summary

The repo is strongest as a research framework because the pipeline is explicit end to end: features, environment, model, PPO updates, evaluation, baselines, and diagnostics all exist as separate modules. The main limitation is not missing architecture, but weak tradable signal in the current 1h feature set.
