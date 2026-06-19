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

## Deterministic Evaluation

Training and evaluation now use different execution modes:

- training resets at random start indices and uses capped episodes for PPO data collection
- evaluation starts at the beginning of the test split and runs sequentially through the full held-out period

This makes standalone evaluation reproducible and directly comparable across assets and baselines.

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
- There is no walk-forward validation yet.
- There is no hyperparameter sweep or repeated-seed benchmark report yet.
- Baselines are intentionally simple.
- This is still a research prototype, not a production trading system.

## Additional Documentation

- `docs/architecture.md`
- `docs/current_status.md`
- `docs/evaluation.md`
- `docs/experiment_tracking.md`
