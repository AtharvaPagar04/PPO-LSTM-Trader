# PPO-LSTM Trader

## Project Overview

PPO-LSTM Trader is an offline reinforcement-learning research project for crypto trading on 1h market data. It tests whether an LSTM + PPO agent can learn continuous long/short position sizing from engineered OHLCV features and cross-asset context.

This is an offline research project, not a live trading bot.

The action space is continuous:

```text
-1.0 = fully short
 0.0 = flat
+1.0 = fully long
```

## Final Status

The project is complete as a research framework.

Final conclusion:

```text
The LSTM + PPO pipeline works technically, but the current approach does not produce a confirmed tradable edge.
```

Final classification:

```text
Successful research framework.
Unsuccessful trading strategy.
```

## Important Disclaimer

This repository does not implement live execution, exchange connectivity, broker integration, or risk controls for real trading. Nothing here should be treated as trading advice, a profitability claim, or a production-ready system.

## What This Project Does

- Builds an offline 1h crypto research pipeline around PPO and an LSTM actor-critic model.
- Trains a policy to size continuous long/short exposure on held-out data.
- Evaluates saved checkpoints with deterministic full-period backtests.
- Compares RL behavior against simple and exposure-equivalent baselines.
- Supports experiment workflows for reward tuning, PPO std tuning, signal audits, feature presets, and supervised validation.

## What This Project Does Not Do

- It does not place trades.
- It does not run paper trading.
- It does not claim stable alpha.
- It does not prove the current strategy is robust after costs.
- It does not show that RL is the right first tool for the current feature set.

## Repository Structure

```text
.
├── configs/
├── docs/
│   ├── architecture.md
│   ├── evaluation.md
│   ├── experiments.md
│   ├── model_reference.md
│   └── project_closure.md
├── src/
├── tests/
├── README.md
└── requirements.txt
```

## Trading Problem Definition

The core question was:

```text
Can an LSTM + PPO agent learn useful market timing from 1h crypto OHLCV data
and basic cross-asset return features?
```

The environment treats each action as a direct target position in `[-1, 1]`, applies transaction costs when exposure changes, and scores behavior with reward terms tied to PnL, drawdown, and position management.

## Model Design

The model is a shared LSTM actor-critic policy:

```text
feature window
  -> LSTM encoder
  -> shared MLP trunk
  -> actor mean head
  -> actor std head
  -> critic value head
```

The actor outputs the mean and standard deviation of a Gaussian policy. PPO samples from that distribution during training and uses deterministic mean actions during evaluation.

## Feature Engineering

The default feature set uses 1h OHLCV-derived technical signals such as:

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

Later experiments added regime and cross-asset context, including ETH/SOL returns, BTC-relative spreads, market-average returns, volatility regime signals, and rolling BTC correlation features.

## Training Pipeline

The training flow is:

```text
raw CSV data
  -> feature engineering
  -> rolling windows
  -> train/test split
  -> scaling on train only
  -> TradingEnv
  -> LSTMPolicy
  -> PPO rollout collection
  -> PPO updates
  -> checkpointing and run artifacts
```

Training uses stochastic action sampling, random episode starts, and capped episode length. Evaluation deliberately does not.

## Evaluation Methodology

The project uses deterministic full-period evaluation as the official check:

- start at the beginning of the test split
- run one sequential pass
- use policy mean actions
- include transaction costs
- compare against static and random baselines
- compare against exposure-equivalent baselines

Walk-forward evaluation is also used to test whether behavior is stable across chronological folds of the held-out period.

The most important validation rule is:

```text
If the model cannot beat a constant strategy with the same average signed exposure,
it has not proven useful timing.
```

In this repository that check is reported through `constant_signed_mean_action`.

## Key Results

Official deterministic RL evaluation from saved checkpoints:

| Asset | Final Equity | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: |
| `btc_usdt` | 1.0252 | 0.65 | 2.98% |
| `eth_usdt` | 0.9773 | -0.42 | 8.40% |
| `sol_usdt` | 0.8983 | -0.86 | 19.45% |

Reduced feature preset validation:

```text
stable_cross_asset_core_v1:
  Best WF Balanced Accuracy: 0.5281
  Trading WF Return: -7.54%
  Trading WF Sharpe: -2.10
  Beat Constant Signed Mean: 0/5

stable_cross_asset_core_v2:
  Best WF Balanced Accuracy: 0.5272
  Trading WF Return: -7.83%
  Trading WF Sharpe: -2.18
  Beat Constant Signed Mean: 0/5
```

Why this matters:

```text
The reduced presets slightly improved label-level prediction, but failed trading validation after costs.
```

## What Worked

- The repo now has a coherent offline research workflow with CLI entrypoints, configs, reproducible artifacts, and focused test coverage.
- Deterministic evaluation is materially stronger than the earlier episode-style prototype checks.
- Walk-forward and exposure-equivalent baselines make failure modes easier to diagnose.
- Feature and target audit tooling can identify weak but measurable predictive structure.

## What Failed

- PPO did not demonstrate a stable timing edge across assets.
- Reward tuning did not convert the policy into a validated strategy.
- PPO std tuning did not solve the flat or weak-confidence behavior in a useful way.
- Objective calibration and action-side adjustments did not produce robust trading gains.
- Reduced feature presets improved label metrics slightly but still lost money after costs.
- The strategy failed the key metric:

```text
beat_constant_signed_mean_action < 3/5
```

If the model cannot beat a constant strategy with the same average signed exposure, it has not proven useful timing.

## Reason For Closure

The limiting factor is feature/data signal, not the PPO implementation.

The current 1h OHLCV and basic cross-asset return features contain weak statistical signal, but not enough stable, cost-aware signal for trading.

The evidence points away from more tuning on the same inputs. This version is unlikely to be fixed by:

- PPO tuning
- reward shaping
- action mapping changes
- larger LSTM model
- more training on the same features

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Common Commands

```bash
./venv/bin/python -m src.cli train --asset btc_usdt
./venv/bin/python -m src.cli evaluate --asset btc_usdt
./venv/bin/python -m src.cli evaluate --all
./venv/bin/python -m src.cli evaluate --asset btc_usdt --walk-forward --baselines
./venv/bin/python -m src.cli diagnose --asset btc_usdt
./venv/bin/python -m src.cli predict --asset btc_usdt
./venv/bin/python -m src.cli experiment reward --asset btc_usdt --quick
./venv/bin/python -m src.cli experiment ppo-std --asset btc_usdt --quick
./venv/bin/python -m src.cli experiment training-signal --asset btc_usdt --quick
```

## Testing

```bash
./venv/bin/python -m pytest tests
```

For this documentation update, only markdown and optional compile sanity are relevant; no strategy claim depends on test execution here.

## Documentation Index

- [Architecture](docs/architecture.md)
- [Evaluation](docs/evaluation.md)
- [Experiments](docs/experiments.md)
- [Model Reference](docs/model_reference.md)
- [Project Closure](docs/project_closure.md)

## Future Work

If the project is ever resumed, the next version should start with data and feature improvements first:

1. Add stronger market information rather than tuning PPO on the same features.
2. Validate new signals with simpler supervised and rule-based baselines.
3. Require cost-aware walk-forward performance and exposure-equivalent baseline wins before returning to RL.

Candidate future inputs include order-flow, open interest, funding, liquidations, and richer regime context.

## Final Verdict

This repository succeeds as an honest research framework for testing continuous position sizing with PPO and an LSTM policy on 1h crypto data. It does not succeed as evidence of a tradable strategy.

The correct bottom line is:

```text
The framework is usable for offline research.
The current strategy is not validated for trading.
```
