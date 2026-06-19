# Evaluation

## Purpose

This document explains the official evaluation path for the project.

The key design rule is:

Training can be stochastic and episode-based, but evaluation must be deterministic and full-period.

## Why The Old Prototype Was Not Enough

The earlier prototype used:

- random environment reset points
- capped episodes
- partial test slices

That was acceptable for PPO data collection, but it was not appropriate as the main research evaluation method because two evaluation runs could cover different parts of the test set.

## Current Evaluation Design

The current evaluator:

- loads the test split for one asset
- builds the policy using the configured architecture
- loads a saved checkpoint
- resets the environment in `eval` mode
- starts from the beginning of the test dataset
- uses the policy mean action deterministically
- runs one sequential pass through the entire held-out test period
- records a full trace of actions, positions, rewards, costs, returns, equity, and drawdown

## Baselines

The evaluator runs the following baselines over the same full test period:

- always long
- always short
- always flat
- random policy with fixed seed
- buy and hold

All of them use the same transaction cost assumption configured for the environment.

## Commands

### Evaluate one asset

```bash
./venv/bin/python src/evaluate.py --asset btc_usdt
```

### Evaluate one asset with an explicit checkpoint

```bash
./venv/bin/python src/evaluate.py --asset btc_usdt --checkpoint models/btc_usdt_best.pt
```

### Evaluate all supported assets

```bash
./venv/bin/python src/evaluate.py --all
```

The same evaluation flow is also available through the CLI:

```bash
./venv/bin/python -m src.cli evaluate --asset btc_usdt
./venv/bin/python -m src.cli evaluate --all
```

Walk-forward evaluation is also available through the CLI:

```bash
./venv/bin/python -m src.cli evaluate --asset btc_usdt --walk-forward
./venv/bin/python -m src.cli evaluate --all --walk-forward
./venv/bin/python -m src.cli evaluate --asset btc_usdt --walk-forward --folds 5
```

Walk-forward baseline comparison:

```bash
./venv/bin/python -m src.cli evaluate --asset btc_usdt --walk-forward --baselines
./venv/bin/python -m src.cli evaluate --all --walk-forward --baselines
```

The CLI also provides deterministic latest-window inference:

```bash
./venv/bin/python -m src.cli predict --asset btc_usdt
./venv/bin/python -m src.cli predict --all
```

The `predict --all` command runs latest-window deterministic inference for every supported asset using existing checkpoints and data. It does not execute trades or connect to an exchange.

Diagnostics are also available through the CLI:

```bash
./venv/bin/python -m src.cli diagnose --asset btc_usdt
./venv/bin/python -m src.cli diagnose --all
./venv/bin/python -m src.cli diagnose --asset btc_usdt --format json
```

### Reward Tuning Experiment

```bash
./venv/bin/python -m src.cli experiment reward --asset btc_usdt
```

### PPO Std Tuning Experiment

```bash
./venv/bin/python -m src.cli experiment ppo-std --asset btc_usdt
```

## Output Files

Per asset:

```text
logs/evaluation/{asset}/metrics.json
logs/evaluation/{asset}/equity_curve.png
logs/evaluation/{asset}/actions.csv
logs/evaluation/{asset}/positions.csv
```

Cross-asset summary:

```text
logs/evaluation/summary.csv
logs/evaluation/summary.json
```

Walk-forward outputs:

```text
logs/walk_forward/{timestamp}_{asset}/fold_metrics.csv
logs/walk_forward/{timestamp}_{asset}/fold_metrics.json
logs/walk_forward/{timestamp}_{asset}/aggregate_metrics.json
logs/walk_forward/{timestamp}_{asset}/summary.txt
logs/walk_forward/{timestamp}_all/all_assets_summary.csv
logs/walk_forward/{timestamp}_all/all_assets_summary.json
```

Walk-forward baseline comparison outputs:

```text
logs/walk_forward/{timestamp}_{asset}_baselines/baseline_comparison.csv
logs/walk_forward/{timestamp}_{asset}_baselines/baseline_comparison.json
logs/walk_forward/{timestamp}_{asset}_baselines/baseline_aggregate.json
logs/walk_forward/{timestamp}_all_baselines/all_assets_baseline_summary.csv
logs/walk_forward/{timestamp}_all_baselines/all_assets_baseline_summary.json
```

Diagnostics outputs:

```text
logs/diagnostics/{timestamp}_{asset}/summary.json
logs/diagnostics/{timestamp}_{asset}/actions.csv
logs/diagnostics/{timestamp}_{asset}/reward_components.csv
logs/diagnostics/{timestamp}_{asset}/diagnostics.txt
logs/diagnostics/{timestamp}_all/all_assets_summary.csv
logs/diagnostics/{timestamp}_all/all_assets_summary.json
```

## Metrics

The RL policy and all baselines currently report:

- `final_equity`
- `total_return`
- `period_return`
- `annualized_return`
- `sharpe`
- `sortino`
- `max_drawdown`
- `calmar`
- `win_rate`
- `average_position`
- `turnover`
- `number_of_steps`
- `average_transaction_cost`

## Deterministic vs Training Behavior

### Training mode

- random episode start
- capped episode length
- stochastic action sampling

### Evaluation mode

- start at the first test window
- full sequential pass
- deterministic mean action

This separation is intentional and should be preserved.

## Walk-Forward Evaluation

Walk-forward v1 splits the existing held-out test period into chronological folds and evaluates the existing trained checkpoint on each fold.

The fold order is chronological:

- no shuffling
- no random resets
- each fold covers one continuous slice of the test period

The purpose is to measure whether the saved model behaves consistently across different time segments of the held-out period.

This v1 implementation does not retrain per fold.

It is an offline research evaluation mode only. It does not imply live trading profitability.

## Walk-Forward Baseline Comparison

The `--baselines` flag compares the PPO/LSTM policy against simple deterministic baseline strategies inside each chronological fold.

The comparison currently includes:

- `always_long`
- `always_short`
- `always_flat`
- `random`

`buy_and_hold` is also computed in detailed traces, but in the current spot-style setup it is equivalent to `always_long`, so it is excluded from rankings and aggregate win counts.

This mode is intended to answer a narrow research question:

Does the saved PPO/LSTM checkpoint outperform simple baseline strategies across different chronological segments of the held-out test period?

Baseline comparison is an offline research diagnostic. It does not execute trades and should not be interpreted as evidence of live trading profitability.

## 1h Model Diagnostics

The `diagnose` command measures what the trained 1h PPO/LSTM policy is doing during deterministic offline evaluation.

It reports:

- action distribution and neutrality ratios
- average exposure and turnover
- policy standard deviation behavior
- value estimate range
- reward decomposition into net PnL, transaction cost, drawdown penalty, position penalty, and action-change penalty
- reward clipping frequency

This mode is useful before any reward tuning or PPO hyperparameter changes, because it shows whether the current policy is inactive, over-penalized, or operating with persistently high uncertainty.

Current diagnostic evidence on the saved checkpoints shows:

- the policy is mostly flat on all three assets
- policy std is effectively constant at `0.8187`
- cumulative penalty terms are materially larger than final net PnL

Diagnostics are offline model behavior measurements. They do not retrain the model, do not modify the reward function, and do not imply live trading profitability.

## Known Limitations

- The baseline family is still simple.
- The evaluator assumes a single fixed transaction cost model.
- Results remain checkpoint-dependent.
- Walk-forward v1 evaluates existing checkpoints only.
- Walk-forward v1 does not retrain per fold.
- Walk-forward baseline comparison v1 evaluates existing checkpoints only.
- Walk-forward baseline comparison does not retrain per fold.
- Diagnostics evaluate existing checkpoints only.
- Diagnostics are implemented for the current 1h model only.
- CLI evaluation is still offline model evaluation only; it does not connect to exchanges or place trades.
- CLI prediction outputs are experimental model signals only and should not be interpreted as financial advice or execution instructions.
