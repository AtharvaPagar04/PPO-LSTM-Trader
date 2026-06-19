# Evaluation

## Purpose

This document defines how model quality is judged in this repository. The key principle is that training may be stochastic, but the official evaluation must be deterministic, full-period, cost-aware, and baseline-relative.

## Deterministic Evaluation

The official evaluation path runs the saved policy over the full held-out test split:

- start at test index `0`
- use policy mean actions
- make one sequential pass
- apply transaction costs on exposure changes
- record full traces and summary metrics

This avoids the ambiguity of episode-based prototype checks where different runs could score different slices of the test set.

Primary commands:

```bash
./venv/bin/python -m src.cli evaluate --asset btc_usdt
./venv/bin/python -m src.cli evaluate --all
```

## Walk-Forward Evaluation

Walk-forward evaluation slices the held-out test block into chronological folds and evaluates the existing checkpoint on each fold.

Primary commands:

```bash
./venv/bin/python -m src.cli evaluate --asset btc_usdt --walk-forward
./venv/bin/python -m src.cli evaluate --asset btc_usdt --walk-forward --baselines
./venv/bin/python -m src.cli evaluate --all --walk-forward
```

What it measures:

- return stability across time segments
- fold-level Sharpe consistency
- worst fold drawdown
- how often the RL policy actually wins a fold

Important limit:

```text
Walk-forward v1 does not retrain per fold.
It evaluates the existing checkpoint across chronological held-out slices.
```

## Baseline Comparison

Each evaluation compares the RL policy against simple baselines run on the same price windows and transaction cost assumption.

Static baselines:

- `always_long`
- `always_short`
- `always_flat`
- `random`
- `buy_and_hold`

In the current spot-style setup, `buy_and_hold` is effectively equivalent to `always_long`.

## Exposure-Equivalent Baselines

Static long/short comparisons alone are not enough because a policy can appear better simply by carrying the right average directional bias.

This repository therefore adds exposure-equivalent baselines:

- `constant_signed_mean_action`
- `constant_abs_mean_long`
- `constant_abs_mean_short`

The most important one is `constant_signed_mean_action`:

- compute the RL policy's average signed action over the evaluation period
- hold that constant exposure for the same period
- compare returns directly

Why it matters:

```text
If the model cannot beat a constant strategy with the same average signed exposure,
it has not proven useful timing.
```

That is the core defense against mistaking static bias for true market timing skill.

## Transaction Costs

Transaction costs are applied whenever the position changes. In both the environment and baseline simulator, the cost is:

```text
abs(new_position - old_position) * transaction_cost
```

Default config:

```text
transaction_cost = 0.0004
```

This matters because small predictive effects can disappear once turnover is priced realistically.

## Action Diagnostics

The repo includes deterministic diagnostics to inspect what the trained policy is actually doing:

```bash
./venv/bin/python -m src.cli diagnose --asset btc_usdt
./venv/bin/python -m src.cli diagnose --all
```

Diagnostics track:

- flat/long/short action ratios
- average absolute action
- turnover
- policy std behavior
- reward decomposition
- clipping frequency
- transaction-cost drag

These metrics help separate "bad model" from "no usable signal" or "over-penalized reward."

## Key Evaluation Metrics

Core metrics reported by the project include:

- `final_equity`
- `total_return`
- `sharpe`
- `sortino`
- `max_drawdown`
- `calmar`
- `win_rate`
- `average_position`
- `turnover`
- `average_transaction_cost`
- walk-forward positive fold count
- walk-forward robustness score

For experiments and supervised trading checks, additional metrics include:

- walk-forward balanced accuracy
- beat counts versus exposure-equivalent baselines
- action-side concentration and flat ratios

## Why Final Equity Alone Is Insufficient

Final equity is necessary but not sufficient.

A strategy can show positive equity while still failing on:

- risk-adjusted return
- drawdown behavior
- consistency across folds
- post-cost robustness
- exposure-equivalent baseline comparison

Example: a mildly positive RL result may still be unconvincing if it only reflects a persistent short bias that a constant-exposure baseline can match or beat.

## Current Deterministic Results

Official saved-checkpoint results:

| Asset | Final Equity | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: |
| `btc_usdt` | 1.0252 | 0.65 | 2.98% |
| `eth_usdt` | 0.9773 | -0.42 | 8.40% |
| `sol_usdt` | 0.8983 | -0.86 | 19.45% |

Interpretation:

- BTC is mildly positive, not decisive.
- ETH is negative.
- SOL is materially negative.
- Cross-asset consistency is not present.

## Reduced Preset Validation

Reduced feature presets improved label-level metrics slightly but failed the trading standard that matters:

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

This is the clearest example of why label metrics alone are not enough. Balanced accuracy moved slightly above random, but the strategies still failed post-cost trading validation and never beat `constant_signed_mean_action`.

## Output Locations

Deterministic evaluation:

```text
logs/evaluation/{asset}/metrics.json
logs/evaluation/{asset}/equity_curve.png
logs/evaluation/{asset}/actions.csv
logs/evaluation/{asset}/positions.csv
logs/evaluation/summary.csv
logs/evaluation/summary.json
```

Walk-forward:

```text
logs/walk_forward/{timestamp}_{asset}/
logs/walk_forward/{timestamp}_{asset}_baselines/
logs/walk_forward/{timestamp}_all/
logs/walk_forward/{timestamp}_all_baselines/
```

Diagnostics:

```text
logs/diagnostics/{timestamp}_{asset}/
logs/diagnostics/{timestamp}_all/
```

## Evaluation Summary

The evaluation framework is one of the successful parts of the project. It makes weak strategies harder to overclaim. The final judgment is negative not because the evaluation is missing, but because the current strategy does not pass it.
