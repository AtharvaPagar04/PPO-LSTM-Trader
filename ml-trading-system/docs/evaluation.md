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

## Known Limitations

- The baseline family is still simple.
- The evaluator assumes a single fixed transaction cost model.
- Results remain checkpoint-dependent.
- Full-period deterministic evaluation improves trustworthiness, but it does not replace broader validation like walk-forward analysis.
