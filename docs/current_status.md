# Current Status

## Summary

The repository has been upgraded from a prototype with ad hoc evaluation into a cleaner research-grade ML/RL project structure.

This project is now effectively closed as an offline research project rather than an active trading-model candidate.

Final conclusion:

```text
The research framework is successful, but the trading approach is not validated.
The current LSTM + PPO setup does not show a confirmed tradable timing edge.
It should not be used for live trading, paper trading, or profitability claims.
```

The most important change is that evaluation is now deterministic and full-period:

- it starts from the beginning of the held-out test split
- it runs sequentially through the full test period
- it no longer depends on random environment resets
- it saves structured artifacts and metrics for each asset

The project remains a research prototype. It is better organized and more trustworthy than before, but it is not a validated trading system. See [project_closure.md](project_closure.md) for the final closure summary.

## What Is Now Fixed

- asset naming is standardized around `btc_usdt`, `eth_usdt`, and `sol_usdt`
- legacy asset/file variants are normalized automatically
- standalone evaluation is now the official reproducible entrypoint
- a CLI now wraps training, evaluation, and model inference
- evaluation outputs are saved under `logs/evaluation/`
- training runs now save structured run metadata under `logs/runs/`
- prediction outputs can be saved under `logs/predictions/`
- multi-asset latest-window prediction is available through `predict --all`
- walk-forward evaluation is available through `evaluate --walk-forward`
- walk-forward baseline comparison is available through `evaluate --walk-forward --baselines`
- deterministic 1h model diagnostics are available through `diagnose`
- reward tuning experiment framework is available
- PPO std/entropy tuning experiment framework is available
- training signal and advantage diagnostics are available
- experiment audit manifests and detailed training traces are generated
- focused pytest coverage exists for core logic
- previously empty placeholder model/PPO modules are now implemented
- unused placeholder files were removed

## Current Official Evaluation Results

The official current results are the deterministic full-period evaluations generated from the existing saved checkpoints in `models/` and saved under `logs/evaluation/`.

| Asset | Checkpoint Used | Final Equity | Total Return | Sharpe | Sortino | Max Drawdown | Steps |
|---|---|---:|---:|---:|---:|---:|---:|
| `btc_usdt` | `models/btc_usdt_model.pth` | 1.0252 | 0.0252 | 0.65 | 0.91 | 2.98% | 7600 |
| `eth_usdt` | `models/ethusdt_model.pth` | 0.9773 | -0.0227 | -0.42 | -0.53 | 8.40% | 7600 |
| `sol_usdt` | `models/solusdt_model.pth` | 0.8983 | -0.1017 | -0.86 | -0.94 | 19.45% | 7602 |

## Baseline Comparison Snapshot

### `btc_usdt`

- RL policy final equity: `1.0252`
- always long: `0.8147`
- always short: `1.2264`
- always flat: `1.0000`
- random baseline: `0.0731`

Interpretation:

- the RL policy is mildly positive on the full test period
- it beats always-long and random
- it does not beat always-short on this checkpoint

### `eth_usdt`

- RL policy final equity: `0.9773`
- always long: `1.0877`
- always short: `0.9187`
- always flat: `1.0000`
- random baseline: `0.0609`

Interpretation:

- the RL policy is slightly negative over the full test period
- it beats always-short and random
- it does not beat always-long or flat

### `sol_usdt`

- RL policy final equity: `0.8983`
- always long: `0.6466`
- always short: `1.5454`
- always flat: `1.0000`
- random baseline: `0.0317`

Interpretation:

- the RL policy is negative on the full test period
- it beats always-long and random
- it trails always-short and flat

## How To Interpret These Results

These metrics are more trustworthy than the earlier prototype logs because they use full-period deterministic evaluation.

They still should not be over-interpreted:

- they are based on a small asset set
- they depend on existing checkpoints rather than repeated retraining
- walk-forward validation is now available, but v1 reuses existing checkpoints rather than retraining per fold
- they do not prove robustness across market regimes

The correct claim is:

The project demonstrates a functioning RL trading research pipeline with reproducible offline evaluation.

The incorrect claim would be:

The model is profitable or production-ready.

## Current Strengths

- reproducible evaluation flow
- explicit config layer
- centralized asset normalization
- structured artifact layout
- lightweight experiment tracking
- focused coverage for preprocessing, environment behavior, evaluation, and PPO utilities

## Current Weaknesses

- walk-forward v1 does not retrain per fold
- no repeated-seed comparison report
- no formal experiment dashboard
- no hyperparameter sweep framework
- baseline family is still intentionally simple
- model quality itself is still mixed across assets

## 1h Model Diagnostics

The `diagnose` command measures what the trained 1h PPO/LSTM policy is actually doing during deterministic offline evaluation.

Example commands:

```bash
./venv/bin/python -m src.cli diagnose --asset btc_usdt
./venv/bin/python -m src.cli diagnose --all
./venv/bin/python -m src.cli diagnose --asset btc_usdt --format json
```

The current diagnostic findings from the saved checkpoints are:

- the policy is overwhelmingly flat across all three assets
- policy std is effectively constant at `0.8187`
- cumulative penalty terms are much larger than final net PnL
- BTC remains mildly positive despite this, while ETH and SOL remain negative

Diagnostics are offline model behavior measurements. They do not retrain the model and do not imply live trading profitability.

## Training Signal Diagnostics

The `training-signal` experiment is a dedicated diagnostic tool to audit the deep PPO updates and advantage signals during training.

Example commands:

```bash
./venv/bin/python -m src.cli experiment training-signal --asset btc_usdt --quick
```

This generates:
- `signal_summary.json`: High-level metrics tracking whether actor mean is stagnant, advantages are collapsed, or PPO updates are too small.
- `training_trace.csv`: Detailed per-iteration tracking of raw vs. normalized advantages, value function scale, actor/critic gradient norms, and approx KL / clip fractions.
- `report.md`: Human-readable interpretation with automated warnings and next step recommendations.

## Walk-Forward Evaluation

Walk-forward v1 splits the existing held-out test period into chronological folds and evaluates the existing trained checkpoint on each fold.

This improves time-segment robustness diagnostics without introducing long retraining jobs.

Example commands:

```bash
./venv/bin/python -m src.cli evaluate --asset btc_usdt --walk-forward
./venv/bin/python -m src.cli evaluate --all --walk-forward
./venv/bin/python -m src.cli evaluate --asset btc_usdt --walk-forward --folds 5
```

Per-asset outputs are saved under:

```text
logs/walk_forward/{timestamp}_{asset}/
```

All-asset summary outputs are saved under:

```text
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

This mode reuses the existing saved checkpoint. It does not retrain a new model per fold.

In the current spot-style setup, buy-and-hold is equivalent to always-long, so it is preserved in detailed traces but excluded from fold rankings.

Baseline comparison is an offline research diagnostic. It does not execute trades and should not be interpreted as evidence of live trading profitability.

## Current Readiness

### Good for

- portfolio and research demonstration
- explaining PPO + LSTM trading design
- controlled experimentation
- command-line model training/evaluation/inference
- cross-asset offline prediction from existing checkpoints
- iteration on feature engineering and evaluation methodology

### Not ready for

- live trading
- production deployment
- exchange execution
- paper broker workflows
- strong alpha claims
- strategy commercialization

CLI prediction outputs are experimental model signals only and should not be interpreted as financial advice or execution instructions.

## Recommended Next Steps

The recommended next step is no longer more PPO tuning on this version.

1. Stop treating the current 1h OHLCV-based PPO setup as a trading candidate.
2. If work resumes, start with stronger data and feature research rather than RL tuning.
3. Revalidate any future feature ideas with supervised walk-forward tests and same-exposure baselines before returning to PPO.
