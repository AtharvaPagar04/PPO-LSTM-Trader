# Current Status

## Summary

The repository has been upgraded from a prototype with ad hoc evaluation into a cleaner research-grade ML/RL project structure.

The most important change is that evaluation is now deterministic and full-period:

- it starts from the beginning of the held-out test split
- it runs sequentially through the full test period
- it no longer depends on random environment resets
- it saves structured artifacts and metrics for each asset

The project remains a research prototype. It is better organized and more trustworthy than before, but it is not a validated trading system.

## What Is Now Fixed

- asset naming is standardized around `btc_usdt`, `eth_usdt`, and `sol_usdt`
- legacy asset/file variants are normalized automatically
- standalone evaluation is now the official reproducible entrypoint
- evaluation outputs are saved under `logs/evaluation/`
- training runs now save structured run metadata under `logs/runs/`
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
- they do not include walk-forward validation
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

- no walk-forward validation
- no repeated-seed comparison report
- no formal experiment dashboard
- no hyperparameter sweep framework
- baseline family is still intentionally simple
- model quality itself is still mixed across assets

## Current Readiness

### Good for

- portfolio and research demonstration
- explaining PPO + LSTM trading design
- controlled experimentation
- iteration on feature engineering and evaluation methodology

### Not ready for

- live trading
- production deployment
- strong alpha claims
- strategy commercialization

## Recommended Next Steps

1. Add walk-forward validation across multiple periods.
2. Run repeated-seed experiments and summarize the variance.
3. Add stronger baselines and ablations.
4. Improve checkpoint/model selection criteria.
5. Introduce lightweight experiment comparison tooling over `logs/runs/`.
