# Model Diagnostics

## Purpose

Diagnostics measure what the trained policy is doing during deterministic offline evaluation. They do not modify the model, do not retrain, and do not execute trades.

This is a 1h-model-only diagnostic pass. It does not add 5m support.

## Commands

```bash
./venv/bin/python -m src.cli diagnose --asset btc_usdt
./venv/bin/python -m src.cli diagnose --all
./venv/bin/python -m src.cli diagnose --asset btc_usdt --format json
```

Walk-forward diagnostics are not implemented in v1 yet.

## Outputs

Single asset:

```text
logs/diagnostics/{timestamp}_{asset}/summary.json
logs/diagnostics/{timestamp}_{asset}/actions.csv
logs/diagnostics/{timestamp}_{asset}/reward_components.csv
logs/diagnostics/{timestamp}_{asset}/diagnostics.txt
```

All assets:

```text
logs/diagnostics/{timestamp}_all/all_assets_summary.csv
logs/diagnostics/{timestamp}_all/all_assets_summary.json
```

## Key Fields

- `flat_ratio`: fraction of steps with action between `-0.25` and `0.25`
- `long_ratio`: fraction of steps with action above `0.25`
- `short_ratio`: fraction of steps with action below `-0.25`
- `policy_std_mean`: average policy standard deviation from the actor head
- `transaction_cost_drag_ratio`: transaction-cost sum relative to absolute gross PnL
- `position_penalty_sum`: cumulative size penalty
- `drawdown_penalty_sum`: cumulative drawdown penalty
- `action_change_penalty_sum`: cumulative turnover penalty
- `reward_clip_ratio`: fraction of steps where the unclipped reward hit the clipping bound

## Interpretation Guidance

- High `flat_ratio` means the policy is spending most of the evaluation period near neutral exposure.
- Constant `policy_std_mean`, `policy_std_min`, and `policy_std_max` suggest the actor uncertainty is effectively stuck.
- If cumulative penalty sums are much larger than `pnl_sum`, the reward design is likely discouraging activity more than it is rewarding edge.
- A low `reward_clip_ratio` means clipping is probably not the primary bottleneck.
- Large `transaction_cost_drag_ratio` indicates turnover is eating too much of gross PnL.

## Current Findings

Current saved checkpoints show:

- the policy is mostly flat across BTC, ETH, and SOL
- policy std is effectively constant at `0.8187`
- cumulative penalty terms are materially larger than final net PnL
- BTC remains mildly positive, while ETH and SOL remain negative

## Limitations

- Diagnostics are offline only.
- They evaluate existing checkpoints only.
- They do not retrain or improve the model by themselves.
- They do not add 5m support.
- They do not add paper or live trading.
