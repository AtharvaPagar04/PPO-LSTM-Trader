# Reward Tuning

## Purpose

Recent diagnostics showed that the 1h model policy is mostly flat (e.g. 99.9% flat on btc_usdt) and that reward penalties dominate the useful trading signal (PnL). The policy standard deviation is high and constant around `0.8187`. 

Reward Tuning v1 is a controlled experiment framework to test whether lighter reward penalties allow the policy to take more meaningful positions and improve against baselines.

## Experimental Presets

The following reward presets are configured in `configs/reward_presets.yaml`:

- `current`: Matches current default reward behavior exactly.
- `no_action_change_penalty`: Tests whether `action_change_penalty` is double-counting turnover (since transaction cost already applies).
- `low_position_penalty`: Tests whether `position_penalty` is forcing the model to stay flat.
- `no_position_penalty`: Tests whether `position_penalty` is the main cause of flat-policy collapse.
- `low_drawdown_penalty`: Tests whether drawdown penalty is making the model overly conservative.
- `reduced_penalty_combo`: A practical candidate config combining reduced penalties.
- `pnl_focused`: Aggressive diagnostic preset to see if directional exposure is learned when structural penalties are removed.

**Note**: Transaction costs remain enabled in all presets as a realistic friction term.

## CLI Commands

You can run an experiment for an asset using the CLI:

```bash
# Run default presets
./venv/bin/python -m src.cli experiment reward --asset btc_usdt

# Run specific presets
./venv/bin/python -m src.cli experiment reward --asset btc_usdt --presets current no_action_change_penalty low_position_penalty

# Run a quick test (fewer iterations)
./venv/bin/python -m src.cli experiment reward --asset btc_usdt --quick

# Run all presets
./venv/bin/python -m src.cli experiment reward --asset btc_usdt --all-presets
```

Currently, `--all` (all assets) is disabled for reward experiments to keep them focused.

## Output Files

Experiments output to `logs/experiments/reward_tuning/{timestamp}_{asset}/`:

- `{preset}/config.yaml`: The applied config
- `{preset}/checkpoint.pth`: The trained checkpoint
- `{preset}/evaluation/`: Deterministic evaluation outputs
- `{preset}/diagnostics/`: Diagnostic outputs
- `{preset}/walk_forward/`: Walk-forward evaluation
- `{preset}/walk_forward_baselines/`: Walk-forward baseline comparisons
- `summary.csv`, `summary.json`: Aggregated metrics for all presets
- `report.md`: Markdown report of the experiment results

## Interpretation

A composite score is used to rank the presets:
`score = 2.0 * rl_best_return_fold_count + 1.0 * rl_beat_always_flat_count + 1.0 * rl_beat_random_count + 1.0 * walk_forward_robustness + 1.0 * walk_forward_mean_sharpe - 2.0 * max_drawdown`

**Important**: This score is only a research comparison helper. It is not proof of trading profitability.

## Limitations

Reward Tuning v1 is experimental.
It only tests reward presets.
It does not change the default model unless explicitly selected.
It does not add 5m support.
It does not add paper/live trading.
Reward experiments need multiple seeds before final selection.
Reward tuning experiments are offline research experiments. They do not execute trades and do not prove live trading profitability.
