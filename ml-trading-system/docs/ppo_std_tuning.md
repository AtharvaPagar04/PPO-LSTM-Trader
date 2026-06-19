# PPO Std / Entropy Tuning

## Purpose

Diagnostics and reward experiments showed that reward tuning alone did not fix the "flat-policy collapse" model behavior. Even with aggressive reward changes, the policy standard deviation remained stuck at `0.8187`.

This `0.8187` value is effectively the upper bound clamp of the current policy log std range (`log_std_max = -0.2` giving `exp(-0.2) ≈ 0.8187`). This means the policy is maximally uncertain and avoids taking decisive positions.

PPO Std / Entropy Tuning v1 is a controlled experiment framework to test whether PPO entropy regularization or stability targets are preventing the policy from becoming confident.

## Experimental Presets

The following presets are configured in `configs/ppo_std_presets.yaml`:

- `current`: Matches current default PPO behavior exactly.
- `low_entropy`: Reduces entropy regularization to lower the incentive for uncertainty.
- `zero_entropy`: Removes the entropy bonus entirely to test if it's keeping std maxed.
- `higher_std_penalty`: Pulls the std towards the stability target more strongly.
- `lower_std_ceiling`: Lowers the hard upper bound of policy std (e.g., max std of `~0.496`).
- `tighter_std_band`: Forces the policy std into a lower, tighter band.
- `combined_std_control`: A practical candidate that lowers entropy, increases penalty, and reduces the ceiling.

## CLI Commands

You can run a PPO std experiment for an asset using the CLI:

```bash
# Run default presets
./venv/bin/python -m src.cli experiment ppo-std --asset btc_usdt

# Run specific presets
./venv/bin/python -m src.cli experiment ppo-std --asset btc_usdt --presets current lower_std_ceiling combined_std_control

# Run a quick test (fewer iterations)
./venv/bin/python -m src.cli experiment ppo-std --asset btc_usdt --quick

# Run all presets
./venv/bin/python -m src.cli experiment ppo-std --asset btc_usdt --all-presets
```

Currently, `--all` (all assets) is disabled for PPO std experiments to keep them focused.

## Output Files

Experiments output to `logs/experiments/ppo_std_tuning/{timestamp}_{asset}/`:

- `{preset}/config.yaml`: The applied config
- `{preset}/checkpoint.pth`: The trained checkpoint
- `{preset}/evaluation/`: Deterministic evaluation outputs
- `{preset}/diagnostics/`: Diagnostic outputs
- `{preset}/walk_forward/`: Walk-forward evaluation
- `{preset}/walk_forward_baselines/`: Walk-forward baseline comparisons
- `summary.csv`, `summary.json`: Aggregated metrics for all presets
- `report.md`: Markdown report of the experiment results

## Interpretation

`policy_std_mean` will indicate if the preset was successful at lowering model uncertainty.
`flat_ratio` will indicate if a more confident model successfully escapes the flat-policy collapse (i.e. takes trades instead of hovering near zero).

A composite score is used to rank the presets. This score heavily rewards beating baselines and penalizes max drawdown and flat ratio.

**Important**: This score is only a research comparison helper. It is not proof of trading profitability.

If the policy remains flat despite successfully lowering standard deviation (which occurred in early tests), the next step is to run **Training Signal Diagnostics** (`experiment training-signal`) to check for Advantage and Gradient collapse.

## Limitations

PPO Std / Entropy Tuning v1 is experimental.
It only tests PPO std/entropy presets.
It does not change the default model unless explicitly selected.
It does not add 5m support.
It does not add paper/live trading.
Experiments need repeated seeds before final model selection.
PPO std tuning experiments are offline research experiments. They do not execute trades and do not prove live trading profitability.
