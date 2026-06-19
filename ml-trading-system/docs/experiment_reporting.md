# Experiment Reporting and Signals

This document outlines the standard artifacts produced by `experiment` commands like `ppo-std` and `reward`.

## Output Structure
Each experiment outputs to `logs/experiments/<type>_tuning/{timestamp}_{asset}/`:

- `audit_manifest.json`: An auditable record of exactly what was run.
- `summary.csv` / `summary.json`: Unified evaluation metrics for all presets.
- `report.md`: Markdown summary comparing the presets.
- `{preset}/training_trace.csv`: Detailed training signal metrics per iteration.
- `{preset}/evaluation/`: Deterministic evaluation run using the BEST checkpoint.
- `{preset}/diagnostics/`: Diagnostic output for the BEST checkpoint.

## Metrics
- **Deterministic Metrics**: (e.g. `deterministic_return`, `deterministic_sharpe`) Computed using `deterministic_full_period` mode. This uses a single continuous pass over the evaluation dataset with actions taken exactly at the mean of the policy distribution (no random sampling).
- **Walk-forward Metrics**: (e.g. `walk_forward_mean_sharpe`) Computed by evaluating the model over sequential rolling folds of the evaluation dataset to estimate out-of-sample robustness.

## Training Trace
`training_trace.csv` records key signals during the PPO update loop:
- `episode_reward`: Total unclipped sum of reward in the iteration.
- `deterministic_action_mean` / `actor_mean_abs_delta_from_prev_iter`: Measures how the actor moves over a fixed diagnostic batch.
- `policy_std_mean`: The average standard deviation output of the policy network.
- `raw_advantage_mean` / `raw_advantage_std`: Raw GAE advantage values before batch normalization.
- `normalized_advantage_mean`: Advantage after `(adv - mean) / std`, ensuring stable update scales.
- `td_delta_std` / `returns_std`: Standard deviation of the underlying value signals.
- `actor_grad_norm` / `critic_grad_norm`: The magnitude of the parameter gradients for the specific network heads.
- `approx_kl` / `clip_fraction`: Indicators of PPO policy shift and bound clipping.
- `value_loss` / `explained_variance`: Measures of critic performance relative to the true returns.
- `early_stop_flag`: Whether this iteration triggered the early stopping condition.

## Important Note
Experiment reports are offline research artifacts. They are used to compare model behavior and do not prove live trading profitability.
