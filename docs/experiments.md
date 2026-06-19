# Experiments

## Purpose

This document consolidates the main experiment tracks used in the project. Each section summarizes why the experiment exists, how to run it, what it showed, and how it affected the final conclusion.

## Reward Tuning

Purpose:

- test whether the policy was staying flat because the reward penalties were too restrictive

Command:

```bash
./venv/bin/python -m src.cli experiment reward --asset btc_usdt --quick
```

What it showed:

- reducing penalties changed behavior diagnostics but did not create a reliable trading edge
- transaction-cost-aware validation remained weak

Final interpretation:

- reward shaping was not the root solution
- the model was not being held back mainly by a reward bug

## PPO Std Tuning

Purpose:

- test whether entropy or std constraints were keeping the policy too uncertain or too flat

Command:

```bash
./venv/bin/python -m src.cli experiment ppo-std --asset btc_usdt --quick
```

What it showed:

- the project could inspect and alter policy std behavior
- lowering or reshaping std constraints did not translate into validated trading performance

Final interpretation:

- confidence calibration alone did not create tradable timing

## Training Signal Diagnostics

Purpose:

- inspect whether PPO updates, advantages, and gradients were collapsing internally

Command:

```bash
./venv/bin/python -m src.cli experiment training-signal --asset btc_usdt --quick
```

What it showed:

- the repo can audit raw advantages, normalized advantages, KL, clip fraction, actor drift, and critic fit
- this helped distinguish "no learning signal" from "weak tradable signal"

Final interpretation:

- training diagnostics were useful for debugging, but they did not overturn the weak-strategy conclusion

## Feature Ablation

Purpose:

- compare smaller and alternative feature sets rather than assuming the default set was best

Command:

```bash
./venv/bin/python -m src.cli experiment feature-ablation --asset btc_usdt --quick
```

What it showed:

- some feature subsets were cleaner than the default set
- smaller sets could modestly improve specific diagnostics or label metrics

Final interpretation:

- feature selection mattered, but it still did not produce a reliable post-cost strategy

## Seed Validation

Purpose:

- check whether better-looking preset results survive repeated random seeds

Command:

```bash
./venv/bin/python -m src.cli experiment seed-validation --asset btc_usdt --feature-presets stable_cross_asset_core_v1 stable_cross_asset_core_v2 --quick
```

What it showed:

- candidate presets needed to survive repeated runs, not just one favorable seed

Final interpretation:

- the project correctly treated isolated wins as insufficient evidence

## Objective Calibration

Purpose:

- test alternate objective terms such as exposure penalties, directional reward, and volatility-aware penalties

Command:

```bash
./venv/bin/python -m src.cli experiment objective-calibration --asset btc_usdt --feature-preset price_action_minimal --quick
```

What it showed:

- changing objective emphasis could alter policy behavior
- it still did not create robust wins versus exposure-equivalent baselines

Final interpretation:

- objective redesign did not solve the signal problem

## Signal Audit

Purpose:

- measure whether the available feature sets contain any predictive structure before asking RL to exploit them

Command:

```bash
./venv/bin/python -m src.cli experiment signal-audit --asset btc_usdt --quick
```

What it showed:

- the data was not pure noise
- some features carried weak predictive signal

Final interpretation:

- weak signal exists, but weak signal is not the same as tradable signal

## Target Audit

Purpose:

- test alternative prediction horizons and thresholded targets to see whether label design was hiding usable structure

Command:

```bash
./venv/bin/python -m src.cli experiment target-audit --asset btc_usdt --feature-preset cross_asset_context_v1 --quick
```

What it showed:

- some target formulations produced slightly better balanced accuracy than naive labels

Final interpretation:

- target engineering helped analysis, but not enough to validate trading usefulness

## Feature Signal Audit

Purpose:

- score individual engineered features directly instead of only judging them through PPO outcomes

Command:

```bash
./venv/bin/python -m src.cli experiment feature-signal-audit --asset btc_usdt --feature-preset cross_asset_context_v1 --quick
```

What it showed:

- several cross-asset features had weak but measurable signal
- candidate stable features could be ranked for reduced-preset testing

Final interpretation:

- this was useful for narrowing the feature search space
- it also confirmed that the available signal was modest

## Supervised Signal Strategy

Purpose:

- test whether simpler supervised trading logic could monetize the same signals better than PPO

Command:

```bash
./venv/bin/python -m src.cli experiment supervised-signal-strategy --asset btc_usdt --feature-preset cross_asset_context_v1 --quick
```

What it showed:

- supervised models could sometimes improve label metrics
- those gains still struggled to beat static same-exposure baselines after costs

Final interpretation:

- the issue was broader than PPO
- the available features were not supporting strong cost-aware timing

## Reduced Feature-Preset Validation

Purpose:

- validate stripped-down presets built from the strongest signal-audit candidates

Command:

```bash
./venv/bin/python -m src.cli experiment supervised-signal-strategy --asset btc_usdt --feature-preset stable_cross_asset_core_v1 --quick
./venv/bin/python -m src.cli experiment supervised-signal-strategy --asset btc_usdt --feature-preset stable_cross_asset_core_v2 --quick
```

What it showed:

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

Final interpretation:

- the reduced presets slightly improved label-level prediction, but failed trading validation after costs
- they did not beat `constant_signed_mean_action`

## Experiment Artifact Pattern

Most experiment families write structured outputs under:

```text
logs/experiments/<experiment_type>/{timestamp}_{asset}/
```

Typical artifacts include:

- `audit_manifest.json`
- `summary.csv`
- `summary.json`
- `report.md`
- per-preset training traces
- evaluation outputs
- diagnostics outputs

## Overall Interpretation

Across experiment types, the same pattern kept repeating:

- implementation and diagnostics improved
- some label-level or behavior-level metrics improved
- trading validation after costs did not improve enough

That consistency is why the project was closed at the data-and-signal layer rather than at the PPO implementation layer.
