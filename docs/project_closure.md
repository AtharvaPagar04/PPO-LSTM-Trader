# Project Closure

## Final Verdict

This repository is closed as an offline research project.

Final classification:

```text
Successful research framework.
Unsuccessful trading strategy.
```

Final conclusion:

```text
The LSTM + PPO pipeline works technically, but the current approach does not
produce a confirmed tradable edge.
```

## What Was Achieved

The project successfully delivered:

- a coherent feature-engineering pipeline for 1h crypto data
- rolling-window dataset generation and scaling
- a continuous-position trading environment
- an LSTM actor-critic PPO implementation
- deterministic full-period evaluation
- walk-forward evaluation
- baseline and exposure-equivalent baseline comparison
- model diagnostics and reward decomposition
- experiment workflows for reward, std, feature, signal, and target studies

The engineering and research framework is usable. The trading hypothesis is what failed.

## What Failed

The project did not show a stable, cost-aware timing edge.

Main failures:

- performance was inconsistent across assets
- ETH and SOL were negative in deterministic evaluation
- walk-forward robustness was weak
- exposure-equivalent baselines were not beaten reliably
- reduced feature presets improved labels slightly but still failed trading validation
- more PPO-side tuning did not change the core outcome

## Evidence From Results

Deterministic saved-checkpoint results:

| Asset | Final Equity | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: |
| `btc_usdt` | 1.0252 | 0.65 | 2.98% |
| `eth_usdt` | 0.9773 | -0.42 | 8.40% |
| `sol_usdt` | 0.8983 | -0.86 | 19.45% |

Reduced feature preset validation:

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

The key failed metric was:

```text
beat_constant_signed_mean_action < 3/5
```

If the model cannot beat a constant strategy with the same average signed exposure, it has not proven useful timing.

## Reason For Failure

The limiting factor is feature/data signal, not the PPO implementation.

The current 1h OHLCV and basic cross-asset return features contain weak statistical signal, but not enough stable, cost-aware signal for trading.

That conclusion is supported by the full body of work:

- deterministic RL evaluation was mixed to negative
- walk-forward results were not robust
- diagnostics showed weak or overly cautious behavior
- supervised audits found only small predictive effects
- reduced presets improved labels slightly but failed trading checks after costs

The central problem is not that PPO was broken. The central problem is that the available inputs do not provide enough durable edge for the model to monetize after friction.

## Why This Approach Should Not Be Used For Trading

This version should not be used for live trading or paper trading because it has not demonstrated:

- reliable timing edge
- consistency across assets
- post-cost robustness
- superiority over same-exposure static baselines
- evidence that increased complexity is being rewarded by better decisions

A technically correct RL pipeline is still unsuitable for trading if the data signal is too weak.

## Why More Tuning Is Not The Right Next Step

This version is unlikely to be rescued by more work on the same inputs, including:

- PPO tuning
- reward shaping
- action mapping changes
- larger LSTM model
- more training on the same features

Those changes may alter behavior, but the experiments already indicate they do not solve the underlying signal problem.

## What A Future Version Would Require

If the project is resumed, the order of work should change:

1. Improve data and features first.
2. Validate predictive signal with simpler supervised models.
3. Require cost-aware walk-forward wins against exposure-equivalent baselines.
4. Only then revisit RL.

Promising future inputs would be richer market structure features such as funding, open interest, liquidations, order-flow, taker imbalance, or stronger regime signals.

## Closure Summary

This repository should be understood as a finished research framework that answered its main question honestly. It showed that the current 1h OHLCV plus basic cross-asset feature set does not support a confirmed tradable edge in this PPO-LSTM setup. The right conclusion is to stop tuning this version and treat future work as a new data-and-signal problem, not a PPO-implementation problem.
