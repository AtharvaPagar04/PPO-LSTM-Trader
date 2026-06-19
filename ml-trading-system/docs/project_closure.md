# Project Closure

## Final Status

This repository is closed as a completed offline research and evaluation project.

Final conclusion:

```text
The current approach does not produce a confirmed tradable timing edge.
The model is not suitable for live trading, paper trading, or profitability claims.
The research framework is successful, but the trading approach is not validated.
```

## What Was Built

The project successfully implemented a full offline research pipeline for:

1. raw OHLCV data loading
2. feature engineering
3. rolling-window sequence generation
4. LSTM actor-critic modeling
5. PPO training
6. deterministic evaluation
7. walk-forward evaluation
8. baseline comparison
9. reward and PPO diagnostics
10. feature ablation and seed validation
11. cross-asset feature experiments
12. supervised signal, target, and reduced-preset audits
13. exposure-equivalent baseline comparison

The engineering framework is successful. The final trading hypothesis is not.

## Final Model Verdict

The current model is not suitable as a trading model.

The core issue is structural rather than a single bug:

```text
The signal available in the current feature set is too weak and unstable for PPO
or simple supervised models to convert into reliable trading actions after costs.
```

The correct summary is:

```text
The framework works.
The evaluation works.
The signal is weak.
The trading edge is not proven.
```

## Why This Version Is Closed

The current 1h OHLCV and basic cross-asset return feature setup did not produce a stable timing edge after:

- deterministic full-period evaluation
- walk-forward validation
- simple baseline comparison
- exposure-equivalent baseline comparison
- supervised target and signal validation

The model repeatedly failed the most important test:

```text
If the policy cannot beat constant_signed_mean_action, it is not demonstrating
useful timing beyond static exposure bias.
```

## Main Findings

### RL policy

- the policy did not consistently beat exposure-equivalent baselines
- behavior often reduced to weak static exposure or noisy low-confidence actions
- more PPO/reward-side tuning improved diagnostics but did not create a validated edge

### Features and targets

- some feature groups showed weak statistical signal
- that signal did not survive trading validation after costs
- reduced stable feature presets slightly improved label metrics but not tradable outcomes

### Supervised baselines

- simple supervised models found small predictive effects
- those effects were not strong enough to beat same-exposure static baselines reliably

## What This Project Proves

- an LSTM + PPO offline crypto trading research pipeline can be built cleanly
- deterministic and walk-forward evaluation materially improve research quality
- weak statistical signal is not enough for a tradable strategy
- exposure-equivalent baselines are necessary to detect fake timing edge
- transaction costs can destroy small predictive effects

## What This Project Does Not Prove

- model profitability
- live or paper trading readiness
- stable production alpha
- that PPO is the right next step for this data regime

## Recommended Future Direction

Any future version should start with stronger data and feature research before returning to RL.

Candidate future inputs:

- funding rates
- open interest
- liquidations
- order book imbalance
- taker buy/sell volume
- volume delta
- market regime labels
- volatility regime labels
- higher-timeframe context

Recommended validation order:

1. feature signal audit
2. supervised walk-forward validation
3. same-exposure baseline comparison
4. transaction-cost-aware trading validation
5. only then consider PPO or LSTM again

## Closure Statement

```text
The project successfully built and validated an offline LSTM+PPO crypto trading
research pipeline. However, extensive evaluation showed that the current OHLCV
and cross-asset return features provide only weak predictive signal. The RL
policy and supervised baselines failed to consistently beat exposure-equivalent
static baselines after transaction costs. Therefore, this approach is not
suitable for live or paper trading in its current form. The project is closed
with the conclusion that stronger data sources and feature design are required
before further RL development is justified.
```
