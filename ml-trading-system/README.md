## Current Project Status

This project is now complete as an offline trading research framework.

The final conclusion is:

```text
The LSTM + PPO pipeline works technically, but the current approach does not produce a confirmed tradable edge.
```

The project should be treated as a successful research/evaluation system, not as a deployable trading bot.

## Final Research Verdict

The model is **not suitable for live trading or paper trading in its current form**.

The main reason is not a single implementation bug. The core issue is that 1h OHLCV and basic cross-asset return features provide only weak, unstable predictive signal. That signal was not strong enough for either PPO or simple supervised models to convert into reliable trading actions after costs.

## What Was Achieved

The project successfully implements:

* LSTM actor-critic policy trained with PPO
* continuous long/short action output in `[-1, 1]`
* deterministic evaluation
* walk-forward evaluation
* baseline strategy comparison
* exposure-equivalent baseline checks
* reward and PPO diagnostics
* PPO standard-deviation diagnostics
* feature ablation
* seed validation
* cross-asset feature engineering
* objective calibration
* supervised signal audit
* target/label audit
* supervised signal trading baseline
* reduced feature-preset validation

This makes the repository a complete offline RL trading research pipeline.

## Key Results

### Deterministic RL Evaluation

| Asset      | Final Equity | Sharpe | Max Drawdown |
| ---------- | -----------: | -----: | -----------: |
| `btc_usdt` |       1.0252 |   0.65 |        2.98% |
| `eth_usdt` |       0.9773 |  -0.42 |        8.40% |
| `sol_usdt` |       0.8983 |  -0.86 |       19.45% |

These results are from saved checkpoints on held-out test data. They are not evidence of live profitability.

### Feature Signal Audit

The `cross_asset_context_v1` preset showed weak measurable signal:

* rows audited: `37,927`
* timestamp range: `2022-01-05` to `2026-05-04`
* best feature-level signals included:

  * `eth_return_72 @ h24`
  * `eth_return_24 @ h24`
  * `log_return @ h1`
  * `market_avg_return_24 @ h24`
  * `trend @ h24`

Example signal metrics:

```text
eth_return_72 @ h24 Spearman: ~0.068
eth_return_24 @ h24 AUC: ~0.544
log_return @ h1 AUC: ~0.541
```

This means the features are not pure noise, but the signal is weak.

### Reduced Feature Preset Validation

Reduced presets were tested using the strongest feature candidates.

| Preset                       | Best WF Balanced Accuracy | Trading WF Return | Trading WF Sharpe | Beat Constant Signed Mean |
| ---------------------------- | ------------------------: | ----------------: | ----------------: | ------------------------: |
| `stable_cross_asset_core_v1` |                    0.5281 |            -7.54% |             -2.10 |                       0/5 |
| `stable_cross_asset_core_v2` |                    0.5272 |            -7.83% |             -2.18 |                       0/5 |

The reduced presets slightly improved label-level balanced accuracy, but failed trading validation.

## What Failed

The current approach failed to produce a robust trading strategy.

Main failures:

* PPO did not learn a reliable timing edge.
* Reward tuning and objective calibration did not fix the issue.
* Cross-asset features showed only weak predictive signal.
* Supervised models found small label-level signal, but it did not survive trading validation.
* Reduced feature sets improved balanced accuracy slightly but lost money after costs.
* Strategies failed exposure-equivalent baseline checks.
* Transaction costs and turnover destroyed small statistical edges.

The most important failed metric is:

```text
beat_constant_signed_mean_action < 3/5
```

If a model cannot beat a constant strategy with the same average signed exposure, it is not proving useful market timing.

## Reason For Failure

The limiting factor is the feature/data signal, not the PPO implementation.

The current feature set is based mostly on:

* 1h OHLCV features
* BTC technical indicators
* ETH/SOL return context
* simple market-average and relative-strength features

These features contain weak statistical signal, but not enough stable, cost-aware signal for trading.

In short:

```text
The model is too complex for the amount of reliable signal available.
```

More PPO tuning, reward shaping, or action mapping is unlikely to solve this version.

## Final Conclusion

This project is closed as:

```text
Successful research framework.
Unsuccessful trading strategy.
```

The system is useful for demonstrating RL trading infrastructure, diagnostics, walk-forward evaluation, and honest model validation.

It should not be used for live trading, paper trading, or profitability claims.

## Future Work

A future version should not begin with PPO. It should first improve the data and feature signal.

Possible future data sources:

* funding rates
* open interest
* liquidation data
* order-book imbalance
* taker buy/sell volume
* volume delta
* volatility regime labels
* higher-timeframe trend context
* BTC dominance or broader market regime features

Recommended future workflow:

1. Build stronger features.
2. Run feature-signal audit.
3. Validate with simple supervised models.
4. Test against transaction costs and exposure-equivalent baselines.
5. Only then consider PPO or LSTM-based RL again.
