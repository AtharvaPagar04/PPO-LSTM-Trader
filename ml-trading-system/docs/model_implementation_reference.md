# Model Implementation Reference

## 1. Project Purpose

The `ml-trading-system` project is an offline RL (Reinforcement Learning) trading research model. It utilizes hourly crypto OHLCV data, engineered technical features, an LSTM-based actor-critic policy, and PPO (Proximal Policy Optimization) training.

The project currently provides a CLI-based pipeline for training, deterministic evaluation, and inference on offline data. It is strictly a research and evaluation pipeline:
- **No live trading** is currently implemented.
- **No paper trading** is available yet.
- **No exchange execution** is built in.

This document serves as a detailed reference for the actual implemented code, highlighting how the data flows, how the environment and reward are structured, and how the model evaluates its actions.

---

## 2. Current Supported Assets and Timeframe

The project normalizes and tracks assets centrally (handled by `src.config.assets.normalize_asset_name`). Currently supported assets are:

- `btc_usdt`
- `eth_usdt`
- `sol_usdt`

The current model assumption is based on **1h (1 hour) candles**. While a future target may involve 5m (5-minute) granularity, the entire current implementation, including feature engineering and default window lengths, is designed and tested for 1h data. Support for other timeframes is not yet implemented.

---

## 3. Data Pipeline

The data flows from raw OHLCV CSVs to processed sequence windows ready for the RL environment. The workflow is fully managed in `src/features/pipeline.py`.

```text
raw OHLCV CSV
↓
feature engineering
↓
rolling windows
↓
train/test split
↓
scaling
↓
processed arrays
↓
training/evaluation/inference
```

**Implementation Details:**
- **Raw Data Loading:** Reads raw data (`resolve_existing_raw_data_path`), ensures datetime parsing, and drops nulls.
- **Feature Engineering:** Adds 10 technical indicators directly to the dataframe (`engineer_features`).
- **Windowing:** `create_windows` creates overlapping sequences of length `window_size`. This happens for both feature arrays and pure price arrays.
- **Splitting:** `split_windows` slices the arrays sequentially using `train_split`. The train data comes strictly before the test data.
- **Scaling:** A `StandardScaler` is fit **only on the training split** (reshaped to 2D) to avoid look-ahead leakage, and then transforms both training and testing splits.
- **Artifacts Saved:** Outputs `_train_windows.npy`, `_test_windows.npy`, `_train_price_windows.npy`, `_test_price_windows.npy`, a scaler `.pkl`, and a `_meta.json` in `data/processed/`.

---

## 4. Engineered Features

The model currently builds **10 features** natively in `src/features/pipeline.py`. All features are strictly backward-looking using rolling windows or shifts, eliminating look-ahead bias at this step.

1. **`log_return`**: `log(close / previous_close)` - Represents step-by-step compounded asset returns.
2. **`volatility_10`**: `rolling(10).std()` of `log_return` - Measures short-term market volatility.
3. **`volatility_20`**: `rolling(20).std()` of `log_return` - Measures medium-term market volatility.
4. **`momentum_5`**: `close.pct_change(5)` - 5-period price momentum.
5. **`momentum_10`**: `close.pct_change(10)` - 10-period price momentum.
6. **`trend`**: `ma_10 - ma_30` (Moving Average differences) - Simple oscillator identifying short-term vs long-term trend direction.
7. **`rsi`**: Standard 14-period Relative Strength Index - Measures overbought/oversold conditions using gain/loss averages.
8. **`body_ratio`**: `(close - open) / ((high - low) + 1e-8)` - Represents how much of the candle's total range is made up by the body.
9. **`range_pct`**: `(high - low) / (close + 1e-8)` - Shows proportional price range.
10. **`vol_z`**: Z-score of volume over a 20-period rolling window: `(volume - rolling_mean(20)) / (rolling_std(20) + 1e-8)`.

---

## 5. Windowing and Scaling

**Window Size:** Controlled by config (default window sizes are typically passed down via config).
**Rolling Windows:** `create_windows` builds an array `[data[i : i + window_size]]` across the length of the time series.
**Train/Test Split:** Uses chronological splitting. The train data comes strictly before the test data.
**StandardScaler:** It reshapes the 3D train windows to 2D, calls `fit_transform()`, and then reshapes back to 3D. The same `scaler` transforms test windows. **No leakage is present here** because `fit()` only sees the `train_split`.
**Metadata:** A `_meta.json` tracks timestamps (start/end for both splits) and matrix shapes.

---

## 6. Trading Environment

The core RL mechanics live in `src/env/trading_env.py`.

- **Observation Structure:** A single step returns `self.X[self.t]`, which corresponds to a full scaled feature window of shape `(window_size, features)`.
- **Action Space:** Continuous action expected in `[-1.0, 1.0]`. The environment strictly enforces this with action clipping.
- **Position Representation:** The continuous action translates directly to portfolio exposure. 1.0 is fully long, -1.0 is fully short, 0.0 is flat.
- **Reset Behavior (Train):** If enough steps remain, the environment randomly samples a starting point to ensure diverse episode trajectories. `episode_limit` is bounded to `max_steps`.
- **Reset Behavior (Eval):** When `mode="eval"`, `self.t` forces to `0` and `episode_limit` is set to `None`. This guarantees deterministic, full-period sequential evaluation.
- **Step Logic:** Looks up the next asset price, calculates the position change from the previous step, applies transaction costs, computes `pnl`, and compounds `equity`.
- **Done Condition:** True when `self.t >= n - 1` (end of data) or `steps >= episode_limit` (for training).
- **Info Dictionary:** Yields full step diagnostics including `index`, `equity`, `drawdown`, `position`, `action`, `pnl`, `transaction_cost`, `position_change`, `log_return`, and `simple_return`.
- **Diagnostic Reward Components:** The environment now also exposes `raw_trading_pnl`, `gross_pnl`, `scaled_pnl_reward`, `drawdown_penalty_value`, `position_penalty_value`, `action_change_penalty_value`, `unclipped_reward`, `clipped_reward`, and `was_clipped` in `info`. These fields are read-only diagnostics and do not change reward behavior.

---

## 7. Reward Function

The reward function strictly implemented in `TradingEnv.step()` balances returns against various risk penalties. 

**Exact Formula Breakdown:**
1. **PnL (Trading Return):** `pnl = position * log_return - transaction_cost`. Transaction cost relies directly on `position_change * self.cost`.
2. **Reward Scale:** `reward = pnl * self.reward_scale` (defaults to 50.0).
3. **Drawdown Penalty:** `reward -= self.drawdown_penalty * drawdown` (defaults to 0.1). Drawdown is computed dynamically as `(peak - equity) / peak`.
4. **Position Size Penalty:** `reward -= self.position_penalty * (self.position ** 2)` (defaults to 0.05). Penalizes taking maximum size; favors smaller or flat positions.
5. **Action Change Penalty:** `reward -= self.action_change_penalty * position_change` (defaults to 0.001). Penalizes high turnover and frequent flipping.
6. **Reward Clipping:** Finally clipped via `np.clip(reward, -self.reward_clip, self.reward_clip)` (defaults to 5.0).

### Reward Design Risks
Based on the code, there are significant diagnostic hypotheses about model weakness tied to this reward design:
- **Penalties Too Strong:** The combination of an explicit transaction cost in PnL *plus* an explicit `action_change_penalty` *plus* a `position_penalty` may aggressively discourage the agent from taking any position. 
- **Model Becoming Too Neutral:** With `position_penalty` penalizing non-zero positions, if the scaled PnL cannot overcome the structural penalties, the optimal mathematical policy is `0.0` (flat).
- **Random Short Episodes:** Due to the random start behavior in training, the agent may not experience full long-term equity compounding trends, skewing the relative weight of the immediate drawdown penalty.
- **Clipping Limitation:** Heavy clipping might obscure massive winning or losing trades, truncating the gradient signal for exceptional trend-following behavior.

---

## 8. Model Architecture

The PPO agent relies on an LSTM actor-critic architecture (`src/models/`).

```text
feature window
↓
LSTM encoder
↓
shared representation
├── actor mean/std
└── critic value
```

**Implementation Details:**
- **`LSTMEncoder`:** A standard `nn.LSTM` with `hidden_dim=128`, `num_layers=2` by default. Returns the final hidden state `output[:, -1, :]`.
- **`LSTMPolicy`:** Glues the encoder, a shared trunk (2-layer MLP with 128 hidden size), and heads. Linear layers are orthogonally initialized.
- **`ActorHead`:** Outputs a `mean` bound by `tanh` (range `[-1.0, 1.0]`). Computes a bounded `std` using clamped `log_std` (`clamp(-1.5, -0.2)` -> `exp`).
- **`CriticHead`:** Simple linear projection to a scalar value.
- **Action Sampling:** Actions are sampled from `Normal(mean, std)` during training and clipped to `[-1.0, 1.0]`. Deterministic inference evaluates purely on the policy mean.

---

## 9. PPO Training Implementation

The Proximal Policy Optimization implementation lives in `src/ppo/`.

- **Rollout Buffer:** Collects `steps=1024` environment transitions. Tracks states, actions, rewards, log_probs, values, and dones.
- **Advantage Calculation:** Generalized Advantage Estimation (GAE) handled via `compute_gae` with `gamma=0.99` and `lam=0.95`. Advantages are normalized at the batch level.
- **Optimizer:** Adam optimizer with `lr=3e-4` and gradient clipping `max_grad_norm=0.5`.
- **Minibatch Logic:** The update loop randomly shuffles the rollout buffer and iterates `epochs=4` times in chunks of `batch_size=64`.
- **Losses:** 
  - `clipped_policy_loss`: Standard PPO surrogate objective with `clip=0.2`.
  - `clipped_value_loss`: Value clipping is active.
  - `entropy`: Encourages exploration using `entropy_coef=0.01`.
  - `std_stability_penalty`: Custom penalty `(std.mean() - 0.5) ** 2` multiplied by `std_penalty_coef=0.01` to stabilize high/exploding variances.
- **Total Loss:** `actor_loss + value_coef * critic_loss - entropy_coef * entropy + std_penalty_coef * std_penalty`.
- **Checkpoints/Logging:** Best models are saved as checkpoints based on `best_reward` in validation, handled directly by `PPOTrainer` wrapping in `src.train.train_asset`.

---

## 10. Training Workflow

Managed primarily by `src/train.py` and invoked via CLI (`src/cli.py`).

**Commands:**
```bash
./venv/bin/python -m src.cli train --asset btc_usdt
./venv/bin/python -m src.cli train --all
```

**Flow:**
- Single-asset training: Initializes the dataset, model, envs, and runs episodes. Checkpoints are periodically evaluated and saved.
- Multi-asset training (`--all`): Iteratively trains each asset sequentially using the same flow.
- Config loading resolves paths, epochs, variables, and seeds.
- Run metadata, configs, and metrics are saved to `logs/runs/`.
- Seed initialization guarantees deterministic paths for random sampling if specified.

---

## 11. Evaluation Workflow

Evaluation uses deterministic, full-period runs starting at index 0 of the test split (`mode="eval"` in the environment).

**Commands:**
```bash
./venv/bin/python -m src.cli evaluate --asset btc_usdt
./venv/bin/python -m src.cli evaluate --all
./venv/bin/python -m src.cli evaluate --asset btc_usdt --walk-forward
./venv/bin/python -m src.cli evaluate --all --walk-forward
./venv/bin/python -m src.cli evaluate --asset btc_usdt --walk-forward --baselines
./venv/bin/python -m src.cli evaluate --all --walk-forward --baselines
```

**Features:**
- **Deterministic Full-Period Eval:** Sweeps the test period linearly to assess full test equity, Sharpe, max drawdown, and PnL. Output files are pushed to `logs/evaluation/`.
- **Walk-Forward v1:** Slices the single held-out test block into `folds` (e.g., 5 folds). The *existing* checkpoint is scored against each chronological fold to gauge robustness. It currently *does not* retrain a new model per fold. Output metrics include fold returns, worst DD, robustness ratios.
- **Baseline Comparison:** Generates comparison logs with `--baselines` ranking the RL agent alongside deterministic benchmarks. Outputs are saved to `logs/walk_forward/`.
- **1h Diagnostics:** `./venv/bin/python -m src.cli diagnose --asset btc_usdt` runs a deterministic full-period trace and saves action statistics, reward decomposition, and summary artifacts to `logs/diagnostics/`.

---

## 12. Baseline Strategies

Implemented in `src/evaluation/baselines.py` to contextualize RL performance.

**Current Baselines:**
- `always_long`: Constant position `1.0`.
- `always_short`: Constant position `-1.0`.
- `always_flat`: Constant position `0.0`.
- `random`: Uniformly sampled random positions between `[-1.0, 1.0]`.
- `buy_and_hold`: Functionally equivalent to `always_long` in the current spot-trading setup, so it is omitted from formal rankings.

**Implementation Logic:**
Each strategy's position vector is simulated identically to the RL agent using `simulate_positions()`. Transaction costs are accurately applied when positions change, using the exact market price array, ensuring fair comparisons. The random baseline is seeded for determinism.

---

## 13. Inference and CLI Prediction

`src/inference.py` wraps the model into a standalone prediction capability.

**Commands:**
```bash
./venv/bin/python -m src.cli predict --asset btc_usdt
./venv/bin/python -m src.cli predict --all
./venv/bin/python -m src.cli predict --assets btc_usdt eth_usdt
./venv/bin/python -m src.cli predict --asset btc_usdt --format json
./venv/bin/python -m src.cli predict --all --save
```

**Flow:**
- Loads the trained PyTorch checkpoint.
- Constructs the very last available observation window from the raw CSV data (via feature engineering and scaling).
- Executes a forward pass, explicitly taking the `mean` of the actor distribution (ignoring `std`).
- Converts the continuous action into a human-readable interpretation (e.g., position thresholds).
- Outputs can be formatted as text/tables or structured `json`. Saved artifacts populate `logs/predictions/`.
- **Note:** This module serves prediction telemetry only and provides no execution instructions.

---

## 14. Current Results Summary

Based on deterministic, full-period offline evaluations generated from current checkpoints (`models/`):

**Deterministic Final Equity (Full Test Period):**
```text
btc_usdt: equity 1.0252, sharpe 0.65, max drawdown 2.98%
eth_usdt: equity 0.9773, sharpe -0.42, max drawdown 8.40%
sol_usdt: equity 0.8983, sharpe -0.86, max drawdown 19.45%
```

**Walk-Forward v1 Results (5 Folds):**
```text
btc_usdt: mean return 0.0053, mean Sharpe 0.18, worst DD 2.49%, positive folds 2/5, robustness 0.40
eth_usdt: mean return -0.0041, mean Sharpe -0.27, worst DD 8.40%, positive folds 3/5, robustness 0.60
sol_usdt: mean return -0.0199, mean Sharpe -1.06, worst DD 9.56%, positive folds 1/5, robustness 0.20
```

**Baseline Comparisons:**
```text
btc_usdt:
RL mean return 0.0053
always_long mean return -0.0228
always_short mean return 0.0596
always_flat mean return 0.0000
RL best folds 0/5

eth_usdt:
RL mean return -0.0041
always_long mean return 0.1252
always_short mean return 0.0614
always_flat mean return 0.0000
RL best folds 0/5

sol_usdt:
RL mean return -0.0199
always_long mean return -0.0311
always_short mean return 0.1409
always_flat mean return 0.0000
RL best folds 0/5
```

*These results are offline research diagnostics and should not be interpreted as live trading profitability.*

---

## 15. Current Model Weaknesses

Based on the explicit code logic and evaluation metrics:

1. **Failure to Beat Baselines:** The RL model does not currently beat simple deterministic baselines consistently across assets.
2. **Weak Performance Variation:** BTC is mildly positive, but ETH and SOL show significant degradation.
3. **Overly Neutral Actions:** Based on the strong structural reward penalties (position penalty, action change penalty), it is highly likely the model is defaulting to near-neutral predictions to avoid bleeding capital via transaction/penalty costs.
4. **Policy Uncertainty:** The custom `std_stability_penalty` attempts to govern standard deviation, but the policy distribution may remain too flat (high uncertainty), leading to noisy action sampling during training.
5. **No Independent Retraining per Fold:** The current walk-forward implementation verifies temporal robustness of *one* checkpoint, but we lack proper retrain-per-fold statistics to prove true generalization power.
6. **No Feature Ablation/Hyperparameter Sweep:** We have not swept PPO parameters or ablated feature sets to find a true optima.
7. **Timeframe Rigidity:** The pipeline exclusively handles 1h data. Features over hourly scales may be insufficient for high-frequency RL edges.

---

## 16. Model Improvement Opportunities

### A. Diagnostics
- action distribution report (min, max, mean, std) across the evaluation fold.
- position exposure report (long/short/flat ratios, average absolute position).
- turnover report.
- reward component logging (pure PnL vs. penalty drag).
- policy std and value estimate tracking.
- fold-wise action behavior analysis.

### B. Feature Work
- feature ablation.
- feature importance/proxy analysis.
- additional trend/momentum features.
- volatility regime features.
- market regime labels.
- avoid leakage rigorously.

### C. Reward Tuning
- reduce position penalty.
- tune drawdown penalty.
- inspect transaction cost impact.
- compare reward clipping levels.
- add reward component metrics.

### D. PPO/Training Tuning
- entropy coefficient.
- learning rate.
- rollout length.
- batch size.
- GAE lambda.
- gamma.
- clip ratio.
- episode length.
- hidden size.

### E. Evaluation Improvements
- repeated seed experiments.
- retrain-per-fold walk-forward.
- experiment comparison table.
- baseline ranking over multiple runs.

### F. Future 5m Plan
- Adapting to 5m granular data should represent an entirely separate model pipeline.
- *Do not reuse the 1h checkpoint directly for 5m paper trading.* Features must be re-calibrated for high-frequency micro-volatility.

---

## 17. Recommended Next Implementation

**1h Model Diagnostics v1**

Before altering the algorithm, add telemetry to deeply inspect what the current policy is doing. 

Proposed future CLI integration:
```bash
./venv/bin/python -m src.cli diagnose --asset btc_usdt
./venv/bin/python -m src.cli diagnose --all
```

Diagnostics should report:
```text
action_mean
action_std_mean
action_min
action_max
long_ratio
short_ratio
flat_ratio
average_abs_position
turnover
reward mean/std
equity change
fold-wise action behavior
```
*Note: This is a design recommendation. It is not currently implemented.*
