# Training Signal Diagnostics

This document explains the internal signal flow of the PPO model and how to interpret the diagnostic output from the `training-signal` experiment.

## The Advantage Flow

In PPO, advantages determine whether a specific action should be made more or less likely. The flow of values in this repository is:

1. **Reward**: Calculated by the `TradingEnv` for each step based on `PnL - DrawdownPenalty - PositionPenalty - TurnoverPenalty`.
2. **Value Estimate**: The critic head (`CriticHead`) predicts the expected return from a given state.
3. **TD Delta**: Computed as `Reward + gamma * NextValue - Value`. This represents the immediate surprise in reward.
4. **GAE Advantage (Raw Advantage)**: The Generalized Advantage Estimate is computed iteratively backwards as an exponentially weighted sum of TD deltas (`gamma * lam`).
5. **Return Target**: Computed as `Raw Advantage + Value Estimate`. This becomes the target for the critic.
6. **Advantage Normalization**: The raw advantages are normalized across the batch `(adv - mean) / std` before being used in the policy loss. This ensures stable gradients regardless of reward scale.
7. **PPO Loss**: The actor is updated using the normalized advantages, clamped by the PPO clip ratio.

### Raw vs. Normalized Advantages

The `advantage_mean` recorded in early experiments was **normalized**, which means it is mathematically forced to be approximately `0.0`. This is perfectly normal and does not indicate signal collapse.

To detect true signal collapse, the `training_trace.csv` records `raw_advantage_std` and `raw_advantage_abs_mean`. If the raw advantage standard deviation is extremely small (e.g., `< 1e-6`), it means the model sees no meaningful difference between any actions or states, indicating true reward or GAE collapse.

## Training Signal Metrics

The `training-signal` experiment generates a summary and a trace with deep PPO metrics:

- **Actor Mean Tracking (`deterministic_action_mean`, `actor_mean_abs_delta_from_prev_iter`)**:
  Measured against a fixed diagnostic batch of 256 states from the training set. If the actor mean does not change across iterations, the policy is stagnant.
  
- **Gradient Norms (`actor_grad_norm`, `critic_grad_norm`, `total_grad_norm`)**:
  Measures the magnitude of the backpropagated gradients. If `actor_grad_norm` is tiny, the actor is not receiving learning signals. If `critic_grad_norm` dominates, the shared encoder may be overfitting to the value function.

- **PPO Update Health (`approx_kl`, `clip_fraction`, `ratio_std`)**:
  - `approx_kl`: Measures how much the policy distribution changes per update. Near `0.0` means no change.
  - `clip_fraction`: The percentage of updates clamped by the PPO clip ratio. Near `0.0` means the updates are very small.

- **Value Function Scale (`returns_std_mean`, `value_loss_mean`, `explained_variance`)**:
  Measures whether the critic is successfully fitting the returns. If `value_error` is massive compared to `returns_std`, the critic is struggling, which destroys advantage quality.

## Interpretation & Next Steps

When reviewing the `signal_summary.json` or `report.md`:

1. **Actor Mean Stagnant (`actor_mean_abs_change < 0.01`) + Healthy Advantages**:
   - The environment provides varied rewards, but the actor isn't moving.
   - **Next Step**: Investigate Actor Learning Rate, Policy Loss Scale, or check if gradients are dying before reaching the actor head.

2. **Raw Advantage Collapsed (`raw_advantage_std < 1e-6`)**:
   - All actions look the same to the model.
   - **Next Step**: Audit Reward Scaling, GAE computation, and episode returns.

3. **PPO Updates Tiny (`approx_kl_mean < 1e-5`, `clip_fraction_mean < 0.01`)**:
   - The policy update step is making no progress despite gradients.
   - **Next Step**: Inspect PPO clip, learning rate, or minibatch size.

4. **Value Scale Problem**:
   - Critic fails to approximate returns.
   - **Next Step**: Audit critic loss, reward normalization, or value clipping.

5. **Healthy Signals but Flat Policy**:
   - The model learns, but determines that "flat" (action=0) is genuinely the best response to the data.
   - **Next Step**: Move to Feature Ablation to find what feature is making the model too scared to trade.

> **Note**: Training-signal diagnostics are offline research tools. They do not execute trades and do not prove live trading profitability.
