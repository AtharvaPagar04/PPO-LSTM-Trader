# Model Reference

## Purpose

This is the technical reference for the implemented LSTM actor-critic PPO model and its reward structure.

## LSTM Actor-Critic Model

The policy class is `src.models.policy.LSTMPolicy`.

It combines:

- `LSTMEncoder`
- shared MLP trunk
- `ActorHead`
- `CriticHead`

Forward outputs:

- actor mean
- actor std
- critic value

## Input Shape

The model consumes one rolling observation window per step:

```text
[batch_size, window_size, num_features]
```

With default config:

- `window_size = 20`
- `num_features` depends on the active feature preset

## Windowing

Feature rows are transformed into overlapping windows in `src.features.pipeline.create_windows()`.

For each asset:

- engineered features are built chronologically
- price rows are aligned to the same timestamps
- overlapping windows are created
- train/test split is chronological
- scaling is fit on train only

Each environment state is one scaled feature window.

## Encoder And Shared Layers

The encoder is a multi-layer LSTM:

- input size = feature dimension
- hidden size = `config["model"]["hidden_size"]`
- number of layers = `config["model"]["lstm_layers"]`
- dropout = `config["model"]["dropout"]`

Only the last LSTM output step is forwarded to the shared trunk.

The shared trunk is:

```text
Linear(hidden_dim, 128)
ReLU
Linear(128, 128)
ReLU
```

## Action Distribution

The actor defines a Gaussian policy over a single continuous action.

Training:

- sample from `Normal(mean, std)`
- clamp sampled action to `[-1, 1]`

Evaluation:

- use the actor mean deterministically
- clamp to `[-1, 1]`

Action semantics:

```text
-1.0 = fully short
 0.0 = flat
+1.0 = fully long
```

## Actor Mean

The actor mean head is a linear layer followed by `tanh`, which naturally bounds the mean to `[-1, 1]`.

This mean is the model's directional and sizing signal.

## Actor Std

The actor std head produces a raw log-std value that is then bounded.

Supported parameterizations:

- `hard_clamp`
- `smooth_bound`

Default bounds from config:

```text
log_std_min = -1.5
log_std_max = -0.2
```

That implies a maximum default std near:

```text
exp(-0.2) ~= 0.8187
```

The model also exposes diagnostics such as:

- raw log-std mean/min/max
- bounded log-std mean/min/max
- high/low saturation ratios

## Critic Value

The critic head maps the shared representation to one scalar value estimate. PPO uses it for:

- TD deltas
- GAE advantages
- return targets
- clipped value loss

## PPO Loss Components

Implemented in `src.ppo.losses` and `src.ppo.ppo_trainer`.

Main components:

- clipped policy loss
- clipped value loss
- entropy bonus
- std stability penalty

Training also uses:

- GAE advantage estimation
- advantage normalization
- minibatch PPO updates
- gradient clipping

Important PPO diagnostics tracked during training include:

- `approx_kl`
- `clip_fraction`
- `policy_std_mean`
- `raw_advantage_std`
- `returns_std`
- `actor_grad_norm`
- `critic_grad_norm`
- `explained_variance`

## Environment Reward Components

The reward is computed in `src.env.trading_env.TradingEnv.step()`.

Base terms:

1. position-weighted log-return PnL
2. transaction cost on position change
3. reward scaling
4. drawdown penalty
5. position penalty
6. action-change penalty
7. optional exposure penalty
8. optional turnover penalty
9. optional directional reward
10. optional volatility-exposure penalty
11. reward clipping

Default environment values from `configs/default.yaml`:

```text
transaction_cost = 0.0004
drawdown_penalty = 0.1
position_penalty = 0.05
action_change_penalty = 0.001
reward_scale = 50.0
reward_clip = 5.0
```

The environment logs component-level diagnostics such as:

- `scaled_pnl_reward`
- `drawdown_penalty_value`
- `position_penalty_value`
- `action_change_penalty_value`
- `transaction_cost_component`
- `directional_reward_component`
- `unclipped_reward`
- `clipped_reward`

## Training And Evaluation Split

The model behaves differently in training and evaluation by design.

Training mode:

- stochastic sampling
- random start index
- capped episodes

Evaluation mode:

- deterministic mean action
- sequential full-period run
- no random reset

This separation is necessary because PPO needs stochastic exploration during optimization, but the research result needs reproducible scoring.

## Reference Summary

The implemented model is technically conventional and internally coherent: rolling time-window input, LSTM encoding, Gaussian actor, scalar critic, PPO optimization, and a cost-aware trading reward. The main limitation of the project is not missing model machinery, but the weakness of the tradable signal available to that machinery.
