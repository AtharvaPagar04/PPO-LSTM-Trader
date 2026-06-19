import numpy as np
import torch


def _simulate_env_trace(env, actions):
    trace = {
        "step": [],
        "index": [],
        "action": [],
        "position": [],
        "reward": [],
        "pnl": [],
        "transaction_cost": [],
        "market_log_return": [],
        "market_simple_return": [],
        "equity": [1.0],
        "drawdown": [0.0],
    }
    position = 0.0
    equity = 1.0
    peak = 1.0

    for step, raw_action in enumerate(actions):
        action = float(np.clip(raw_action, -1.0, 1.0))
        prev_position = position
        position = action

        curr_price = env.price[step][-1][3]
        next_price = env.price[step + 1][-1][3]
        log_return = np.log((next_price + 1e-8) / (curr_price + 1e-8))
        simple_return = (next_price / (curr_price + 1e-8)) - 1.0
        position_change = abs(position - prev_position)
        transaction_cost = position_change * env.cost
        pnl = position * log_return - transaction_cost

        equity *= np.exp(pnl)
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak
        reward = pnl * env.reward_scale
        reward -= env.drawdown_penalty * drawdown
        reward -= env.position_penalty * (position ** 2)
        reward -= env.action_change_penalty * position_change
        reward = float(np.clip(reward, -env.reward_clip, env.reward_clip))

        trace["step"].append(step)
        trace["index"].append(step + 1)
        trace["action"].append(action)
        trace["position"].append(position)
        trace["reward"].append(reward)
        trace["pnl"].append(float(pnl))
        trace["transaction_cost"].append(float(transaction_cost))
        trace["market_log_return"].append(float(log_return))
        trace["market_simple_return"].append(float(simple_return))
        trace["equity"].append(float(equity))
        trace["drawdown"].append(float(drawdown))

    return {key: np.asarray(value) for key, value in trace.items()}


def run_policy_backtest(env, policy, deterministic_policy=True):
    device = next(policy.parameters()).device
    policy.eval()

    if deterministic_policy:
        states = torch.tensor(env.X[:-1], dtype=torch.float32).to(device)
        with torch.no_grad():
            mean, _, _ = policy(states)
        return _simulate_env_trace(env, mean.squeeze(-1).cpu().numpy())

    state = env.reset(mode="eval")
    trace = {
        "step": [],
        "index": [],
        "action": [],
        "position": [],
        "reward": [],
        "pnl": [],
        "transaction_cost": [],
        "market_log_return": [],
        "market_simple_return": [],
        "equity": [1.0],
        "drawdown": [0.0],
    }
    done = False
    step = 0
    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, std, _ = policy(state_t)
            action = torch.distributions.Normal(mean, std).sample().cpu().numpy()[0][0]

        next_state, reward, done, info = env.step(action)
        trace["step"].append(step)
        trace["index"].append(int(info["index"]))
        trace["action"].append(float(action))
        trace["position"].append(float(info["position"]))
        trace["reward"].append(float(reward))
        trace["pnl"].append(float(info["pnl"]))
        trace["transaction_cost"].append(float(info["transaction_cost"]))
        trace["market_log_return"].append(float(info["log_return"]))
        trace["market_simple_return"].append(float(info["simple_return"]))
        trace["equity"].append(float(info["equity"]))
        trace["drawdown"].append(float(info["drawdown"]))
        state = next_state
        step += 1
    return {key: np.asarray(value) for key, value in trace.items()}
