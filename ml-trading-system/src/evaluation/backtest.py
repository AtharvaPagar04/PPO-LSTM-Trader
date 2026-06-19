import numpy as np
import torch


def run_policy_backtest(env, policy, deterministic_policy=True):
    device = next(policy.parameters()).device
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
    policy.eval()
    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, std, _ = policy(state_t)
            if deterministic_policy:
                action = mean.cpu().numpy()[0][0]
            else:
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
