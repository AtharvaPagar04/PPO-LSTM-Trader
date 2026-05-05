import numpy as np
import torch

def run_backtest(env, policy):
    device = next(policy.parameters()).device  # 🔥 FIX

    state = env.reset()

    equity_curve = [1.0]
    actions = []

    done = False

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            mean, _, _ = policy(state_t)
            action = mean.cpu().numpy()[0][0]

        next_state, reward, done, info = env.step(action)

        equity_curve.append(info["equity"])
        actions.append(info["position"])

        state = next_state

    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-8)

    return equity_curve, returns, actions