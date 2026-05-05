import numpy as np
import torch
import matplotlib.pyplot as plt

from src.env.trading_env import TradingEnv
from src.models.policy import LSTMPolicy


# -------------------------
# Load data
# -------------------------
def load_test_data(prefix="btc_usdt"):
    X = np.load(f"data/processed/{prefix}_test_windows.npy")
    price = np.load(f"data/processed/{prefix}_test_price_windows.npy")
    return X, price


# -------------------------
# Load model
# -------------------------
def load_model(path, input_dim, device):
    model = LSTMPolicy(input_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# -------------------------
# Run PPO policy
# -------------------------
def run_agent(env, model, device):
    state = env.reset()

    equities = [1.0]
    rewards = []
    actions = []

    done = False

    while not done:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            mean, _, _ = model(state_t)
            action = mean.cpu().numpy()[0][0]

        state, reward, done, info = env.step(action)

        equities.append(info["equity"])
        rewards.append(reward)
        actions.append(action)

    return np.array(equities), np.array(rewards), np.array(actions)


# -------------------------
# Baselines
# -------------------------
def run_baseline(price_windows, mode="long"):
    equity = [1.0]

    for i in range(len(price_windows) - 1):
        curr = price_windows[i][-1][3]
        next_p = price_windows[i + 1][-1][3]

        ret = (next_p / (curr + 1e-8)) - 1.0

        if mode == "long":
            pnl = ret
        elif mode == "short":
            pnl = -ret
        elif mode == "random":
            action = np.random.choice([-1, 0, 1])
            pnl = action * ret
        else:
            pnl = 0

        equity.append(equity[-1] * (1 + pnl))

    return np.array(equity)


# -------------------------
# Metrics
# -------------------------
def compute_metrics(equity):
    returns = np.diff(equity) / (equity[:-1] + 1e-8)

    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24)

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    return {
        "final_equity": equity[-1],
        "sharpe": sharpe,
        "max_drawdown": max_dd
    }


# -------------------------
# Plot
# -------------------------
def plot_results(agent_eq, long_eq, short_eq, random_eq, title):
    plt.figure()
    plt.plot(agent_eq, label="PPO Agent")
    plt.plot(long_eq, label="Always Long")
    plt.plot(short_eq, label="Always Short")
    plt.plot(random_eq, label="Random")
    plt.legend()
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Equity")
    plt.show()


# -------------------------
# Main
# -------------------------
def evaluate(prefix, model_path):
    print(f"\n📊 Evaluating on: {prefix}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, price = load_test_data(prefix)
    env = TradingEnv(X, price)

    input_dim = X.shape[2]
    model = load_model(model_path, input_dim, device)

    # Agent
    agent_eq, rewards, actions = run_agent(env, model, device)

    # Baselines
    long_eq = run_baseline(price, "long")
    short_eq = run_baseline(price, "short")
    random_eq = run_baseline(price, "random")

    # Metrics
    print("\n--- METRICS ---")
    print("Agent:", compute_metrics(agent_eq))
    print("Long :", compute_metrics(long_eq))
    print("Short:", compute_metrics(short_eq))
    print("Random:", compute_metrics(random_eq))

    # Plot
    plot_results(agent_eq, long_eq, short_eq, random_eq, prefix)


if __name__ == "__main__":
    evaluate("btc_usdt", "models/btc_model.pth")
    evaluate("ethusdt", "models/eth_model.pth")