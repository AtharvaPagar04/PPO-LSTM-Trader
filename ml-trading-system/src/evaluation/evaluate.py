import numpy as np
import torch

from src.env.trading_env import TradingEnv
from src.models.policy import LSTMPolicy


def load_data():
    X = np.load("data/test_windows.npy")
    price = np.load("data/test_price_windows.npy")
    return X, price


def compute_metrics(equity_curve):
    returns = np.diff(equity_curve) / (equity_curve[:-1] + 1e-8)

    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(8760)

    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    mdd = np.max(drawdown)

    return sharpe, mdd


def evaluate():
    print("📊 Evaluating model...")

    X, price = load_data()

    input_dim = X.shape[2]

    model = LSTMPolicy(input_dim)
    model.load_state_dict(torch.load("models/ppo_lstm.pth"))
    model.eval()

    env = TradingEnv(X, price)

    state = env.reset()

    equity_curve = [1.0]
    actions = []

    while True:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            mean, std, _ = model(state_tensor)

        action = mean.item()  # deterministic policy

        state, reward, done, info = env.step(action)

        equity_curve.append(info["equity"])
        actions.append(action)

        if done:
            break

    equity_curve = np.array(equity_curve)

    sharpe, mdd = compute_metrics(equity_curve)

    print("\n===== RESULTS =====")
    print(f"Final Equity: {equity_curve[-1]:.4f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {mdd:.2%}")
    print(f"Avg Position: {np.mean(actions):.2f}")

    return equity_curve


if __name__ == "__main__":
    evaluate()