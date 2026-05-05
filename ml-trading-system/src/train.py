import random
import numpy as np
import torch
import os   # 🔥 ADD THIS

from src.evaluation.backtest import run_backtest
from src.evaluation.baselines import *
from src.evaluation.metrics import *
from src.evaluation.utils import compute_market_returns
from src.evaluation.plot import plot_equity_curves

from src.env.trading_env import TradingEnv
from src.models.policy import LSTMPolicy
from src.ppo.ppo_trainer import PPOTrainer


def load_data():
    data_prefix = os.environ.get("DATA_PREFIX", "btc_usdt")

    train_X = np.load(f"data/processed/{data_prefix}_train_windows.npy")
    train_price = np.load(f"data/processed/{data_prefix}_train_price_windows.npy")

    return train_X, train_price


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    model_path = os.environ.get("MODEL_PATH", "models/best_model.pth")
    data_prefix = os.environ.get("DATA_PREFIX", "btc_usdt")
    print(f"📊 Training on: {data_prefix}")
    print("🚀 Starting PPO Training")
    

    set_seed(42)

    # -------------------------
    # Device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # Load data
    # -------------------------
    train_X, train_price = load_data()
    print("Data loaded:", train_X.shape)

    # -------------------------
    # Create environment
    # -------------------------
    env = TradingEnv(train_X, train_price)

    # -------------------------
    # Model
    # -------------------------
    input_dim = train_X.shape[2]
    model = LSTMPolicy(input_dim).to(device)
    model.train()

    # -------------------------
    # PPO Trainer
    # -------------------------
    trainer = PPOTrainer(env, model)

    # -------------------------
    # Training setup
    # -------------------------
    ITERATIONS = 120
    best_reward = -float("inf")
    rolling_reward = []

    patience = 50
    no_improve = 0

    # -------------------------
    # Training loop
    # -------------------------
    for i in range(ITERATIONS):
        rollout = trainer.collect_rollout()
        trainer.update(rollout)

        total_reward = rollout["rewards"].sum()
        rolling_reward.append(total_reward)

        avg_reward = np.mean(rolling_reward[-10:])
        avg_pos = np.mean(rollout["actions"])

        print(
            f"Iter {i+1} | R: {total_reward:.2f} | AvgR: {avg_reward:.2f} | Pos: {avg_pos:.2f}"
        )

        # -------------------------
        # Save best model
        # -------------------------
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(model.state_dict(), model_path)   
            no_improve = 0
            print("✅ New best model saved")
        else:
            no_improve += 1

        # -------------------------
        # Early stopping
        # -------------------------
        if no_improve > patience:
            print("⛔ Early stopping triggered")
            break

        # optional GPU cleanup
        if device.type == "cuda":
            torch.cuda.empty_cache()
        with torch.no_grad():
            sample_state = torch.tensor(train_X[:64], dtype=torch.float32).to(device)
            _, std, _ = model(sample_state)
            print(f"Std: {std.mean().item():.3f}")


    # -------------------------
    # Save final model
    # -------------------------
    torch.save(model.state_dict(), model_path.replace(".pth", "_final.pth"))
    # print(f"✅ Training complete. Best: {model_path}, Final: {model_path.replace('.pth','_final.pth')}")
    
    # =========================
    # TEST EVALUATION START
    # =========================

    

    print("\n===== TEST EVALUATION =====")

    # -------------------------
    # Load TEST DATA (IMPORTANT)
    # -------------------------
    test_X = np.load(f"data/processed/{data_prefix}_test_windows.npy")
    test_price = np.load(f"data/processed/{data_prefix}_test_price_windows.npy")

    test_env = TradingEnv(test_X, test_price)
    model.eval() 
    # -------------------------
    # Run RL policy
    # -------------------------
    rl_equity, rl_returns, rl_actions = run_backtest(test_env, model)

    # -------------------------
    # Compute MARKET returns (FIXED)
    # -------------------------
    market_returns = compute_market_returns(test_price)

    # -------------------------
    # Baselines (CORRECT)
    # -------------------------
    long_equity = always_long(market_returns)
    short_equity = always_short(market_returns)
    flat_equity = always_flat(market_returns)
    rand_equity = random_policy(market_returns)

    # -------------------------
    # Metrics function
    # -------------------------
    def print_metrics(name, equity, returns):
        print(f"\n{name}")
        print(f"Final Equity: {final_return(equity):.4f}")
        print(f"Sharpe: {sharpe_ratio(returns):.2f}")
        print(f"Max DD: {max_drawdown(equity):.2%}")

    # -------------------------
    # Print metrics
    # -------------------------
    print_metrics("RL Model", rl_equity, rl_returns)
    print_metrics("Always Long", long_equity, market_returns)
    print_metrics("Always Short", short_equity, market_returns)
    print_metrics("Always Flat", flat_equity, market_returns)
    print_metrics("Random", rand_equity, market_returns)

    # -------------------------
    # Plot curves
    # -------------------------
    plot_equity_curves({
        "RL": rl_equity,
        "Long": long_equity,
        "Short": short_equity,
        "Flat": flat_equity,
        "Random": rand_equity
    }, prefix=data_prefix)

    # -------------------------
    # Policy behavior
    # -------------------------
    print("\nPolicy Behavior:")
    print(f"Avg Position: {np.mean(rl_actions):.3f}")
    print(f"Position Std: {np.std(rl_actions):.3f}")

    # -------------------------
    # Reverse test (sanity check)
    # -------------------------
    rev_test_X = test_X[::-1]
    rev_test_price = test_price[::-1]

    rev_env = TradingEnv(rev_test_X, rev_test_price)

    rev_equity_rl, _, _ = run_backtest(rev_env, model)

    print("\nReverse RL Equity:", rev_equity_rl[-1])

    # print("\nReverse Test Equity:", rev_equity[-1])
if __name__ == "__main__":
    main()