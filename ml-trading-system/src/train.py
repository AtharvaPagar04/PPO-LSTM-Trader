import random
import numpy as np
import torch

from src.env.trading_env import TradingEnv
from src.models.policy import LSTMPolicy
from src.ppo.ppo_trainer import PPOTrainer


def load_data():
    train_X = np.load("data/train_windows.npy")
    train_price = np.load("data/train_price_windows.npy")

    return train_X, train_price
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    print("🚀 Starting PPO Training")
    set_seed(42)

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

    model = LSTMPolicy(input_dim)
    model.train()

    # -------------------------
    # PPO Trainer
    # -------------------------
    trainer = PPOTrainer(env, model)

    best_reward = -float("inf")

    # -------------------------
    # Training loop
    # -------------------------
    ITERATIONS = 300

    for i in range(ITERATIONS):
        rollout = trainer.collect_rollout()
        trainer.update(rollout)

        total_reward = rollout["rewards"].sum()

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(model.state_dict(), "models/best_model.pth")

    print(f"Iteration {i+1} completed | Reward: {total_reward:.2f}")

    # -------------------------
    # Save model
    # -------------------------
    torch.save(model.state_dict(), "models/ppo_lstm.pth")

    print("✅ Training complete. Model saved.")


if __name__ == "__main__":
    main()