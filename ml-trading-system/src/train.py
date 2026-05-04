
import numpy as np
import torch

from src.env.trading_env import TradingEnv
from src.models.policy import LSTMPolicy
from src.ppo.ppo_trainer import PPOTrainer


def load_data():
    train_X = np.load("data/train_windows.npy")
    train_price = np.load("data/train_price_windows.npy")

    return train_X, train_price


def main():
    print("🚀 Starting PPO Training")

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

    # -------------------------
    # Training loop
    # -------------------------
    ITERATIONS = 200

    for i in range(ITERATIONS):
        rollout = trainer.collect_rollout()

        trainer.update(rollout)

        print(f"Iteration {i+1} completed")

    # -------------------------
    # Save model
    # -------------------------
    torch.save(model.state_dict(), "models/ppo_lstm.pth")

    print("✅ Training complete. Model saved.")


if __name__ == "__main__":
    main()