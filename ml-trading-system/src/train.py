import random
import numpy as np
import torch
import os   # 🔥 ADD THIS

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
    ITERATIONS = 300
    best_reward = -float("inf")
    rolling_reward = []

    patience = 200
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
    print(f"✅ Training complete. Best: {model_path}, Final: {model_path.replace('.pth','_final.pth')}")

if __name__ == "__main__":
    main()