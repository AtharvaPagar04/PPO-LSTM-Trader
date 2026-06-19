import numpy as np
import torch

from src.models.policy import LSTMPolicy
from src.ppo.losses import clipped_policy_loss, normalize_advantages
from src.ppo.rollout_buffer import compute_gae


def test_compute_gae_output_shapes():
    rewards = np.array([1.0, 0.5, -0.1], dtype=np.float32)
    values = np.array([0.2, 0.1, 0.0], dtype=np.float32)
    dones = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    advantages, returns = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
    assert advantages.shape == rewards.shape
    assert returns.shape == rewards.shape


def test_advantage_normalization_and_clipped_loss_are_finite():
    advantages = normalize_advantages(torch.tensor([[1.0], [2.0], [3.0]]))
    loss = clipped_policy_loss(
        torch.tensor([[0.1], [0.2], [0.3]]),
        torch.tensor([[0.1], [0.15], [0.25]]),
        advantages,
        clip_ratio=0.2,
    )
    assert torch.isfinite(advantages).all()
    assert torch.isfinite(loss)


def test_model_forward_returns_mean_std_and_value():
    model = LSTMPolicy(input_dim=10)
    x = torch.randn(4, 20, 10)
    mean, std, value = model(x)
    assert mean.shape == (4, 1)
    assert std.shape == (4, 1)
    assert value.shape == (4, 1)
