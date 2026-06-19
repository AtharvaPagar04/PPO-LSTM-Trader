import numpy as np
import pytest
from src.ppo.ppo_trainer import explained_variance
import torch

from src.models.actor import ActorHead

def test_explained_variance():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 2.1, 2.9, 4.1])
    
    ev = explained_variance(y_pred, y_true)
    assert 0.9 < ev <= 1.0
    
    y_true_const = np.array([1.0, 1.0, 1.0, 1.0])
    ev_const = explained_variance(y_pred, y_true_const)
    assert np.isnan(ev_const)


def test_hard_clamp_preserves_std_bounds():
    actor = ActorHead(4, log_std_min=-1.5, log_std_max=-0.2, std_parameterization="hard_clamp")
    x = torch.randn(8, 4)
    _, std, diagnostics = actor(x, return_diagnostics=True)
    log_std = torch.log(std)
    assert torch.all(log_std <= -0.2 + 1e-6)
    assert torch.all(log_std >= -1.5 - 1e-6)
    assert "std_high_saturation_ratio" in diagnostics


def test_smooth_bound_preserves_std_bounds():
    actor = ActorHead(4, log_std_min=-1.5, log_std_max=-0.2, std_parameterization="smooth_bound")
    x = torch.randn(8, 4)
    _, std, _ = actor(x, return_diagnostics=True)
    log_std = torch.log(std)
    assert torch.all(log_std <= -0.2 + 1e-6)
    assert torch.all(log_std >= -1.5 - 1e-6)


def test_smooth_bound_allows_non_zero_gradient():
    actor = ActorHead(4, log_std_min=-1.5, log_std_max=-0.2, std_parameterization="smooth_bound")
    with torch.no_grad():
        actor.actor_std.weight.fill_(1.0)
        actor.actor_std.bias.fill_(5.0)
    x = torch.ones(2, 4, requires_grad=False)
    _, std = actor(x)
    loss = std.mean()
    loss.backward()
    assert actor.actor_std.weight.grad is not None
    assert actor.actor_std.weight.grad.abs().sum().item() > 0.0


def test_hard_clamp_can_zero_gradient_when_saturated():
    actor = ActorHead(4, log_std_min=-1.5, log_std_max=-0.2, std_parameterization="hard_clamp")
    with torch.no_grad():
        actor.actor_std.weight.fill_(1.0)
        actor.actor_std.bias.fill_(5.0)
    x = torch.ones(2, 4, requires_grad=False)
    _, std = actor(x)
    loss = std.mean()
    loss.backward()
    assert actor.actor_std.weight.grad is not None
    assert actor.actor_std.weight.grad.abs().sum().item() == pytest.approx(0.0)
