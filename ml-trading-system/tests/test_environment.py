import numpy as np

from src.env.trading_env import TradingEnv


def make_env():
    x = np.random.randn(30, 20, 10).astype(np.float32)
    prices = np.ones((30, 20, 5), dtype=np.float32)
    prices[:, :, 3] = np.linspace(100.0, 130.0, 30).reshape(-1, 1)
    return TradingEnv(x, prices, cost=0.001, max_steps=5)


def test_reset_returns_expected_observation_shape():
    env = make_env()
    state = env.reset(mode="train")
    assert state.shape == (20, 10)


def test_step_returns_standard_rl_tuple_and_clips_actions():
    env = make_env()
    env.reset(mode="train")
    next_state, reward, done, info = env.step(3.0)
    assert next_state.shape == (20, 10)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert info["position"] == 1.0


def test_transaction_cost_applies_when_position_changes():
    env = make_env()
    env.reset(mode="eval")
    _, _, _, info1 = env.step(0.0)
    _, _, _, info2 = env.step(1.0)
    assert info1["transaction_cost"] == 0.0
    assert info2["transaction_cost"] > 0.0


def test_eval_reset_starts_from_beginning():
    env = make_env()
    env.reset(mode="eval")
    assert env.t == 0
