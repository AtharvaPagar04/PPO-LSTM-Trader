import numpy as np
from src.env.trading_env import TradingEnv

def test_reward_components_default_zero():
    # setup dummy env
    env = TradingEnv(
        feature_windows=np.zeros((10, 5, 2)),
        price_windows=np.zeros((10, 5, 4)),
    )
    env.reset()
    _, reward, _, info = env.step(0.5)
    
    assert info["exposure_penalty_component"] == 0.0
    assert info["directional_reward_component"] == 0.0
    assert info["volatility_exposure_penalty_component"] == 0.0

def test_exposure_penalty_applied_correctly():
    env = TradingEnv(
        feature_windows=np.zeros((10, 5, 2)),
        price_windows=np.zeros((10, 5, 4)),
        exposure_penalty_coef=0.01
    )
    env.reset()
    _, _, _, info = env.step(0.5)
    assert info["exposure_penalty_component"] == 0.01 * 0.5
    
    env.reset()
    _, _, _, info2 = env.step(-0.5)
    assert info2["exposure_penalty_component"] == 0.01 * 0.5

def test_directional_reward_applied_correctly():
    # Make price drop
    price_windows = np.ones((10, 5, 4))
    price_windows[0, -1, 3] = 100.0
    price_windows[1, -1, 3] = 90.0 # Return is -0.1
    
    env = TradingEnv(
        feature_windows=np.zeros((10, 5, 2)),
        price_windows=price_windows,
        directional_reward_coef=0.05
    )
    env.reset()
    # Correct action (short)
    _, _, _, info = env.step(-0.5)
    assert info["directional_reward_component"] > 0
    assert np.isclose(info["directional_reward_component"], 0.05 * (-1.0) * (-0.1))

def test_volatility_exposure_penalty():
    feature_windows = np.zeros((10, 5, 2))
    feature_windows[0, -1, 1] = 0.2  # high vol
    
    env = TradingEnv(
        feature_windows=feature_windows,
        price_windows=np.ones((10, 5, 4)),
        volatility_exposure_penalty_coef=0.01
    )
    env.reset()
    _, _, _, info = env.step(0.5)
    assert np.isclose(info["volatility_exposure_penalty_component"], 0.01 * 0.5 * 0.2)
