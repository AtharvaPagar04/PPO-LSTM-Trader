import numpy as np

from src.evaluation.backtest import run_action_backtest
from src.evaluation.diagnostics import compute_action_bucket_ratios
from src.experiments.action_mapping import _evaluate_scaled_actions


class DummyEnv:
    def __init__(self):
        self.price = np.array(
            [
                [[0.0, 0.0, 0.0, 100.0, 0.0]],
                [[0.0, 0.0, 0.0, 101.0, 0.0]],
                [[0.0, 0.0, 0.0, 102.0, 0.0]],
                [[0.0, 0.0, 0.0, 101.0, 0.0]],
            ],
            dtype=np.float32,
        )
        self.cost = 0.001
        self.reward_scale = 50.0
        self.drawdown_penalty = 0.1
        self.position_penalty = 0.05
        self.action_change_penalty = 0.001
        self.reward_clip = 5.0


def test_compute_action_bucket_ratios_basic_case():
    ratios = compute_action_bucket_ratios(np.array([-0.2, 0.0, 0.2]), 0.1)
    assert ratios["flat_ratio"] == 1 / 3
    assert ratios["long_ratio"] == 1 / 3
    assert ratios["short_ratio"] == 1 / 3


def test_scale_evaluation_clips_actions_correctly():
    env = DummyEnv()
    trace_df = {
        "action": np.array([0.4, -0.6, 0.9]),
    }
    import pandas as pd

    result = _evaluate_scaled_actions(pd.DataFrame(trace_df), 3.0, env)
    assert result["avg_abs_scaled_action"] <= 1.0
    assert result["flat_ratio_010"] <= 1.0


def test_run_action_backtest_uses_scaled_actions_not_training_path():
    env = DummyEnv()
    trace = run_action_backtest(env, np.array([0.2, 0.2, 0.2]))
    assert len(trace["action"]) == 3
    assert np.allclose(trace["position"], trace["action"])
