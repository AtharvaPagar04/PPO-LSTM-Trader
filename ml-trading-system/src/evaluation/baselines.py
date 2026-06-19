import numpy as np


BASELINE_STRATEGIES = (
    "always_long",
    "always_short",
    "always_flat",
    "random",
)

EXPOSURE_EQUIVALENT_BASELINES = (
    "constant_signed_mean_action",
    "constant_abs_mean_long",
    "constant_abs_mean_short",
)


def compute_market_returns(price_windows):
    curr = price_windows[:-1, -1, 3]
    nxt = price_windows[1:, -1, 3]
    simple_returns = (nxt / (curr + 1e-8)) - 1.0
    log_returns = np.log((nxt + 1e-8) / (curr + 1e-8))
    return log_returns, simple_returns


def simulate_positions(price_windows, positions, transaction_cost):
    log_returns, simple_returns = compute_market_returns(price_windows)
    trace = {
        "step": [],
        "index": [],
        "action": [],
        "position": [],
        "reward": [],
        "pnl": [],
        "transaction_cost": [],
        "market_log_return": [],
        "market_simple_return": [],
        "equity": [1.0],
        "drawdown": [0.0],
    }

    equity = 1.0
    peak = 1.0
    prev_position = 0.0
    for step, (position, log_r, simple_r) in enumerate(
        zip(positions, log_returns, simple_returns)
    ):
        position = float(np.clip(position, -1.0, 1.0))
        position_change = abs(position - prev_position)
        step_cost = position_change * transaction_cost
        pnl = position * log_r - step_cost
        equity *= np.exp(pnl)
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak

        trace["step"].append(step)
        trace["index"].append(step + 1)
        trace["action"].append(position)
        trace["position"].append(position)
        trace["reward"].append(pnl)
        trace["pnl"].append(pnl)
        trace["transaction_cost"].append(step_cost)
        trace["market_log_return"].append(log_r)
        trace["market_simple_return"].append(simple_r)
        trace["equity"].append(equity)
        trace["drawdown"].append(drawdown)
        prev_position = position

    return {key: np.asarray(value) for key, value in trace.items()}


def build_exposure_equivalent_positions(reference_actions):
    reference_actions = np.asarray(reference_actions, dtype=np.float32)
    mean_action = float(np.mean(reference_actions)) if len(reference_actions) else 0.0
    abs_mean_action = (
        float(np.mean(np.abs(reference_actions))) if len(reference_actions) else 0.0
    )
    positions = {
        "constant_signed_mean_action": np.full(
            len(reference_actions), mean_action, dtype=np.float32
        ),
        "constant_abs_mean_long": np.full(
            len(reference_actions), abs_mean_action, dtype=np.float32
        ),
        "constant_abs_mean_short": np.full(
            len(reference_actions), -abs_mean_action, dtype=np.float32
        ),
    }
    return positions


def run_baselines(price_windows, transaction_cost, seed, reference_actions=None):
    steps = len(price_windows) - 1
    rng = np.random.default_rng(seed)
    random_positions = rng.uniform(-1.0, 1.0, size=steps).astype(np.float32)

    traces = {
        "always_long": simulate_positions(
            price_windows, np.ones(steps, dtype=np.float32), transaction_cost
        ),
        "always_short": simulate_positions(
            price_windows, -np.ones(steps, dtype=np.float32), transaction_cost
        ),
        "always_flat": simulate_positions(
            price_windows, np.zeros(steps, dtype=np.float32), transaction_cost
        ),
        "random": simulate_positions(price_windows, random_positions, transaction_cost),
        # In the current spot-style setup, buy-and-hold is equivalent to always-long.
        "buy_and_hold": simulate_positions(
            price_windows, np.ones(steps, dtype=np.float32), transaction_cost
        ),
    }
    if reference_actions is not None:
        for name, positions in build_exposure_equivalent_positions(reference_actions).items():
            traces[name] = simulate_positions(price_windows, positions, transaction_cost)
    return traces
