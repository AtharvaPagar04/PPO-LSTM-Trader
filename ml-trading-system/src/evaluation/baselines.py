import numpy as np

def always_long(returns):
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    return np.array(equity)


def always_short(returns):
    equity = [1.0]
    for r in returns:
        equity.append(equity[-1] * (1 - r))
    return np.array(equity)


def always_flat(returns):
    return np.ones(len(returns) + 1)


def random_policy(returns):
    equity = [1.0]
    for r in returns:
        action = np.random.choice([-1, 0, 1])
        equity.append(equity[-1] * (1 + action * r))
    return np.array(equity)