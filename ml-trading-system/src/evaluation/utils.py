import numpy as np

def compute_market_returns(price_windows):
    returns = []

    for i in range(len(price_windows) - 1):
        curr = price_windows[i][-1][3]
        next_ = price_windows[i + 1][-1][3]

        r = (next_ / (curr + 1e-8)) - 1
        returns.append(r)

    return np.array(returns)