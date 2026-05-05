import numpy as np

def sharpe_ratio(returns):
    if len(returns) < 2:
        return 0
    return np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(8760)


def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    return np.max(dd)


def final_return(equity):
    return equity[-1]