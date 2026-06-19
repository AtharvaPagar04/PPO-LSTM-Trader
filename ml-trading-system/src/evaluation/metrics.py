import numpy as np


def _equity_returns(equity):
    if len(equity) < 2:
        return np.array([], dtype=np.float32)
    return np.diff(equity) / (equity[:-1] + 1e-8)


def sharpe_ratio(returns, periods_per_year=8760):
    if len(returns) < 2:
        return 0.0
    return float(
        np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(periods_per_year)
    )


def sortino_ratio(returns, periods_per_year=8760):
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0:
        return 0.0
    return float(
        np.mean(returns) / (np.std(downside) + 1e-8) * np.sqrt(periods_per_year)
    )


def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-8)
    return float(np.max(dd))


def compute_performance_metrics(trace, periods_per_year=8760):
    equity = trace["equity"]
    returns = _equity_returns(equity)
    pnl = np.asarray(trace["pnl"], dtype=np.float32)
    positions = np.asarray(trace["position"], dtype=np.float32)
    transaction_costs = np.asarray(trace["transaction_cost"], dtype=np.float32)
    drawdown = np.asarray(trace["drawdown"], dtype=np.float32)
    num_steps = int(len(pnl))
    final_equity = float(equity[-1]) if len(equity) else 1.0
    total_return = final_equity - 1.0
    annualized_return = (
        float(final_equity ** (periods_per_year / max(num_steps, 1)) - 1.0)
        if num_steps > 0
        else 0.0
    )
    max_dd = float(np.max(drawdown)) if len(drawdown) else 0.0

    return {
        "final_equity": final_equity,
        "total_return": float(total_return),
        "period_return": float(total_return),
        "annualized_return": annualized_return,
        "sharpe": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "sortino": sortino_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_dd,
        "calmar": float(annualized_return / (max_dd + 1e-8)) if max_dd > 0 else 0.0,
        "win_rate": float(np.mean(pnl > 0)) if num_steps else 0.0,
        "average_position": float(np.mean(positions)) if num_steps else 0.0,
        "turnover": float(np.sum(np.abs(np.diff(np.concatenate([[0.0], positions])))))
        if num_steps
        else 0.0,
        "number_of_steps": num_steps,
        "average_transaction_cost": float(np.mean(transaction_costs))
        if num_steps
        else 0.0,
    }
