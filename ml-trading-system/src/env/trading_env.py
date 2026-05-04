import numpy as np


class TradingEnv:
    def __init__(self, feature_windows, price_windows, cost=0.0001):
        self.X = feature_windows
        self.price = price_windows
        self.cost = cost

        self.n = len(self.X)

    def reset(self):
        self.t = 0
        self.position = 0.0  # continuous [-1, 1]
        self.equity = 1.0
        self.peak = 1.0

        return self._get_state()

    def _get_state(self):
        return self.X[self.t]  # (seq_len, features)

    def step(self, action):
        # clamp action to [-1, 1]
        action = float(np.clip(action, -1.0, 1.0))

        prev_position = self.position
        self.position = action

        # -------------------------
        # Compute return
        # -------------------------
        close = self.price[self.t]
        curr_price = close[-1][3]  # close price at t
        next_price = self.price[self.t + 1][-1][3]

        ret = (next_price / (curr_price + 1e-8)) - 1.0

        # -------------------------
        # Transaction cost
        # -------------------------
        position_change = abs(self.position - prev_position)
        cost = position_change * self.cost

        # -------------------------
        # PnL
        # -------------------------
        pnl = self.position * ret - cost

        # -------------------------
        # Equity update
        # -------------------------
        self.equity *= (1 + pnl)
        self.equity = max(self.equity, 1e-8)

        # -------------------------
        # Drawdown
        # -------------------------
        self.peak = max(self.peak, self.equity)
        drawdown = (self.peak - self.equity) / self.peak

        # -------------------------
        # Reward (final tuned version)
        # -------------------------
    

        # risk control
        reward = pnl * 100

        reward -= 0.1 * drawdown
        reward -= 0.012 * (self.position ** 2)
        reward -= 0.002 * position_change

        reward = np.clip(reward, -10, 10)

        # -------------------------
        # Step forward
        # -------------------------
        self.t += 1
        done = self.t >= self.n - 2

        next_state = self._get_state()

        info = {
            "equity": self.equity,
            "drawdown": drawdown,
            "position": self.position
        }

        return next_state, reward, done, info