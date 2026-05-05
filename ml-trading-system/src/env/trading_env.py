import numpy as np


class TradingEnv:
    def __init__(self, feature_windows, price_windows, cost=0.0004):
        self.X = feature_windows
        self.price = price_windows
        self.cost = cost

        self.n = len(self.X)

        # ✅ control episode length (important for PPO)
        self.max_steps = 512

    def reset(self):
        # ✅ start from a safe range to allow full episode
        self.t = np.random.randint(0, self.n - 2000)
        self.steps = 0

        self.position = 0.0
        self.equity = 1.0
        self.peak = 1.0

        return self._get_state()

    def _get_state(self):
        return self.X[self.t]

    def step(self, action):
        # -------------------------
        # Action handling
        # -------------------------
        action = float(np.clip(action, -1.0, 1.0))

        prev_position = self.position

        # ✅ prevent extreme exposure
        self.position = float(np.clip(action, -0.8, 0.8))

        # -------------------------
        # Price movement
        # -------------------------
        close = self.price[self.t]
        curr_price = close[-1][3]
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
        # Reward (PHASE 3 FINAL TUNED)
        # -------------------------

        reward = pnl * 7

        # risk penalty
        reward -= 0.05 * drawdown

        # position control
        reward -= 0.02 * (self.position ** 2)

        # smooth transitions
        reward -= 0.001 * (self.position - prev_position) ** 2

        # neutrality bias
        reward += 0.005 * (1 - abs(self.position))

        # 🔥 discourage directional bias (existing)
        reward -= 0.01 * self.position

        # 🔥 stronger correct direction reward
        reward += 0.1 * self.position * ret * 10

        # 🔥 NEW: penalize constant bias behavior
        reward -= 0.02 * abs(self.position + 0.5)

        # 🔥 NEW: encourage dynamic decisions
        reward += 0.02 * abs(self.position - prev_position)

        # stabilize
        reward = np.clip(reward, -1, 1)

        # -------------------------
        # Step forward
        # -------------------------
        self.t += 1
        self.steps += 1

        done = self.t >= self.n - 2 or self.steps >= self.max_steps

        next_state = self._get_state()

        info = {
            "equity": self.equity,
            "drawdown": drawdown,
            "position": self.position,
            "pnl": pnl
        }

        return next_state, reward, done, info