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
        self.t = np.random.randint(0, self.n - self.max_steps - 1)
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
        self.position = float(np.clip(action, -1.0, 1.0))
        # -------------------------
        # Price movement
        # -------------------------
        if self.t + 1 >= self.n:
            return self._get_state(), 0.0, True, {}

        curr_price = self.price[self.t][-1][3]
        next_price = self.price[self.t + 1][-1][3]

        ret = np.log((next_price + 1e-8) / (curr_price + 1e-8))

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
        self.equity *= np.exp(pnl)
        self.equity = max(self.equity, 1e-8)

        # -------------------------
        # Drawdown
        # -------------------------
        self.peak = max(self.peak, self.equity)
        drawdown = (self.peak - self.equity) / self.peak

        # -------------------------
        # Reward (STABLE VERSION)
        # -------------------------

        reward = pnl * 50  # strong signal

        # drawdown penalty (important)
        reward -= 0.1 * drawdown

        # position regularization (prevent extreme exposure)
        reward -= 0.05 * (self.position ** 2)

        # transaction cost awareness
        reward -= 0.001 * abs(self.position - prev_position)

        # clip for stability
        reward = np.clip(reward, -5, 5)

        # -------------------------
        # Step forward
        # -------------------------
        self.t += 1
        self.steps += 1

        done = self.steps >= self.max_steps or self.t >= self.n - 2

        next_state = self._get_state()

        info = {
            "equity": self.equity,
            "drawdown": drawdown,
            "position": self.position,
            "pnl": pnl
        }

        return next_state, reward, done, info