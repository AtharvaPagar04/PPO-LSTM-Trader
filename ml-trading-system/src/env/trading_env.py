import numpy as np


class TradingEnv:
    def __init__(
        self,
        feature_windows,
        price_windows,
        cost=0.0004,
        max_steps=512,
        drawdown_penalty=0.1,
        position_penalty=0.05,
        action_change_penalty=0.001,
        reward_scale=50.0,
        reward_clip=5.0,
    ):
        self.X = feature_windows
        self.price = price_windows
        self.cost = cost
        self.max_steps = max_steps
        self.drawdown_penalty = drawdown_penalty
        self.position_penalty = position_penalty
        self.action_change_penalty = action_change_penalty
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip
        self.n = len(self.X)
        self.mode = "train"
        self.episode_limit = self.max_steps

    def reset(self, mode="train", start_index=None):
        self.mode = mode
        if start_index is not None:
            self.t = int(start_index)
        elif mode == "eval":
            self.t = 0
        elif self.n <= self.max_steps + 1:
            self.t = 0
        else:
            self.t = np.random.randint(0, self.n - self.max_steps - 1)

        self.steps = 0
        self.episode_limit = None if mode == "eval" else self.max_steps
        self.position = 0.0
        self.equity = 1.0
        self.peak = 1.0
        return self._get_state()

    def _get_state(self):
        return self.X[self.t]

    def step(self, action):
        action = float(np.clip(action, -1.0, 1.0))
        prev_position = self.position
        self.position = float(np.clip(action, -1.0, 1.0))

        if self.t + 1 >= self.n:
            return self._get_state(), 0.0, True, {}

        curr_price = self.price[self.t][-1][3]
        next_price = self.price[self.t + 1][-1][3]
        log_return = np.log((next_price + 1e-8) / (curr_price + 1e-8))
        simple_return = (next_price / (curr_price + 1e-8)) - 1.0

        position_change = abs(self.position - prev_position)
        transaction_cost = position_change * self.cost
        pnl = self.position * log_return - transaction_cost

        self.equity *= np.exp(pnl)
        self.equity = max(self.equity, 1e-8)
        self.peak = max(self.peak, self.equity)
        drawdown = (self.peak - self.equity) / self.peak

        reward = pnl * self.reward_scale
        reward -= self.drawdown_penalty * drawdown
        reward -= self.position_penalty * (self.position ** 2)
        reward -= self.action_change_penalty * position_change
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        self.t += 1
        self.steps += 1

        done = self.t >= self.n - 1
        if self.episode_limit is not None:
            done = done or self.steps >= self.episode_limit

        next_state = self._get_state()
        info = {
            "index": self.t,
            "equity": self.equity,
            "drawdown": drawdown,
            "position": self.position,
            "action": action,
            "pnl": pnl,
            "transaction_cost": transaction_cost,
            "position_change": position_change,
            "log_return": log_return,
            "simple_return": simple_return,
        }
        return next_state, reward, done, info
