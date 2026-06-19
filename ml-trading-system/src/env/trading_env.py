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
        exposure_penalty_coef=0.0,
        turnover_penalty_coef=0.0,
        directional_reward_coef=0.0,
        volatility_exposure_penalty_coef=0.0,
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
        self.exposure_penalty_coef = exposure_penalty_coef
        self.turnover_penalty_coef = turnover_penalty_coef
        self.directional_reward_coef = directional_reward_coef
        self.volatility_exposure_penalty_coef = volatility_exposure_penalty_coef
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

        scaled_pnl_reward = pnl * self.reward_scale
        drawdown_penalty_value = self.drawdown_penalty * drawdown
        position_penalty_value = self.position_penalty * (self.position ** 2)
        action_change_penalty_value = self.action_change_penalty * position_change
        turnover_penalty_value = self.turnover_penalty_coef * position_change
        exposure_penalty_value = self.exposure_penalty_coef * abs(self.position)
        directional_reward_value = self.directional_reward_coef * np.sign(self.position) * simple_return
        
        volatility = 0.0
        # Check if volatility_10 is available in features
        if self.X.shape[2] >= 2:
            volatility = self.X[self.t][-1][1] # Fallback to index 1 which is usually volatility_10
        volatility_exposure_penalty_value = self.volatility_exposure_penalty_coef * abs(self.position) * volatility
        
        unclipped_reward = (
            scaled_pnl_reward
            - drawdown_penalty_value
            - position_penalty_value
            - action_change_penalty_value
            - turnover_penalty_value
            - exposure_penalty_value
            + directional_reward_value
            - volatility_exposure_penalty_value
        )
        reward = float(np.clip(unclipped_reward, -self.reward_clip, self.reward_clip))
        was_clipped = not np.isclose(reward, unclipped_reward)

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
            "raw_trading_pnl": pnl,
            "gross_pnl": self.position * log_return,
            "scaled_pnl_reward": scaled_pnl_reward,
            "drawdown_penalty_value": drawdown_penalty_value,
            "position_penalty_value": position_penalty_value,
            "action_change_penalty_value": action_change_penalty_value,
            "pnl_component": scaled_pnl_reward,
            "transaction_cost_component": transaction_cost * self.reward_scale,
            "drawdown_penalty_component": drawdown_penalty_value,
            "position_penalty_component": position_penalty_value,
            "turnover_penalty_component": action_change_penalty_value + turnover_penalty_value,
            "exposure_penalty_component": exposure_penalty_value,
            "directional_reward_component": directional_reward_value,
            "volatility_exposure_penalty_component": volatility_exposure_penalty_value,
            "total_reward": float(np.clip(unclipped_reward, -self.reward_clip, self.reward_clip)),
            "unclipped_reward": float(unclipped_reward),
            "clipped_reward": reward,
            "was_clipped": bool(was_clipped),
        }
        return next_state, reward, done, info
