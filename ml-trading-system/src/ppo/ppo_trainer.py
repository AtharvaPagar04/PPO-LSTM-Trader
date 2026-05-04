import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PPOTrainer:
    def __init__(self, env, model, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2):
        self.env = env
        self.model = model

        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip = clip

    def collect_rollout(self, steps=2048):
        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []

        state = self.env.reset()

        for _ in range(steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            mean, std, value = self.model(state_tensor)

            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _ = self.env.step(action.item())

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())
            dones.append(done)

            state = next_state

            if done:
                state = self.env.reset()

        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "log_probs": np.array(log_probs),
            "values": np.array(values),
            "dones": np.array(dones)
        }

    def compute_gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            next_value = values[t + 1] if t + 1 < len(values) else 0
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

            last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout, epochs=10, batch_size=64):
        states = torch.tensor(rollout["states"], dtype=torch.float32)
        actions = torch.tensor(rollout["actions"], dtype=torch.float32).unsqueeze(-1)
        old_log_probs = torch.tensor(rollout["log_probs"], dtype=torch.float32).unsqueeze(-1)

        advantages, returns = self.compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"]
        )

        advantages = torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)

        for _ in range(epochs):
            for i in range(0, dataset_size, batch_size):
                s = states[i:i+batch_size]
                a = actions[i:i+batch_size]
                old_lp = old_log_probs[i:i+batch_size]
                adv = advantages[i:i+batch_size]
                ret = returns[i:i+batch_size]

                mean, std, value = self.model(s)

                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(a)

                ratio = torch.exp(log_prob - old_lp)

                # PPO loss
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # critic loss
                critic_loss = nn.MSELoss()(value, ret)

                # entropy bonus
                entropy = dist.entropy().mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()