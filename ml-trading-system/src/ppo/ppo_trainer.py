import numpy as np
import torch
import torch.optim as optim

from src.ppo.losses import (
    clipped_policy_loss,
    clipped_value_loss,
    normalize_advantages,
    std_stability_penalty,
)
from src.ppo.rollout_buffer import compute_gae


class PPOTrainer:
    def __init__(
        self,
        env,
        model,
        lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        std_penalty_coef=0.01,
        max_grad_norm=0.5,
    ):
        self.env = env
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.std_penalty_coef = std_penalty_coef
        self.max_grad_norm = max_grad_norm
        self.device = next(model.parameters()).device

    def collect_rollout(self, steps=1024):
        states, actions, rewards = [], [], []
        log_probs, values, dones = [], [], []
        state = self.env.reset(mode="train")

        for _ in range(steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(
                self.device
            )
            with torch.no_grad():
                mean, std, value = self.model(state_tensor)
                dist = torch.distributions.Normal(mean, std)
                action = torch.clamp(dist.sample(), -1.0, 1.0)
                log_prob = dist.log_prob(action).sum(dim=-1)
                next_state, reward, done, _ = self.env.step(action.item())

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.detach().cpu().item())
            values.append(value.detach().cpu().item())
            dones.append(done)

            state = next_state if not done else self.env.reset(mode="train")

        return {
            "states": np.array(states),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "log_probs": np.array(log_probs),
            "values": np.array(values),
            "dones": np.array(dones),
        }

    def compute_gae(self, rewards, values, dones):
        return compute_gae(rewards, values, dones, self.gamma, self.lam)

    def update(self, rollout, epochs=4, batch_size=64):
        states = torch.tensor(rollout["states"], dtype=torch.float32).to(self.device)
        actions = (
            torch.tensor(rollout["actions"], dtype=torch.float32)
            .unsqueeze(-1)
            .to(self.device)
        )
        old_log_probs = (
            torch.tensor(rollout["log_probs"], dtype=torch.float32)
            .unsqueeze(-1)
            .to(self.device)
        )
        old_values = (
            torch.tensor(rollout["values"], dtype=torch.float32)
            .unsqueeze(-1)
            .to(self.device)
        )

        advantages, returns = self.compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"]
        )
        advantages = normalize_advantages(
            torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1).to(self.device)
        )
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1).to(
            self.device
        )

        dataset_size = states.size(0)
        for _ in range(epochs):
            indices = torch.randperm(dataset_size, device=self.device)
            for i in range(0, dataset_size, batch_size):
                idx = indices[i : i + batch_size]
                s = states[idx]
                a = actions[idx]
                old_lp = old_log_probs[idx]
                old_v = old_values[idx]
                adv = advantages[idx]
                ret = returns[idx]

                mean, std, value = self.model(s)
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(a).sum(dim=-1, keepdim=True)

                actor_loss = clipped_policy_loss(log_prob, old_lp, adv, self.clip)
                critic_loss = clipped_value_loss(value, old_v, ret.detach(), self.clip)
                entropy = dist.entropy().sum(dim=-1).mean()
                std_penalty = std_stability_penalty(std)

                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                    + self.std_penalty_coef * std_penalty
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
