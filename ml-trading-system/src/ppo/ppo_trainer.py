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

        # device
        self.device = next(model.parameters()).device

    def collect_rollout(self, steps=1024):
        states, actions, rewards = [], [], []
        log_probs, values, dones = [], [], []

        state = self.env.reset()

        for _ in range(steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                mean, std, value = self.model(state_tensor)
                dist = torch.distributions.Normal(mean, std)

                action = dist.sample()
                action = torch.clamp(action, -1.0, 1.0)

                log_prob = dist.log_prob(action).sum(dim=-1)

                next_state, reward, done, _ = self.env.step(action.item())

                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                log_probs.append(log_prob.detach().cpu().item())
                values.append(value.detach().cpu().item())
                dones.append(done)

                state = next_state if not done else self.env.reset()

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
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

            last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, rollout, epochs=4, batch_size=64):  # 🔥 reduced epochs
        states = torch.tensor(rollout["states"], dtype=torch.float32).to(self.device)
        actions = torch.tensor(rollout["actions"], dtype=torch.float32).unsqueeze(-1).to(self.device)
        old_log_probs = torch.tensor(rollout["log_probs"], dtype=torch.float32).unsqueeze(-1).to(self.device)
        old_values = torch.tensor(rollout["values"], dtype=torch.float32).unsqueeze(-1).to(self.device)

        advantages, returns = self.compute_gae(
            rollout["rewards"],
            rollout["values"],
            rollout["dones"]
        )

        advantages = torch.tensor(advantages, dtype=torch.float32).unsqueeze(-1).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1).to(self.device)

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_size = states.size(0)

        for _ in range(epochs):
            indices = torch.randperm(dataset_size, device=self.device)

            for i in range(0, dataset_size, batch_size):
                idx = indices[i:i + batch_size]

                s = states[idx]
                a = actions[idx]
                old_lp = old_log_probs[idx]
                old_v = old_values[idx]
                adv = advantages[idx]
                ret = returns[idx]

                mean, std, value = self.model(s)
                dist = torch.distributions.Normal(mean, std)

                log_prob = dist.log_prob(a).sum(dim=-1, keepdim=True)
                ratio = torch.exp(log_prob - old_lp)

                # -------------------------
                # Actor Loss
                # -------------------------
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
                actor_loss = -torch.min(surr1, surr2).mean()

                # -------------------------
                # Critic Loss (clipped)
                # -------------------------
                value_clipped = old_v + (value - old_v).clamp(-0.2, 0.2)

                critic_loss_unclipped = (value - ret.detach()) ** 2
                critic_loss_clipped = (value_clipped - ret.detach()) ** 2

                critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped).mean()

                # -------------------------
                # Entropy
                # -------------------------
                entropy = dist.entropy().sum(dim=-1).mean()

                # -------------------------
                # Total Loss
                # -------------------------
                # 🔥 std stabilization
                std_penalty = (std.mean() - 0.5) ** 2

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy + 0.01 * std_penalty

                self.optimizer.zero_grad()
                loss.backward()

                # gradient clipping (important for LSTM)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                self.optimizer.step()