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

def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = np.var(y_true)
    return float(np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y)


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
        std_target=0.5,
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
        self.std_target = std_target
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

        raw_advantages, raw_returns = self.compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"]
        )
        
        advantages = normalize_advantages(
            torch.tensor(raw_advantages, dtype=torch.float32).unsqueeze(-1).to(self.device)
        )
        returns = torch.tensor(raw_returns, dtype=torch.float32).unsqueeze(-1).to(
            self.device
        )
        
        next_values = np.append(rollout["values"][1:], 0.0)
        td_deltas = rollout["rewards"] + self.gamma * next_values * (1 - rollout["dones"]) - rollout["values"]
        
        value_error = raw_returns - rollout["values"]
        
        dataset_size = states.size(0)
        
        metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "std_penalty": [],
            "approx_kl": [],
            "clip_fraction": [],
            "ratio_mean": [],
            "ratio_std": [],
            "ratio_min": [],
            "ratio_max": [],
            "actor_mean_mean": [],
            "actor_mean_abs": [],
            "actor_mean_min": [],
            "actor_mean_max": [],
            "policy_std_mean": [],
            "policy_std_min": [],
            "policy_std_max": [],
            "normalized_advantage_mean": [],
            "normalized_advantage_std": [],
            "normalized_advantage_min": [],
            "normalized_advantage_max": [],
            "normalized_advantage_abs_mean": [],
            "actor_grad_norm": [],
            "critic_grad_norm": [],
            "shared_lstm_grad_norm": [],
            "total_grad_norm": [],
        }

        base_metrics = {
            "raw_advantage_mean": float(np.mean(raw_advantages)),
            "raw_advantage_std": float(np.std(raw_advantages)),
            "raw_advantage_min": float(np.min(raw_advantages)),
            "raw_advantage_max": float(np.max(raw_advantages)),
            "raw_advantage_abs_mean": float(np.mean(np.abs(raw_advantages))),
            "returns_mean": float(np.mean(raw_returns)),
            "returns_std": float(np.std(raw_returns)),
            "returns_min": float(np.min(raw_returns)),
            "returns_max": float(np.max(raw_returns)),
            "td_delta_mean": float(np.mean(td_deltas)),
            "td_delta_std": float(np.std(td_deltas)),
            "td_delta_min": float(np.min(td_deltas)),
            "td_delta_max": float(np.max(td_deltas)),
            "td_delta_abs_mean": float(np.mean(np.abs(td_deltas))),
            "reward_batch_mean": float(np.mean(rollout["rewards"])),
            "reward_batch_std": float(np.std(rollout["rewards"])),
            "reward_batch_min": float(np.min(rollout["rewards"])),
            "reward_batch_max": float(np.max(rollout["rewards"])),
            "value_pred_mean": float(np.mean(rollout["values"])),
            "value_pred_std": float(np.std(rollout["values"])),
            "value_pred_min": float(np.min(rollout["values"])),
            "value_pred_max": float(np.max(rollout["values"])),
            "value_error_mean": float(np.mean(value_error)),
            "value_error_std": float(np.std(value_error)),
            "value_error_abs_mean": float(np.mean(np.abs(value_error))),
            "explained_variance": explained_variance(rollout["values"], raw_returns),
        }


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

                mean, std, value, actor_diagnostics = self.model(
                    s, return_diagnostics=True
                )
                dist = torch.distributions.Normal(mean, std)
                log_prob = dist.log_prob(a).sum(dim=-1, keepdim=True)

                actor_loss = clipped_policy_loss(log_prob, old_lp, adv, self.clip)
                critic_loss = clipped_value_loss(value, old_v, ret.detach(), self.clip)
                entropy = dist.entropy().sum(dim=-1).mean()
                std_penalty = std_stability_penalty(std, target_std=self.std_target)

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
                
                def get_grad_norm(model, prefix=""):
                    total_norm = 0.0
                    for name, p in model.named_parameters():
                        if p.grad is not None and name.startswith(prefix):
                            total_norm += p.grad.data.norm(2).item() ** 2
                    return total_norm ** 0.5
                
                metrics["actor_grad_norm"].append(get_grad_norm(self.model, "actor"))
                metrics["critic_grad_norm"].append(get_grad_norm(self.model, "critic"))
                metrics["shared_lstm_grad_norm"].append((get_grad_norm(self.model, "encoder")**2 + get_grad_norm(self.model, "shared")**2) ** 0.5)
                metrics["total_grad_norm"].append(get_grad_norm(self.model))

                with torch.no_grad():
                    ratio = torch.exp(log_prob - old_lp)
                    metrics["approx_kl"].append(((ratio - 1) - torch.log(ratio)).mean().item())
                    metrics["clip_fraction"].append((torch.abs(ratio - 1.0) > self.clip).float().mean().item())
                    metrics["ratio_mean"].append(ratio.mean().item())
                    metrics["ratio_std"].append(ratio.std().item())
                    metrics["ratio_min"].append(ratio.min().item())
                    metrics["ratio_max"].append(ratio.max().item())
                    metrics["policy_loss"].append(actor_loss.item())
                    metrics["value_loss"].append(critic_loss.item())
                    metrics["entropy"].append(entropy.item())
                    metrics["std_penalty"].append(std_penalty.item())
                    metrics["actor_mean_mean"].append(mean.mean().item())
                    metrics["actor_mean_abs"].append(mean.abs().mean().item())
                    metrics["actor_mean_min"].append(mean.min().item())
                    metrics["actor_mean_max"].append(mean.max().item())
                    metrics["policy_std_mean"].append(std.mean().item())
                    metrics["policy_std_min"].append(std.min().item())
                    metrics["policy_std_max"].append(std.max().item())
                    metrics.setdefault("raw_log_std_mean", []).append(
                        actor_diagnostics["raw_log_std_mean"].item()
                    )
                    metrics.setdefault("raw_log_std_min", []).append(
                        actor_diagnostics["raw_log_std_min"].item()
                    )
                    metrics.setdefault("raw_log_std_max", []).append(
                        actor_diagnostics["raw_log_std_max"].item()
                    )
                    metrics.setdefault("log_std_mean", []).append(
                        actor_diagnostics["log_std_mean"].item()
                    )
                    metrics.setdefault("log_std_min", []).append(
                        actor_diagnostics["log_std_min"].item()
                    )
                    metrics.setdefault("log_std_max", []).append(
                        actor_diagnostics["log_std_max"].item()
                    )
                    metrics.setdefault("std_high_saturation_ratio", []).append(
                        actor_diagnostics["std_high_saturation_ratio"].item()
                    )
                    metrics.setdefault("std_low_saturation_ratio", []).append(
                        actor_diagnostics["std_low_saturation_ratio"].item()
                    )
                    metrics["normalized_advantage_mean"].append(adv.mean().item())
                    metrics["normalized_advantage_std"].append(adv.std().item())
                    metrics["normalized_advantage_min"].append(adv.min().item())
                    metrics["normalized_advantage_max"].append(adv.max().item())
                    metrics["normalized_advantage_abs_mean"].append(adv.abs().mean().item())

        res = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
        res.update(base_metrics)
        return res
