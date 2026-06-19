import torch


def normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)


def clipped_policy_loss(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float,
) -> torch.Tensor:
    ratio = torch.exp(log_prob - old_log_prob)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    return -torch.min(unclipped, clipped).mean()


def clipped_value_loss(
    value: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    clip_ratio: float,
) -> torch.Tensor:
    clipped_value = old_value + (value - old_value).clamp(-clip_ratio, clip_ratio)
    unclipped = (value - returns) ** 2
    clipped = (clipped_value - returns) ** 2
    return torch.max(unclipped, clipped).mean()


def std_stability_penalty(std: torch.Tensor, target_std: float = 0.5) -> torch.Tensor:
    return (std.mean() - target_std) ** 2
