"""GRPO advantages and loss.

Separated from the trainer so the maths is unit-testable on CPU without a
checkpoint (tools/test_grpo_loss.py).

GRPO (Shao et al., 2024) drops PPO's value network and gets its baseline from
the group instead: sample G responses per prompt, normalise their rewards
within the group, and use that as the advantage. No critic means no second
7B to hold, which is what makes it tractable here.

The one numerical trap worth knowing about is a group whose rewards are all
equal. Then std is 0 and (r - mean)/std is 0/0 = NaN, which propagates
silently into the gradients. With rule-based rewards this is not an edge
case -- every rollout satisfying (or failing) the same rule is the common
outcome early in training. group_advantages zeroes those groups and counts
them, so the trainer can report the rate: a high one means the reward
function cannot tell the samples apart, which is a data problem masquerading
as a training problem.
"""
from typing import Dict, Tuple

import torch


def group_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, int]:
    """Normalise rewards within each group of `group_size` rollouts.

    Args:
        rewards: (N,) with N = num_prompts * group_size, laid out so that
            each contiguous block of `group_size` belongs to one prompt.
        group_size: rollouts per prompt.
        eps: a group whose reward spread is below this is treated as
            degenerate and given zero advantage.

    Returns:
        advantages: (N,) fp32, zero for degenerate groups.
        n_degenerate: how many groups were zeroed.
    """
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")
    if rewards.numel() % group_size != 0:
        raise ValueError(
            f"{rewards.numel()} rewards is not a whole number of groups of {group_size}"
        )

    grouped = rewards.float().view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)

    # Population std (unbiased=False): with group_size == 1 the unbiased
    # version is NaN rather than 0, which would defeat the guard below.
    std = grouped.std(dim=1, keepdim=True, unbiased=False)

    live = std.squeeze(1) > eps
    advantages = torch.zeros_like(grouped)
    if live.any():
        advantages[live] = (grouped[live] - mean[live]) / std[live]

    return advantages.reshape(-1), int((~live).sum())


def grpo_loss(
    policy_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    beta: float = 0.04,
    clip_eps: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Token-level GRPO objective.

    Args:
        policy_logps:  (N, T) per-token log-probs from the current policy.
        old_logps:     (N, T) from the policy that generated the rollouts.
        ref_logps:     (N, T) from the frozen reference.
        advantages:    (N,) one scalar per sequence, broadcast over tokens.
        completion_mask: (N, T) 1 on generated tokens, 0 on prompt/padding.
        beta: weight on the KL penalty toward the reference.
        clip_eps: PPO-style trust region on the importance ratio.

    Returns:
        loss and a dict of scalars for logging.

    The KL term is the k3 estimator, exp(d) - d - 1 where d = ref - policy.
    It is unbiased and, unlike the plain difference, never negative -- so a
    rising KL in the logs always means real drift rather than sampling noise.

    Averaging is per-token over the whole batch rather than per-sequence,
    which weights long completions more. That is the formulation in the
    paper; the length bias it carries is a property of the objective.
    """
    mask = completion_mask.float()
    n_tokens = mask.sum().clamp(min=1.0)

    # old_logps and ref_logps are fixed quantities: one is a record of the
    # policy that sampled the data, the other a frozen baseline. Detaching
    # here makes that explicit rather than relying on the trainer to have
    # produced them under no_grad.
    old_logps = old_logps.detach()
    ref_logps = ref_logps.detach()

    # Ratio between the current policy and the one that sampled the data.
    log_ratio = policy_logps - old_logps
    ratio = torch.exp(log_ratio)

    adv = advantages.unsqueeze(1)  # (N, 1) broadcasts over tokens
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy_term = -torch.min(unclipped, clipped)

    # k3 KL estimator: exp(d) - d - 1, with d = ref - policy.
    d = ref_logps - policy_logps
    kl = torch.exp(d) - d - 1.0

    per_token = policy_term + beta * kl
    loss = (per_token * mask).sum() / n_tokens

    with torch.no_grad():
        clip_frac = (
            ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float() * mask
        ).sum() / n_tokens
        metrics = {
            "kl": float((kl * mask).sum() / n_tokens),
            "ratio": float((ratio * mask).sum() / n_tokens),
            "clip_frac": float(clip_frac),
            "policy_loss": float((policy_term * mask).sum() / n_tokens),
            "advantage_mean": float(advantages.mean()),
            "advantage_std": float(advantages.std(unbiased=False)),
        }

    return loss, metrics
