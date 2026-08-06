"""DPO loss and log-probability helpers.

Kept separate from the trainer so the maths can be unit-tested on CPU without
a checkpoint (see tools/test_dpo_loss.py).

Two things here are specific to this codebase rather than to DPO in general:

1. Log-probs are computed in fp32. vita_qwen2.py's custom_forward has
   `logits = logits.float()` commented out, so it returns bf16 logits, and
   summing bf16 log-probs over a sequence loses ~2.9 nats against fp32 on a
   152k vocabulary. DPO takes a difference of differences of those sums and
   multiplies by beta (typically 0.1), so that error swamps the signal.
   Casting to fp32 first brings it to ~0.07.

2. Sequence log-probs are summed, not averaged. Averaging is a known DPO
   variant but changes the objective; the original formulation sums, and the
   length bias it introduces is a property of DPO, not a bug to patch here.
"""
from typing import Tuple

import torch
import torch.nn.functional as F

from vita.constants import IGNORE_INDEX


def batch_sequence_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average: bool = False,
) -> torch.Tensor:
    """Sum the log-probability of each sequence's supervised tokens.

    Args:
        logits: (B, T, V) raw model output. Cast to fp32 internally.
        labels: (B, T) with IGNORE_INDEX on positions that carry no
            supervision (the prompt, image/audio spans, and padding).
        average: divide by the number of supervised tokens. Off by default;
            see the module docstring.

    Returns:
        (B,) tensor of per-sequence log-probabilities, in fp32.
    """
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)[:2]} and labels {tuple(labels.shape)} disagree"
        )

    # Same shift as custom_forward: token t predicts token t+1.
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]

    mask = labels != IGNORE_INDEX
    # gather() cannot take IGNORE_INDEX, so park masked positions on 0 and
    # drop them afterwards.
    safe_labels = labels.masked_fill(~mask, 0)

    logps = torch.log_softmax(logits.float(), dim=-1)
    token_logps = torch.gather(logps, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
    token_logps = token_logps * mask

    summed = token_logps.sum(dim=-1)
    if average:
        summed = summed / mask.sum(dim=-1).clamp(min=1)
    return summed


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct Preference Optimization loss (Rafailov et al., 2023).

    All four inputs are (B,) sequence log-probabilities.

    Returns:
        losses:           (B,) per-pair loss
        chosen_rewards:   (B,) beta * (policy - ref) on the chosen response
        rejected_rewards: (B,) same on the rejected response

    The rewards are the implicit reward DPO optimises, scaled by beta. They
    are the useful diagnostic: their difference should climb above zero as
    training separates the two responses.

    With an untrained adapter the policy equals the reference, so both
    differences are zero, logits is zero, and the loss is exactly -log(0.5)
    = 0.6931. That identity is the sharpest check that the reference model is
    wired up correctly.
    """
    policy_delta = policy_chosen_logps - policy_rejected_logps
    # The reference is a fixed baseline, never a target of optimisation.
    # compute_loss already runs it under no_grad, but detaching here means the
    # loss is correct on its own terms rather than relying on the caller.
    ref_delta = (ref_chosen_logps - ref_rejected_logps).detach()
    logits = policy_delta - ref_delta

    if label_smoothing > 0.0:
        # Conservative DPO: assume the preference label is flipped with
        # probability label_smoothing.
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    else:
        losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards
