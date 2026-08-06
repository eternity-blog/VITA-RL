#!/usr/bin/env python
"""Unit-test the GRPO advantages and loss. CPU only, no checkpoint.

    PYTHONPATH=./ python tools/test_grpo_loss.py

The check that earns its keep is the degenerate group: when every rollout in
a group scores the same, (r - mean) / std is 0/0. With rule-based rewards
that happens constantly, and a NaN there would flow into the gradients and
poison the run while the loss curve still printed numbers.
"""
import math
import sys

import torch

from vita.train.grpo_loss import grpo_loss, group_advantages

FAILURES = []


def check(name, condition, detail=""):
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


print("=== group_advantages ===")

# Normal case: two groups of four, each with spread.
rewards = torch.tensor([0.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0])
adv, degen = group_advantages(rewards, group_size=4)
check("no degenerate groups here", degen == 0, f"got {degen}")
check("one advantage per rollout", adv.shape == (8,), f"got {tuple(adv.shape)}")
for g in range(2):
    chunk = adv[g * 4 : (g + 1) * 4]
    check(f"group {g} mean is 0", abs(chunk.mean().item()) < 1e-5, f"{chunk.mean().item():.2e}")
    check(
        f"group {g} std is 1",
        abs(chunk.std(unbiased=False).item() - 1.0) < 1e-5,
        f"{chunk.std(unbiased=False).item():.6f}",
    )
check(
    "ordering is preserved within a group",
    bool((adv[:4].argsort() == rewards[:4].argsort()).all()),
)

# The case that would otherwise produce NaN.
flat = torch.tensor([1.0, 1.0, 1.0, 1.0])
adv, degen = group_advantages(flat, group_size=4)
check("all-equal group is flagged", degen == 1, f"got {degen}")
check("all-equal group has no NaN", not torch.isnan(adv).any(), f"{adv.tolist()}")
check("all-equal group is zeroed", bool((adv == 0).all()), f"{adv.tolist()}")

# group_size 1: population std is 0, so every group is degenerate. The
# unbiased std would be NaN here, which is why the implementation asks for
# the population one.
adv, degen = group_advantages(torch.tensor([5.0, 7.0]), group_size=1)
check("group_size=1 does not produce NaN", not torch.isnan(adv).any(), f"{adv.tolist()}")
check("group_size=1 gives 2 degenerate groups", degen == 2, f"got {degen}")

# Mixed: one live group, one flat.
mixed = torch.tensor([0.0, 1.0, 0.0, 1.0, 3.0, 3.0, 3.0, 3.0])
adv, degen = group_advantages(mixed, group_size=4)
check("mixed batch counts only the flat group", degen == 1, f"got {degen}")
check("live group still normalised", abs(adv[:4].std(unbiased=False).item() - 1.0) < 1e-5)
check("flat group zeroed", bool((adv[4:] == 0).all()))

# Near-flat but not exactly: below eps should also be treated as degenerate,
# otherwise dividing by 1e-9 explodes the advantage.
near = torch.tensor([1.0, 1.0 + 1e-9, 1.0, 1.0])
adv, degen = group_advantages(near, group_size=4, eps=1e-6)
check("near-flat group is caught by eps", degen == 1, f"got {degen}")
check("near-flat advantage stays bounded", float(adv.abs().max()) < 1e-3)

for bad in (0, -1):
    try:
        group_advantages(rewards, group_size=bad)
        check(f"rejects group_size={bad}", False, "no error")
    except ValueError:
        check(f"rejects group_size={bad}", True)
try:
    group_advantages(torch.zeros(7), group_size=4)
    check("rejects a batch that does not divide", False, "no error")
except ValueError:
    check("rejects a batch that does not divide", True)

print("\n=== grpo_loss ===")

N, T = 4, 6
torch.manual_seed(0)
old = torch.randn(N, T) - 2.0
mask = torch.ones(N, T)
adv = torch.tensor([1.0, -1.0, 0.5, -0.5])

# policy == old == ref: ratio 1, KL 0, loss collapses to -mean(advantage).
loss, m = grpo_loss(old.clone(), old, old.clone(), adv, mask, beta=0.04)
check("ratio is 1 when policy == old", abs(m["ratio"] - 1.0) < 1e-5, f"{m['ratio']:.6f}")
check("KL is 0 when policy == ref", abs(m["kl"]) < 1e-6, f"{m['kl']:.2e}")
check(
    "loss equals -mean(advantage)",
    abs(loss.item() + adv.mean().item()) < 1e-5,
    f"{loss.item():.6f} vs {-adv.mean().item():.6f}",
)
check("nothing is clipped at ratio 1", abs(m["clip_frac"]) < 1e-6)

# KL must be non-negative whichever way the policy moved -- that is the
# point of the k3 estimator over a plain difference.
for shift in (-0.5, -0.1, 0.1, 0.5):
    _, mk = grpo_loss(old + shift, old, old, adv, mask, beta=0.04)
    check(f"KL >= 0 when policy shifts by {shift:+.1f}", mk["kl"] >= 0, f"{mk['kl']:.6f}")

# Raising the log-prob of a positively-advantaged sequence should lower the
# loss; the objective is a maximisation of advantage-weighted likelihood.
pos = old.clone()
pos[0] += 0.1  # sequence 0 has advantage +1
loss_pos, _ = grpo_loss(pos, old, old.clone(), adv, mask, beta=0.0)
check("favouring a good sequence lowers the loss", loss_pos.item() < loss.item())

neg = old.clone()
neg[1] += 0.1  # sequence 1 has advantage -1
loss_neg, _ = grpo_loss(neg, old, old.clone(), adv, mask, beta=0.0)
check("favouring a bad sequence raises the loss", loss_neg.item() > loss.item())

# Clipping should engage once the ratio leaves the trust region.
far = old + math.log(2.0)  # ratio == 2
_, mf = grpo_loss(far, old, old.clone(), adv, mask, beta=0.0, clip_eps=0.2)
check("clip_frac is 1 at ratio 2 with eps 0.2", abs(mf["clip_frac"] - 1.0) < 1e-5)
check("ratio metric reports ~2", abs(mf["ratio"] - 2.0) < 1e-4, f"{mf['ratio']:.4f}")

# Masked-out tokens must not contribute.
half = torch.ones(N, T)
half[:, T // 2 :] = 0
policy_tail = old.clone()
policy_tail[:, T // 2 :] += 5.0  # nonsense in the masked region
loss_masked, _ = grpo_loss(policy_tail, old, old.clone(), adv, half, beta=0.04)
loss_clean, _ = grpo_loss(old.clone(), old, old.clone(), adv, half, beta=0.04)
check(
    "masked tokens are ignored",
    abs(loss_masked.item() - loss_clean.item()) < 1e-5,
    f"{loss_masked.item():.6f} vs {loss_clean.item():.6f}",
)

# Beta must actually control the KL penalty.
drifted = old + 0.3
l0, _ = grpo_loss(drifted, old, old.clone(), adv, mask, beta=0.0)
l1, _ = grpo_loss(drifted, old, old.clone(), adv, mask, beta=0.5)
check("larger beta raises the loss under drift", l1.item() > l0.item())

# Gradients flow to the policy only.
p = (old.clone()).requires_grad_(True)
o = old.clone().requires_grad_(True)
r = old.clone().requires_grad_(True)
grpo_loss(p, o, r, adv, mask, beta=0.04)[0].backward()
check("policy gets gradient", p.grad is not None and p.grad.abs().sum() > 0)
check("old policy gets no gradient", o.grad is None or o.grad.abs().sum() == 0)
check("reference gets no gradient", r.grad is None or r.grad.abs().sum() == 0)

print("\n=== end-to-end: degenerate group contributes no gradient ===")
flat_rewards = torch.ones(4)
adv_flat, _ = group_advantages(flat_rewards, group_size=4)
p2 = old.clone().requires_grad_(True)
grpo_loss(p2, old, old.clone(), adv_flat, mask, beta=0.0)[0].backward()
check("zero advantage means zero gradient", float(p2.grad.abs().sum()) < 1e-9)
check("and no NaN in the gradient", not torch.isnan(p2.grad).any())

print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")
