#!/usr/bin/env python
"""Unit-test the DPO loss and log-prob helpers. CPU only, no checkpoint.

    PYTHONPATH=./ python tools/test_dpo_loss.py

Covers the identity that matters most in practice: an untrained adapter makes
the policy identical to the reference, which must produce exactly
-log(0.5) = 0.6931 and a zero reward margin. If a real run's first step does
not land there, the reference model is wired up wrong -- that is the whole
point of checking it here first, where it costs a second instead of a GPU.
"""
import math
import sys

import torch

from vita.constants import IGNORE_INDEX
from vita.train.dpo_loss import batch_sequence_logps, dpo_loss

FAILURES = []


def check(name, condition, detail=""):
    status = "ok  " if condition else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


print("=== batch_sequence_logps ===")

torch.manual_seed(0)
B, T, V = 2, 6, 32
logits = torch.randn(B, T, V)
labels = torch.randint(0, V, (B, T))

# Mask the first two positions of each sequence, as a prompt would be.
labels[:, :2] = IGNORE_INDEX
lp = batch_sequence_logps(logits, labels)
check("returns one value per sequence", lp.shape == (B,), f"got {tuple(lp.shape)}")
check("is fp32", lp.dtype == torch.float32, f"got {lp.dtype}")
check("log-probs are negative", bool((lp < 0).all()), f"{lp.tolist()}")

# Hand-compute one sequence to confirm the shift and the mask.
shift_logits = logits[0, :-1, :]
shift_labels = labels[0, 1:]
manual = 0.0
for t in range(shift_labels.shape[0]):
    if shift_labels[t] != IGNORE_INDEX:
        manual += torch.log_softmax(shift_logits[t].float(), -1)[shift_labels[t]].item()
check(
    "matches a hand-computed sum",
    abs(manual - lp[0].item()) < 1e-4,
    f"manual {manual:.6f} vs fn {lp[0].item():.6f}",
)

# A fully masked sequence contributes nothing rather than crashing on gather.
all_masked = torch.full((1, T), IGNORE_INDEX)
lp_masked = batch_sequence_logps(logits[:1], all_masked)
check("fully masked sequence gives 0", abs(lp_masked.item()) < 1e-9, f"{lp_masked.item()}")

# Mismatched shapes should be rejected loudly, not silently broadcast.
try:
    batch_sequence_logps(torch.randn(2, 5, V), torch.zeros(2, 7, dtype=torch.long))
    check("rejects mismatched shapes", False, "no error raised")
except ValueError:
    check("rejects mismatched shapes", True)

print("\n=== dpo_loss ===")

pc = torch.tensor([-10.0, -20.0])
pr = torch.tensor([-12.0, -25.0])

# 1. Policy identical to reference -> exactly -log(0.5).
losses, cr, rr = dpo_loss(pc, pr, pc.clone(), pr.clone(), beta=0.1)
expected = -math.log(0.5)
check(
    "policy == ref gives -log(0.5)",
    torch.allclose(losses, torch.full_like(losses, expected), atol=1e-6),
    f"{losses.tolist()} vs {expected:.4f}",
)
check("policy == ref gives zero chosen reward", bool((cr.abs() < 1e-6).all()))
check("policy == ref gives zero margin", bool(((cr - rr).abs() < 1e-6).all()))

# 2. Policy prefers chosen more than reference does -> loss below 0.693.
ref_c, ref_r = torch.tensor([-11.0, -21.0]), torch.tensor([-11.5, -24.0])
losses_better, cr_b, rr_b = dpo_loss(pc, pr, ref_c, ref_r, beta=0.1)
check(
    "preferring chosen lowers the loss",
    bool((losses_better < expected).all()),
    f"{losses_better.tolist()}",
)
check("margin is positive", bool(((cr_b - rr_b) > 0).all()), f"{(cr_b - rr_b).tolist()}")

# 3. The reverse must move the loss the other way.
losses_worse, cr_w, rr_w = dpo_loss(pr, pc, ref_c, ref_r, beta=0.1)
check("preferring rejected raises the loss", bool((losses_worse > expected).all()))
check("margin is negative", bool(((cr_w - rr_w) < 0).all()))

# 4. Larger beta sharpens the same preference.
l_small, _, _ = dpo_loss(pc, pr, ref_c, ref_r, beta=0.05)
l_large, _, _ = dpo_loss(pc, pr, ref_c, ref_r, beta=0.5)
check("larger beta pushes loss further from 0.693", bool((l_large < l_small).all()))

# 5. Label smoothing pulls the loss back toward the midpoint.
l_plain, _, _ = dpo_loss(pc, pr, ref_c, ref_r, beta=0.1, label_smoothing=0.0)
l_smooth, _, _ = dpo_loss(pc, pr, ref_c, ref_r, beta=0.1, label_smoothing=0.3)
check("label smoothing raises a confident loss", bool((l_smooth > l_plain).all()))

# 6. Gradients must reach the policy terms only.
p_c = torch.tensor([-10.0], requires_grad=True)
p_r = torch.tensor([-12.0], requires_grad=True)
r_c = torch.tensor([-11.0], requires_grad=True)
r_r = torch.tensor([-11.5], requires_grad=True)
dpo_loss(p_c, p_r, r_c, r_r, beta=0.1)[0].sum().backward()
check("policy chosen gets gradient", p_c.grad is not None and p_c.grad.abs().item() > 0)
check("policy rejected gets gradient", p_r.grad is not None and p_r.grad.abs().item() > 0)
check("reference gets no gradient", r_c.grad is None and r_r.grad is None)

print("\n=== fp32 vs bf16 (why the cast in batch_sequence_logps matters) ===")

torch.manual_seed(1)
big = torch.randn(1, 200, 152064) * 3
lab = torch.randint(0, 152064, (1, 200))
ref32 = batch_sequence_logps(big, lab).item()
via_bf16 = batch_sequence_logps(big.bfloat16(), lab).item()

# Sum bf16 log-probs without the internal upcast, to show what is avoided.
sl, sb = big.bfloat16()[:, :-1, :], lab[:, 1:]
naive = (
    torch.gather(torch.log_softmax(sl, -1), 2, sb.unsqueeze(2)).squeeze(2).sum().item()
)

print(f"  fp32 reference          : {ref32:.4f}")
print(f"  bf16 input, fp32 inside : {via_bf16:.4f}   error {abs(via_bf16 - ref32):.4f}")
print(f"  bf16 throughout         : {naive:.4f}   error {abs(naive - ref32):.4f}")
check(
    "internal fp32 cast beats bf16 throughout",
    abs(via_bf16 - ref32) < abs(naive - ref32) / 10,
    f"{abs(via_bf16 - ref32):.4f} vs {abs(naive - ref32):.4f}",
)

print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")
