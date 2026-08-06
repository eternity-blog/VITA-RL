#!/usr/bin/env python
"""Verify that deduplicated image encoding is bit-identical. CPU, no weights.

    PYTHONPATH=./ python tools/test_image_dedup.py

When several sequences in a batch look at the same picture -- DPO's
chosen/rejected pair, later GRPO's rollout group -- vita_arch would encode
that picture once per sequence. The vision tower is deterministic, so
encoding one copy and repeating the features gives exactly the same numbers
for a fraction of the cost.

"Exactly" is the claim under test. An approximation would be a silent
correctness bug: the features feed straight into the LLM, and a small drift
would look like ordinary training noise. So these checks assert bitwise
equality with torch.equal, not allclose.
"""
import sys

import torch
import torch.nn as nn

from vita.model.vita_arch import VITAMetaForCausalLM

FAILURES = []


def check(name, condition, detail=""):
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


class StubTower(nn.Module):
    """Stands in for InternViT: deterministic, and counts what it sees."""

    def __init__(self, dim=8, tokens=4):
        super().__init__()
        self.dim, self.tokens = dim, tokens
        self.seen = 0

    def forward(self, images):
        self.seen += images.shape[0]
        # Content-dependent so a wrong tile order cannot pass unnoticed.
        base = images.flatten(1).sum(dim=1, keepdim=True)
        return base.unsqueeze(1).expand(-1, self.tokens, self.dim).contiguous()


class StubModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = StubTower()
        self.mm_projector = nn.Identity()

    def get_vision_tower(self):
        return self.vision_tower


class StubVITA(VITAMetaForCausalLM, nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self._model = StubModel()

    def get_model(self):
        return self._model


print("=== encode_images_deduped ===")

torch.manual_seed(0)
GROUP = 5
one = torch.randn(GROUP, 3, 16, 16)

for repeats in (2, 3, 4):
    m = StubVITA()
    duplicated = torch.cat([one] * repeats, dim=0)

    full = m.encode_images(duplicated)
    encoded_full = m._model.vision_tower.seen

    m._model.vision_tower.seen = 0
    deduped = m.encode_images_deduped(duplicated, GROUP)
    encoded_dedup = m._model.vision_tower.seen

    check(
        f"x{repeats}: identical output",
        torch.equal(full, deduped),
        f"max diff {(full - deduped).abs().max().item()}",
    )
    check(
        f"x{repeats}: encoded {GROUP} tiles instead of {encoded_full}",
        encoded_dedup == GROUP,
        f"saw {encoded_dedup}",
    )

# A single group must be a plain passthrough, not a needless copy.
m = StubVITA()
single = m.encode_images_deduped(one, GROUP)
check("one group behaves like encode_images", torch.equal(single, m.encode_images(one)))

# Guard against a caller passing a size that does not divide the batch --
# silently mis-tiling the features would be far worse than an exception.
for bad in (3, 0, -1):
    try:
        StubVITA().encode_images_deduped(torch.randn(10, 3, 16, 16), bad)
        check(f"rejects group_size={bad}", False, "no error raised")
    except ValueError:
        check(f"rejects group_size={bad}", True)

print("\n=== order is preserved, not just the multiset ===")
# Distinct tiles: if the repeat tiled them wrongly the values would shuffle.
distinct = torch.arange(GROUP * 3 * 16 * 16, dtype=torch.float32).reshape(GROUP, 3, 16, 16)
m = StubVITA()
dup = torch.cat([distinct, distinct], dim=0)
check(
    "tile order survives the repeat",
    torch.equal(m.encode_images(dup), m.encode_images_deduped(dup, GROUP)),
)

print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")
