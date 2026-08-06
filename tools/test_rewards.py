#!/usr/bin/env python
"""Unit-test the rule-based rewards. CPU only, no model.

    PYTHONPATH=./ python tools/test_rewards.py

Beyond the obvious edge cases, these check the property GRPO actually
depends on: a reward has to separate the rollouts in a group. A rule that
returns the same number for a good and a bad answer makes the group
degenerate, which costs a training step and looks like nothing happened.
"""
import sys

from vita.train.rewards import (
    REWARD_REGISTRY,
    RewardCombiner,
    keyword_reward,
    length_reward,
    no_repeat_reward,
    parse_reward_spec,
    state_token_reward,
)

FAILURES = []


def check(name, condition, detail=""):
    print(f"  [{'ok  ' if condition else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not condition:
        FAILURES.append(name)


def in_unit(v):
    return 0.0 <= v <= 1.0


print("=== registry ===")
for want in ("keyword", "length", "no_repeat", "state_token"):
    check(f"'{want}' is registered", want in REWARD_REGISTRY)

print("\n=== keyword ===")
meta = {"keywords": ["cat", "furry"]}
check("both keywords", keyword_reward("", "☜A furry cat sat down.", meta) == 1.0)
check("one of two", keyword_reward("", "☜A cat sat down.", meta) == 0.5)
check("neither", keyword_reward("", "☜A dog barked.", meta) == 0.0)
check("case-insensitive", keyword_reward("", "☜A FURRY CAT.", meta) == 1.0)
check("state token is not counted", keyword_reward("", "☜", meta) == 0.0)
check("no keywords in meta gives 0", keyword_reward("", "anything", {}) == 0.0)
check("empty response", keyword_reward("", "", meta) == 0.0)

print("\n=== length ===")
meta = {"target_len": [10, 30]}
check("inside range", length_reward("", "☜" + "x" * 20, meta) == 1.0)
check("at lower bound", length_reward("", "☜" + "x" * 10, meta) == 1.0)
check("at upper bound", length_reward("", "☜" + "x" * 30, meta) == 1.0)
short = length_reward("", "☜xxxxx", meta)
check("too short is penalised", 0.0 < short < 1.0, f"{short:.3f}")
long_ = length_reward("", "☜" + "x" * 45, meta)
check("too long is penalised", 0.0 < long_ < 1.0, f"{long_:.3f}")
check("empty gives 0", length_reward("", "", meta) == 0.0)
check("only a state token gives 0", length_reward("", "☜", meta) == 0.0)
check("extremely long floors at 0", length_reward("", "x" * 10000, meta) == 0.0)
check(
    "shorter-than-target is graded, not binary",
    length_reward("", "x" * 8, meta) > length_reward("", "x" * 3, meta),
)

print("\n=== no_repeat ===")
check("no repetition", no_repeat_reward("", "a b c d e f g", {}) == 1.0)
loop = no_repeat_reward("", "a cat a cat a cat a cat", {})
check("a loop scores low", loop < 0.6, f"{loop:.3f}")
check("shorter than the n-gram is fine", no_repeat_reward("", "a b", {}) == 1.0)
check("empty is fine", no_repeat_reward("", "", {}) == 1.0)
check(
    "partial repetition sits in between",
    0.0 < no_repeat_reward("", "a b c a b c d e f", {}) < 1.0,
)

print("\n=== state_token ===")
check("any state token accepted by default", state_token_reward("", "☜hello", {}) == 1.0)
check("right token also accepted", state_token_reward("", "☞hello", {}) == 1.0)
check("missing token scores 0", state_token_reward("", "hello", {}) == 0.0)
check("matches the requested state", state_token_reward("", "☞hi", {"state": "right"}) == 1.0)
check("rejects the wrong state", state_token_reward("", "☜hi", {"state": "right"}) == 0.0)
check("empty response scores 0", state_token_reward("", "", {}) == 0.0)

print("\n=== every reward stays inside [0, 1] ===")
samples = ["", "☜", "hello", "☜" + "x" * 5000, "☟a a a a a a", "☞正常的中文回答。"]
metas = [{}, {"keywords": ["x"]}, {"target_len": [5, 20]}, {"state": "left"}]
bad = []
for name, fn in REWARD_REGISTRY.items():
    for s in samples:
        for m in metas:
            v = fn("prompt", s, m)
            if not in_unit(v):
                bad.append((name, s[:12], v))
check("no reward escaped [0, 1]", not bad, str(bad[:3]))

print("\n=== parse_reward_spec ===")
check("name:weight pairs", parse_reward_spec("keyword:1.0,length:0.5") == [("keyword", 1.0), ("length", 0.5)])
check("bare name defaults to 1.0", parse_reward_spec("keyword") == [("keyword", 1.0)])
check("tolerates whitespace", parse_reward_spec(" keyword : 2 , length ") == [("keyword", 2.0), ("length", 1.0)])
try:
    parse_reward_spec("")
    check("rejects an empty spec", False, "no error")
except ValueError:
    check("rejects an empty spec", True)

print("\n=== RewardCombiner ===")
c = RewardCombiner("keyword:1.0,length:1.0")
meta = {"keywords": ["cat"], "target_len": [5, 40]}
good, parts = c("Describe a cat", "☜A small furry cat.", meta)
poor, _ = c("Describe a cat", "☜x", meta)
check("combined score is in [0, 1]", in_unit(good), f"{good:.3f}")
check("breakdown lists every part", set(parts) == {"keyword", "length"}, str(parts))
check("a good answer beats a poor one", good > poor, f"{good:.3f} vs {poor:.3f}")
check("weights are normalised", abs(RewardCombiner("keyword:3.0")("", "☜cat", {"keywords": ["cat"]})[0] - 1.0) < 1e-9)
try:
    RewardCombiner("nonexistent")
    check("rejects an unknown reward", False, "no error")
except ValueError:
    check("rejects an unknown reward", True)

print("\n=== the property GRPO needs: rollouts must separate ===")
c = RewardCombiner("keyword:1.0,length:1.0,no_repeat:1.0")
meta = {"keywords": ["cat", "furry"], "target_len": [15, 60]}
group = [
    "☜A furry cat naps in the sun.",   # hits both keywords, right length
    "☜A cat.",                          # one keyword, too short
    "☜A dog barks loudly at cars.",     # no keywords
    "☜cat cat cat cat cat cat cat",     # keyword but repetitive
]
scores = [c("Describe a cat", g, meta)[0] for g in group]
print("   scores:", [f"{s:.3f}" for s in scores])
check("the best answer wins", scores[0] == max(scores), f"{scores}")
check("scores are not all equal", len(set(round(s, 4) for s in scores)) > 1)
spread = max(scores) - min(scores)
check("spread is usable for advantage", spread > 0.2, f"{spread:.3f}")

print()
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("ALL CHECKS PASSED")
