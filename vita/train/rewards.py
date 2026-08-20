"""Pluggable reward functions for GRPO.

GRPO scores its own rollouts during training, so unlike DPO -- where the
preference is baked into the data -- something has to produce a number per
response. This is a registry so that number can come from a hand-written
rule now and a learned reward model later without touching the trainer.

Every reward takes (prompt, response, meta) and returns a float in [0, 1].
Bounded output keeps the weights in `--reward_fns keyword:1.0,length:0.5`
meaningful; a reward free to return 100 would silently dominate whatever it
was combined with.

`meta` is the record's `reward_meta` field, so a rule can be told what to
look for per sample rather than hardcoding it.

Two things to keep in mind when writing a new rule:

Rules must discriminate. GRPO's advantage is (r - mean) / std within a
group, so a rule that returns the same value for every rollout of a prompt
contributes nothing -- the group is degenerate and gets zero advantage. A
binary rule that all eight rollouts pass is worth exactly as much as no
rule at all. Prefer graded output over pass/fail.

Rules must be cheap. They run on every rollout of every step: at group size
8 and batch 2, that is 16 calls per step.
"""
import re
from typing import Callable, Dict, List, Sequence, Tuple

REWARD_REGISTRY: Dict[str, Callable[[str, str, dict], float]] = {}

# The state tokens VITA prefixes onto every reply (data_utils:128-135).
STATE_TOKENS = ("☜", "☞", "☟")  # 'left', 'right', 'down'


def register_reward(name: str):
    def wrap(fn):
        if name in REWARD_REGISTRY:
            raise ValueError(f"reward '{name}' is already registered")
        REWARD_REGISTRY[name] = fn
        return fn

    return wrap


def _strip_state_token(text: str) -> str:
    return text[1:] if text[:1] in STATE_TOKENS else text


@register_reward("keyword")
def keyword_reward(prompt: str, response: str, meta: dict) -> float:
    """Fraction of the expected keywords that appear in the response.

    meta: {"keywords": ["cat", "animal"]}

    Graded rather than all-or-nothing precisely so a group of rollouts can
    be ranked against each other.
    """
    keywords: Sequence[str] = meta.get("keywords") or []
    if not keywords:
        return 0.0
    body = _strip_state_token(response).lower()
    hits = sum(1 for k in keywords if k.lower() in body)
    return hits / len(keywords)


@register_reward("length")
def length_reward(prompt: str, response: str, meta: dict) -> float:
    """1.0 inside the target character range, decaying linearly outside.

    meta: {"target_len": [15, 60]}

    Both directions are penalised: a one-word answer and a rambling one are
    each worse than a right-sized one.
    """
    target = meta.get("target_len") or [10, 200]
    lo, hi = int(target[0]), int(target[1])
    n = len(_strip_state_token(response).strip())
    if n == 0:
        return 0.0
    if lo <= n <= hi:
        return 1.0
    if n < lo:
        return max(0.0, n / lo)
    # Score halves once the response is twice the upper bound.
    return max(0.0, 1.0 - (n - hi) / max(hi, 1))


@register_reward("no_repeat")
def no_repeat_reward(prompt: str, response: str, meta: dict) -> float:
    """Ratio of distinct 3-grams, i.e. 1.0 when nothing repeats.

    Degenerate sampling loops ("a cat a cat a cat") are a real failure mode
    once a policy starts chasing another reward, so this acts as a brake.
    """
    words = _strip_state_token(response).split()
    n = meta.get("ngram", 3)
    if len(words) < n:
        return 1.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_answer(response: str) -> str:
    """The <answer> tag contents, falling back to the last number in the text.

    The fallback keeps the reward from being all-zero before the policy has
    learned the tag format -- otherwise the answer and format rewards would
    be perfectly correlated early on and the group would be degenerate.
    """
    body = _strip_state_token(response)
    m = _ANSWER_TAG.search(body)
    if m:
        return m.group(1).strip()
    nums = _NUMBER.findall(body)
    return nums[-1] if nums else ""


@register_reward("answer")
def answer_reward(prompt: str, response: str, meta: dict) -> float:
    """Exact match against a verifiable gold answer (R1-style binary reward).

    meta: {"answer": "3"}

    Binary is fine for GRPO: the signal is the *within-group* variance of
    pass/fail across the G rollouts, not the smoothness of any single score.
    Numeric answers compare as numbers ("3" == "3.0"); anything else
    compares as case-insensitive text.
    """
    gold = str(meta.get("answer", "")).strip()
    if not gold:
        return 0.0
    pred = _extract_answer(response)
    if not pred:
        return 0.0
    try:
        return 1.0 if float(pred) == float(gold) else 0.0
    except ValueError:
        return 1.0 if pred.lower() == gold.lower() else 0.0


@register_reward("format")
def format_reward(prompt: str, response: str, meta: dict) -> float:
    """R1-style structure check: <think>...</think> then <answer>...</answer>.

    Graded (1.0 full structure / 0.5 answer tag only / 0.0 neither) so early
    groups where nobody has the full format can still rank rollouts.
    """
    body = _strip_state_token(response).strip()
    if re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", body, re.S | re.I):
        return 1.0
    if _ANSWER_TAG.search(body):
        return 0.5
    return 0.0


@register_reward("state_token")
def state_token_reward(prompt: str, response: str, meta: dict) -> float:
    """1.0 when the reply opens with the expected state token.

    meta: {"state": "left"} -- or omit it to accept any of the three.

    VITA is trained to prefix every reply with one of these, so dropping it
    is a regression in format even when the content is fine.
    """
    first = response[:1]
    if first not in STATE_TOKENS:
        return 0.0
    want = meta.get("state")
    if not want:
        return 1.0
    expected = {"left": "☜", "right": "☞", "down": "☟"}.get(want)
    return 1.0 if expected and first == expected else 0.0


class JudgeReward:
    """Score responses with a small instruct model.

    Registered lazily under "judge" by the trainer when --judge_model_path
    is given, because loading it costs memory that a rules-only run should
    not pay.

    Rather than parsing a number out of the reply, this reads the
    probability the model assigns to each of the tokens "1".."5" at the
    first generated position and returns their weighted mean, rescaled to
    [0, 1]. That gives a continuous signal -- which is what GRPO needs to
    rank a group -- and cannot fail to parse.
    """

    TEMPLATE = (
        "Rate how well the response answers the question, from 1 (poor) to "
        "5 (excellent). Reply with a single digit.\n\n"
        "Question: {prompt}\n\nResponse: {response}\n\nRating:"
    )

    # When the sample carries the gold answer (reward_meta["gold"]), grade
    # against it. A text-only judge cannot see the image, so without the
    # reference it can only rate fluency/relevance; with it, agreement with
    # the reference is a groundedness check.
    TEMPLATE_WITH_REF = (
        "Rate how well the response agrees with the reference answer to the "
        "question, from 1 (contradicts or fabricates) to 5 (matches its "
        "content). Reply with a single digit.\n\n"
        "Question: {prompt}\n\nReference answer: {gold}\n\n"
        "Response: {response}\n\nRating:"
    )

    def __init__(self, model_path: str, device: str = "cuda", dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype or torch.bfloat16
        ).to(device).eval()
        self.device = device
        # Single-token ids for the digits 1-5.
        self.digit_ids = []
        for d in "12345":
            ids = self.tokenizer.encode(d, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    f"judge tokenizer splits the digit {d!r} into {len(ids)} tokens; "
                    "this scorer needs single-token digits"
                )
            self.digit_ids.append(ids[0])

    def __call__(self, prompt: str, response: str, meta: dict) -> float:
        torch = self.torch
        gold = (meta or {}).get("gold", "")
        if gold:
            text = self.TEMPLATE_WITH_REF.format(
                prompt=prompt.strip()[:1000],
                gold=gold.strip()[:1000],
                response=_strip_state_token(response).strip()[:1000],
            )
        else:
            text = self.TEMPLATE.format(
                prompt=prompt.strip()[:1000],
                response=_strip_state_token(response).strip()[:1000],
            )
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1]
        probs = torch.softmax(logits[self.digit_ids].float(), dim=-1)
        score = float((probs * torch.arange(1, 6, device=probs.device)).sum())
        return (score - 1.0) / 4.0  # 1..5 -> 0..1


def parse_reward_spec(spec: str) -> List[Tuple[str, float]]:
    """Parse "keyword:1.0,length:0.5" into [("keyword", 1.0), ("length", 0.5)].

    A bare name defaults to weight 1.0.
    """
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, w = part.rsplit(":", 1)
            out.append((name.strip(), float(w)))
        else:
            out.append((part, 1.0))
    if not out:
        raise ValueError(f"no reward functions parsed from {spec!r}")
    return out


class RewardCombiner:
    """Weighted sum of registered rewards, normalised back into [0, 1]."""

    def __init__(self, spec: str, extra: Dict[str, Callable] = None):
        self.parts = []
        registry = dict(REWARD_REGISTRY)
        registry.update(extra or {})
        for name, weight in parse_reward_spec(spec):
            if name not in registry:
                raise ValueError(
                    f"unknown reward '{name}'. available: {sorted(registry)}"
                )
            self.parts.append((name, weight, registry[name]))
        self.total_weight = sum(w for _, w, _ in self.parts) or 1.0

    def __call__(self, prompt: str, response: str, meta: dict) -> Tuple[float, Dict[str, float]]:
        breakdown = {}
        total = 0.0
        for name, weight, fn in self.parts:
            v = float(fn(prompt, response, meta or {}))
            breakdown[name] = v
            total += weight * v
        return total / self.total_weight, breakdown

    def names(self) -> List[str]:
        return [n for n, _, _ in self.parts]
