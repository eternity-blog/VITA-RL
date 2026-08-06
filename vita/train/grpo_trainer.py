"""GRPO trainer for VITA.

Each step: sample G completions per prompt, score them, normalise the scores
within the group to get advantages, then take a policy-gradient step with a
KL leash to the reference. No critic -- the group is the baseline. That is
what makes GRPO fit here, where a second 7B for a value head would not.

Three details are specific to this model rather than to GRPO:

Sampling has to route around VITAQwen2ForCausalLM.generate, which raises
NotImplementedError on an inputs_embeds kwarg (vita_qwen2.py:209). Calling
Qwen2ForCausalLM.generate directly accepts it, which lets the prompt
embeddings be computed once and reused -- and, once this supports images,
will mean the vision tower runs once per prompt rather than once per
rollout.

Log-probs are recomputed rather than taken from generation. Verified on this
checkpoint that concatenating the sampled tokens' embeddings onto the cached
prompt embeddings and running a forward reproduces generation's own scores
to ~4e-3 (bf16 noise). The recomputed pass is the one that carries
gradients, so it has to happen regardless; taking old_logps from the same
pass keeps the ratio at exactly 1 on the first inner step.

The reference is this model with the adapter off, as in DPO. That requires
every trainable parameter to live inside the adapter -- see train_grpo.py,
which freezes mm_projector for the reason documented there.
"""
from typing import Any, Dict, List, Union

import torch
from torch import nn
from transformers import Qwen2ForCausalLM

from vita.train.grpo_loss import grpo_loss, group_advantages
from vita.train.vita_trainer import VITATrainer


class VITAGRPOTrainer(VITATrainer):
    def __init__(
        self,
        *args,
        reward_fn=None,
        group_size: int = 8,
        grpo_beta: float = 0.04,
        clip_eps: float = 0.2,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if reward_fn is None:
            raise ValueError("VITAGRPOTrainer needs a reward_fn")
        self.reward_fn = reward_fn
        self.group_size = group_size
        self.grpo_beta = grpo_beta
        self.clip_eps = clip_eps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self._degenerate = 0
        self._groups = 0

    # -- rollout ---------------------------------------------------------

    @torch.no_grad()
    def _rollout(self, model, prompt_embeds, prompt_mask):
        """Sample group_size completions for each prompt.

        Returns the sampled ids (N, T_new) and the padding mask over them,
        where N = num_prompts * group_size.
        """
        unwrapped = self.accelerator.unwrap_model(model)
        B, P, _ = prompt_embeds.shape
        G = self.group_size

        # (B, P, D) -> (B*G, P, D), each prompt's copies adjacent so the
        # layout matches group_advantages' expectation.
        embeds = prompt_embeds.unsqueeze(1).expand(B, G, P, prompt_embeds.shape[-1])
        embeds = embeds.reshape(B * G, P, -1)
        mask = prompt_mask.unsqueeze(1).expand(B, G, P).reshape(B * G, P)

        was_training = unwrapped.training
        # eval() disables LoRA dropout: sampling has to come from the policy
        # whose log-probs we will score against, not a dropped-out variant.
        unwrapped.eval()
        try:
            out = Qwen2ForCausalLM.generate(
                unwrapped,
                inputs_embeds=embeds,
                attention_mask=mask,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            if was_training:
                unwrapped.train()

        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id
        # Everything up to and including the first EOS counts; the rest is
        # padding the sampler emitted after finishing.
        is_end = (out == eos) | (out == pad)
        completion_mask = (is_end.cumsum(dim=1) - is_end.long()) == 0
        return out, completion_mask

    # -- scoring ---------------------------------------------------------

    def _score(self, prompts: List[str], completions: List[str], metas: List[dict]):
        rewards, breakdowns = [], []
        G = self.group_size
        for i, text in enumerate(completions):
            prompt_idx = i // G
            r, parts = self.reward_fn(prompts[prompt_idx], text, metas[prompt_idx])
            rewards.append(r)
            breakdowns.append(parts)
        return torch.tensor(rewards, dtype=torch.float32), breakdowns

    # -- log-probs -------------------------------------------------------

    def _sequence_logps(self, model, prompt_embeds, prompt_mask, sampled_ids):
        """Per-token log-probs of the sampled tokens under `model`.

        Rebuilds the full sequence as [cached prompt embeddings | embeddings
        of the sampled tokens] so the prompt is not re-encoded.
        """
        unwrapped = self.accelerator.unwrap_model(model)
        N, T = sampled_ids.shape
        P = prompt_embeds.shape[1]

        token_embeds = unwrapped.get_model().embed_tokens(sampled_ids)
        full = torch.cat([prompt_embeds, token_embeds], dim=1)
        full_mask = torch.cat([prompt_mask, torch.ones_like(sampled_ids, dtype=prompt_mask.dtype)], dim=1)

        logits = model(
            inputs_embeds=full,
            attention_mask=full_mask,
            labels=None,
            use_cache=False,
            return_dict=True,
        ).logits

        # Position P-1 predicts the first sampled token, and so on.
        pred = logits[:, P - 1 : P - 1 + T, :]
        logps = torch.log_softmax(pred.float(), dim=-1)
        return torch.gather(logps, 2, sampled_ids.unsqueeze(2)).squeeze(2)

    # -- loss ------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False):
        prompts = inputs["prompt_texts"]
        metas = inputs["reward_metas"]
        unwrapped = self.accelerator.unwrap_model(model)

        prompt_embeds = unwrapped.get_model().embed_tokens(inputs["input_ids"])
        prompt_mask = inputs["attention_mask"]

        sampled_ids, completion_mask = self._rollout(model, prompt_embeds, prompt_mask)

        # Widen the cached prompt to one copy per rollout.
        B, P, D = prompt_embeds.shape
        G = self.group_size
        rep_embeds = prompt_embeds.unsqueeze(1).expand(B, G, P, D).reshape(B * G, P, D)
        rep_mask = prompt_mask.unsqueeze(1).expand(B, G, P).reshape(B * G, P)

        texts = self.tokenizer.batch_decode(
            sampled_ids.masked_fill(~completion_mask, self.tokenizer.pad_token_id),
            skip_special_tokens=True,
        )
        rewards, breakdowns = self._score(prompts, texts, metas)
        rewards = rewards.to(sampled_ids.device)

        advantages, n_degenerate = group_advantages(rewards, G)
        self._degenerate += n_degenerate
        self._groups += B

        policy_logps = self._sequence_logps(model, rep_embeds, rep_mask, sampled_ids)

        # First inner step, so the sampling policy is the current one; using
        # the same numbers keeps the ratio at exactly 1 rather than
        # introducing bf16 drift between two passes.
        old_logps = policy_logps.detach()

        if not hasattr(unwrapped, "disable_adapter"):
            raise RuntimeError(
                "GRPO needs a peft-wrapped model for the reference policy. "
                "Pass --lora_enable True."
            )
        with torch.no_grad(), unwrapped.disable_adapter():
            ref_logps = self._sequence_logps(model, rep_embeds, rep_mask, sampled_ids).detach()

        loss, metrics = grpo_loss(
            policy_logps,
            old_logps,
            ref_logps,
            advantages,
            completion_mask,
            beta=self.grpo_beta,
            clip_eps=self.clip_eps,
        )

        lengths = completion_mask.sum(dim=1).float()
        log = {
            "reward/mean": float(rewards.mean()),
            "reward/std": float(rewards.std(unbiased=False)),
            "reward/max": float(rewards.max()),
            "completion/len": float(lengths.mean()),
            "groups/degenerate_frac": self._degenerate / max(self._groups, 1),
            **{f"grpo/{k}": v for k, v in metrics.items()},
        }
        if breakdowns:
            for name in breakdowns[0]:
                log[f"reward/{name}"] = sum(b[name] for b in breakdowns) / len(breakdowns)
        self.log(log)

        if return_outputs:
            return loss, {"rewards": rewards, "advantages": advantages}
        return loss

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        # Skip VITATrainer.training_step, which is a bare pass-through, so
        # the Trainer implementation calls the compute_loss above.
        return super(VITATrainer, self).training_step(model, inputs)
