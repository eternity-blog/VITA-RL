"""DPO trainer for VITA.

Subclasses VITATrainer so the optimizer grouping, sampler and adapter-saving
behaviour stay as they are, and overrides only compute_loss. Putting the loss
here rather than inside the model (which is what vita_fo_qwen2.py does for
its state-prediction head) keeps VITAQwen2ForCausalLM free of any RL
concerns, so a GRPO trainer can reuse the same model class later.

Two implementation notes that are specific to this model:

Multimodal fusion runs once per step, not four times. VITAQwen2ForCausalLM.
forward only calls prepare_inputs_labels_for_multimodal when inputs_embeds is
None, so calling it here and passing the result to both the policy and the
reference avoids re-running InternViT and whale for each of them. The call
also returns re-aligned labels, which is not optional: splicing image
embeddings changes the sequence length, so the collator's labels do not line
up with the fused sequence.

The reference model is the same weights with the LoRA adapter switched off.
peft's disable_adapter() restores the base output exactly, so this costs no
extra memory -- as opposed to holding a second frozen 7B.
"""
from typing import Any, Dict, Union

import torch
from torch import nn

from vita.train.dpo_loss import batch_sequence_logps, dpo_loss
from vita.train.vita_trainer import VITATrainer


class VITADPOTrainer(VITATrainer):
    def __init__(self, *args, dpo_beta: float = 0.1, dpo_label_smoothing: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dpo_beta = dpo_beta
        self.dpo_label_smoothing = dpo_label_smoothing

    def _fuse(self, model, inputs):
        """Run the multimodal splice once, returning embeddings and labels."""
        unwrapped = self.accelerator.unwrap_model(model)
        _, position_ids, attention_mask, _, inputs_embeds, labels = (
            unwrapped.prepare_inputs_labels_for_multimodal(
                inputs["input_ids"],
                None,
                inputs["attention_mask"],
                None,
                inputs["labels"],
                inputs.get("images"),
                inputs.get("audios") or None,
                inputs.get("sf_masks"),
            )
        )
        return inputs_embeds, labels, position_ids, attention_mask

    def compute_loss(self, model, inputs, return_outputs=False):
        num_pairs = int(inputs.pop("num_pairs").flatten()[0])

        inputs_embeds, labels, position_ids, attention_mask = self._fuse(model, inputs)

        def logps() -> torch.Tensor:
            out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=None,  # we score the sequences ourselves
                use_cache=False,
                return_dict=True,
            )
            return batch_sequence_logps(out.logits, labels)

        policy_logps = logps()

        # Reference pass: same weights, adapter off, no gradients.
        unwrapped = self.accelerator.unwrap_model(model)
        if not hasattr(unwrapped, "disable_adapter"):
            raise RuntimeError(
                "DPO needs a peft-wrapped model to derive the reference policy "
                "from disable_adapter(). Pass --lora_enable True."
            )
        with torch.no_grad(), unwrapped.disable_adapter():
            ref_logps = logps().detach()

        # The collator lays the batch out as [chosen..., rejected...].
        policy_chosen, policy_rejected = policy_logps[:num_pairs], policy_logps[num_pairs:]
        ref_chosen, ref_rejected = ref_logps[:num_pairs], ref_logps[num_pairs:]

        # A pair whose labels were voided by the tokenization-mismatch path
        # (data_utils_video_audio_neg_patch.py:642 sets target[:] = IGNORE_INDEX
        # and only prints) scores 0.0 on both sides, contributes exactly
        # -log(0.5) forever, and is indistinguishable from healthy training.
        # Count them so a silently useless dataset cannot masquerade as one
        # that simply is not learning.
        dead = int(((policy_chosen == 0.0) & (policy_rejected == 0.0)).sum())
        if dead:
            self._dead_pairs = getattr(self, "_dead_pairs", 0) + dead
            if self._dead_pairs in (1, 10, 100) or self._dead_pairs % 500 == 0:
                print(
                    f"[DPO] warning: {self._dead_pairs} pair(s) so far have no "
                    "supervised tokens on either side and cannot contribute a "
                    "gradient -- check for 'tokenization mismatch' warnings"
                )

        losses, chosen_rewards, rejected_rewards = dpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
            beta=self.dpo_beta,
            label_smoothing=self.dpo_label_smoothing,
        )
        loss = losses.mean()

        margin = chosen_rewards - rejected_rewards
        self.log(
            {
                "rewards/chosen": chosen_rewards.mean().item(),
                "rewards/rejected": rejected_rewards.mean().item(),
                "rewards/margin": margin.mean().item(),
                "rewards/accuracy": (margin > 0).float().mean().item(),
                "logps/chosen": policy_chosen.mean().item(),
                "logps/rejected": policy_rejected.mean().item(),
            }
        )

        if return_outputs:
            return loss, {"chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}
        return loss

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        # VITATrainer.training_step only forwards to super(); go straight to
        # the Trainer implementation so compute_loss above is what runs.
        return super(VITATrainer, self).training_step(model, inputs)
