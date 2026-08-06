#!/usr/bin/env python
"""GRPO training entry point.

Reuses vita/train/train.py for model construction, freeze handling and LoRA
setup -- same arrangement as train_dpo.py -- and swaps in the prompt dataset
and the GRPO trainer.

    deepspeed --include localhost:0 vita/train/train_grpo.py \\
        --lora_enable True --dataset_use GRPOSmokeTest \\
        --reward_fns keyword:1.0,length:0.5 ...

--lora_enable True is mandatory: the reference policy is this model with the
adapter switched off.
"""
from dataclasses import dataclass, field
from typing import Optional

from vita.train.grpo_data import make_grpo_data_module
from vita.train.grpo_trainer import VITAGRPOTrainer
from vita.train.rewards import RewardCombiner
from vita.train.train import train


@dataclass
class GRPOArguments:
    grpo_group_size: int = field(
        default=8,
        metadata={
            "help": "Rollouts per prompt. The group is GRPO's baseline, so "
            "too few makes the advantage estimate noisy."
        },
    )
    grpo_beta: float = field(
        default=0.04, metadata={"help": "Weight on the KL penalty toward the reference."}
    )
    grpo_clip_eps: float = field(
        default=0.2, metadata={"help": "PPO-style trust region on the importance ratio."}
    )
    grpo_max_new_tokens: int = field(default=128, metadata={"help": "Rollout length cap."})
    grpo_temperature: float = field(default=1.0, metadata={"help": "Sampling temperature."})
    grpo_top_p: float = field(default=0.95, metadata={"help": "Nucleus sampling cutoff."})
    reward_fns: str = field(
        default="keyword:1.0,length:0.5,no_repeat:0.5",
        metadata={
            "help": "Comma-separated name:weight list, e.g. "
            "'keyword:1.0,length:0.5'. Names come from vita/train/rewards.py."
        },
    )
    judge_model_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to a small instruct model to score rollouts. When "
            "set, a 'judge' reward is registered and can be referenced in "
            "--reward_fns."
        },
    )


def _data_module(tokenizer, data_args, extras):
    return make_grpo_data_module(tokenizer=tokenizer, data_args=data_args)


def _trainer(model, tokenizer, training_args, data_module, extras):
    (grpo_args,) = extras
    if not getattr(training_args, "lora_enable", False):
        raise SystemExit(
            "GRPO requires --lora_enable True.\n"
            "The reference policy is this model with the LoRA adapter "
            "disabled, which costs no extra memory."
        )

    # Same reason as train_dpo.py: train.py applies LoRA before
    # initialize_vision_modules, which then force-enables mm_projector's
    # gradients ("In case it is frozen by LoRA", vita_arch.py:59-61).
    # disable_adapter() cannot undo that, so anything left trainable outside
    # the adapter makes the reference drift away from the base policy while
    # the KL term keeps reporting plausible numbers.
    frozen = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            continue
        if param.requires_grad:
            param.requires_grad = False
            frozen += param.numel()
    if frozen:
        print(
            f"[GRPO] froze {frozen / 1e6:.1f}M non-adapter parameters "
            "so the reference policy stays fixed"
        )

    extra_rewards = {}
    if grpo_args.judge_model_path:
        from vita.train.rewards import JudgeReward

        print(f"[GRPO] loading judge model from {grpo_args.judge_model_path}")
        extra_rewards["judge"] = JudgeReward(grpo_args.judge_model_path)

    reward_fn = RewardCombiner(grpo_args.reward_fns, extra=extra_rewards)
    print(f"[GRPO] rewards: {reward_fn.names()}")

    return VITAGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        reward_fn=reward_fn,
        group_size=grpo_args.grpo_group_size,
        grpo_beta=grpo_args.grpo_beta,
        clip_eps=grpo_args.grpo_clip_eps,
        max_new_tokens=grpo_args.grpo_max_new_tokens,
        temperature=grpo_args.grpo_temperature,
        top_p=grpo_args.grpo_top_p,
        **data_module,
    )


if __name__ == "__main__":
    train(
        extra_arg_classes=(GRPOArguments,),
        data_module_factory=_data_module,
        trainer_factory=_trainer,
    )
