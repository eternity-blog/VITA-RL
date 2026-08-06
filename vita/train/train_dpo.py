#!/usr/bin/env python
"""DPO training entry point.

Reuses vita/train/train.py for model construction, freeze handling and LoRA
setup, and swaps in the preference dataset and the DPO trainer. Everything
about how the model is built is therefore identical to SFT, which is the
point -- a divergence there would be invisible and would show up as a
mysterious difference in results.

Usage: see script/train/dpo_smoke_test.sh. The short version:

    deepspeed --include localhost:0 vita/train/train_dpo.py \\
        --lora_enable True --dataset_use DPOSmokeTest --dpo_beta 0.1 ...

--lora_enable True is mandatory: the reference policy is this same model with
the adapter switched off, so without an adapter there is nothing to switch.
"""
from dataclasses import dataclass, field

from vita.train.dpo_data import make_dpo_data_module
from vita.train.dpo_trainer import VITADPOTrainer
from vita.train.train import train


@dataclass
class DPOArguments:
    dpo_beta: float = field(
        default=0.1,
        metadata={
            "help": "Strength of the KL constraint toward the reference policy. "
            "Lower means the policy may drift further."
        },
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": "Conservative DPO: assumed probability that a preference "
            "label is flipped. 0 disables it."
        },
    )


def _data_module(tokenizer, data_args, extras):
    return make_dpo_data_module(tokenizer=tokenizer, data_args=data_args)


def _trainer(model, tokenizer, training_args, data_module, extras):
    (dpo_args,) = extras
    if not getattr(training_args, "lora_enable", False):
        raise SystemExit(
            "DPO requires --lora_enable True.\n"
            "The reference policy is this model with the LoRA adapter disabled, "
            "which costs no extra memory. Full-parameter DPO would need a "
            "second frozen 7B and is not implemented."
        )

    # The reference policy is this model with the adapter switched off, so
    # every trainable parameter must live *inside* the adapter. mm_projector
    # does not: train.py applies LoRA (line 388) before
    # initialize_vision_modules (line 395), and that method force-enables
    # mm_projector's grads with the comment "In case it is frozen by LoRA"
    # (vita_arch.py:59-61). Left alone it would keep training, disable_adapter()
    # would not revert it, and the reference would silently drift away from the
    # base model after step 1 -- while the loss still looked plausible.
    #
    # Freezing it keeps the reference exactly equal to the base policy, which
    # is what makes the first-step loss == -log(0.5) check meaningful.
    frozen = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            continue
        if param.requires_grad:
            param.requires_grad = False
            frozen += param.numel()
    if frozen:
        print(f"[DPO] froze {frozen / 1e6:.1f}M non-adapter parameters "
              f"so the reference policy stays fixed")

    return VITADPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        dpo_beta=dpo_args.dpo_beta,
        dpo_label_smoothing=dpo_args.dpo_label_smoothing,
        **data_module,
    )


if __name__ == "__main__":
    train(
        extra_arg_classes=(DPOArguments,),
        data_module_factory=_data_module,
        trainer_factory=_trainer,
    )
