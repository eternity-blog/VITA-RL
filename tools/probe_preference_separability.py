"""Measure how well a model already separates preference pairs, before training.

DPO's gradient is proportional to how wrong the model currently is about a
pair. If the base policy assigns chosen and rejected nearly the same sequence
log-probability, there is very little to learn from -- and a flat
rewards/margin during training is the data telling you so, not a bug.

This scores pairs with the frozen base model and reports:

  - base accuracy: how often logp(chosen) > logp(rejected) already. 50% means
    the model cannot tell them apart at all; well above 50% means the signal
    is there and a flat training curve is a training problem instead.
  - the log-prob gap distribution, which is the scale beta acts on.

Both are worth knowing before spending an hour on a run.

Scoring goes through DPODataset and DPODataCollator -- the same objects the
trainer uses -- rather than reassembling the prompt here. An earlier version
of this script built its own prompt and tripped vita_arch's
"one image feature per <image> token" assertion; reusing the real pipeline
means the numbers are guaranteed to be the ones training sees.

Usage:
    VITA_RLAIF_DATA_DIR=/path/to/rlaif_v_dpo \\
    python tools/probe_preference_separability.py --n 100
"""
import argparse
import os
import sys
from dataclasses import dataclass, field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("VITA_CKPT"))
    ap.add_argument("--vision-tower", default=None)
    ap.add_argument("--audio-encoder", default=None)
    ap.add_argument("--dataset-use", default="RLAIFV")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model-type", default="qwen2p5_instruct")
    args = ap.parse_args()

    if not args.model:
        sys.exit("pass --model or set VITA_CKPT")

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)

    import torch
    import transformers
    from vita import conversation as conversation_lib
    from vita.model.builder import load_pretrained_model
    from vita.train.dpo_data import make_dpo_data_module
    from vita.train.dpo_loss import batch_sequence_logps
    from vita.util.mm_utils import get_model_name_from_path

    # preprocess() dispatches on this module-level global, which train.py sets
    # from --version. Leaving it at the default routes into the mixtral branch,
    # where four live pdb.set_trace() calls hang the process with no output
    # (HANDBOOK 6.1). Set it before touching the dataset.
    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.model_type
    ]

    weights_root = os.path.dirname(args.model.rstrip("/"))
    vision_tower = args.vision_tower or os.path.join(weights_root, "InternViT-300M-448px")

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model, None, get_model_name_from_path(args.model),
        model_type=args.model_type, device_map="auto",
    )
    model.eval()

    # builder.py casts the towers to fp16 (builder.py:271) but the LLM may
    # load in another dtype, and InternViTVisionTower.forward re-casts its
    # input to self.dtype -- a property reading the wrapped module. Under
    # DeepSpeed the trainer never sees the mismatch because everything is cast
    # up front; scoring outside the trainer does. Pin the whole model to one
    # dtype so the conv layers and their inputs agree.
    run_dtype = torch.float16
    model.to(dtype=run_dtype)

    # LazySupervisedDataset reads its configuration off a data_args object.
    # These are the same values script/train/dpo_rlaif_v.sh passes.
    @dataclass
    class DataArgs:
        dataset_use: str = args.dataset_use
        lazy_preprocess: bool = True
        is_multimodal: bool = True
        image_aspect_ratio: str = "square"
        image_grid_pinpoints: str = None
        min_dynamic_patch: int = 1
        max_dynamic_patch: int = 12
        use_thumbnail: bool = True
        image_processor: object = None
        data_ratio: float = 1.0

    data_args = DataArgs()
    data_args.image_processor = image_processor
    module = make_dpo_data_module(tokenizer=tokenizer, data_args=data_args)
    dataset, collator = module["train_dataset"], module["data_collator"]

    n = min(args.n, len(dataset))
    print(f"scoring {n} of {len(dataset)} pairs with the frozen base model\n")

    wins, usable, gaps = 0, 0, []
    for i in range(n):
        try:
            batch = collator([dataset[i]])
            batch.pop("num_pairs")
            group = batch.pop("image_group_size", None)
            gs = int(group.flatten()[0]) if group is not None else None

            dev = next(model.parameters()).device

            def _cast(t, dtype):
                if hasattr(t, "is_floating_point") and t.is_floating_point():
                    return t.to(device=dev, dtype=dtype)
                return t.to(dev) if hasattr(t, "to") else t

            images = batch.get("images")
            if images is not None:
                # builder.py loads both towers in fp16 regardless of the LLM's
                # dtype, and the collator hands over fp32 pixels and a dummy
                # fp32 audio tensor even for records with no audio. Cast each
                # to the module that actually consumes it.
                images = _cast(images, next(model.get_vision_tower().parameters()).dtype)

            audios = batch.get("audios") or None
            if isinstance(audios, dict):
                adtype = next(model.get_audio_encoder().parameters()).dtype
                audios = {k: _cast(v, adtype) for k, v in audios.items()}

            sf_masks = batch.get("sf_masks")
            if hasattr(sf_masks, "to"):
                sf_masks = sf_masks.to(dev)

            with torch.no_grad():
                _, position_ids, attention_mask, _, embeds, labels = (
                    model.prepare_inputs_labels_for_multimodal(
                        batch["input_ids"].to(dev), None,
                        batch["attention_mask"].to(dev), None,
                        batch["labels"].to(dev), images,
                        audios, sf_masks,
                        image_group_size=gs,
                    )
                )
                out = model(inputs_embeds=embeds, attention_mask=attention_mask,
                            position_ids=position_ids, labels=None,
                            use_cache=False, return_dict=True)
                logps = batch_sequence_logps(out.logits, labels)
        except Exception as exc:  # noqa: BLE001
            if usable == 0 and i < 3:
                import traceback
                print(f"  record {i} failed: {type(exc).__name__}: {exc}")
                traceback.print_exc()
            continue

        # Collator layout is [chosen..., rejected...] with one pair here.
        chosen, rejected = float(logps[0]), float(logps[1])
        usable += 1
        gaps.append(chosen - rejected)
        wins += chosen > rejected
        if usable % 20 == 0:
            print(f"  {usable} scored, running accuracy {wins/usable:.1%}")

    if not usable:
        sys.exit("no pairs could be scored")

    import math
    import statistics as st
    p = wins / usable
    # Wald CI on the estimate, and a two-sided test against chance using the
    # null's own variance (0.25/n), which is the right denominator here.
    ci = 1.96 * math.sqrt(p * (1 - p) / usable)
    z = (p - 0.5) / math.sqrt(0.25 / usable)
    pv = math.erfc(abs(z) / math.sqrt(2))

    print(f"\npairs scored: {usable}")
    print(f"base accuracy: {p:.1%}   ({wins}/{usable})")
    print(f"  95% CI: [{max(0.0, p-ci):.1%}, {min(1.0, p+ci):.1%}]")
    print(f"  vs chance: z={z:.2f}, two-sided p={pv:.3f}"
          f"  -> {'above chance' if pv < 0.05 else 'NOT distinguishable from chance'}")
    mean_gap, sd_gap = st.mean(gaps), st.pstdev(gaps)
    print(f"logp gap chosen-rejected: mean={mean_gap:+.2f} "
          f"median={st.median(gaps):+.2f} sd={sd_gap:.2f}")
    if sd_gap > 0:
        print(f"  signal-to-noise: {abs(mean_gap)/sd_gap:.3f}")

    print("\nHow to read this:")
    print("  ~50% and CI spanning it -> the base cannot separate these pairs;")
    print("     a flat rewards/margin is the data, not the trainer.")
    print("  Above chance but SNR below ~0.2 -> real but weak. Expect little")
    print("     movement at a few hundred steps; budget more data or pick")
    print("     pairs closer to the base's own decision boundary.")
    print("  65%+ -> plenty of signal. A flat margin then IS a training bug.")


if __name__ == "__main__":
    main()
