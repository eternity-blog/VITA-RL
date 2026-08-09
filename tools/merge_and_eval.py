"""Evaluate a LoRA adapter with VLMEvalKit by merging it into a copy of the base.

VLMEvalKit's VITA wrapper takes a single model path and calls
load_pretrained_model on it, with no argument for an adapter. Rather than
patch the wrapper, this merges the adapter into the base weights and writes a
self-contained checkpoint that the unmodified wrapper can load. The merged
copy is ~16 GB, so it is written once and reused across benchmarks.

Merging is also the honest thing to evaluate: it is the model you would ship.
An adapter applied at runtime and a merged checkpoint are numerically the
same thing (peft's merge folds B@A*scaling into the base weight), but the
merged one cannot silently forget to apply the adapter -- which is exactly
the failure that would produce a "DPO changed nothing" result.

Usage:
    python tools/merge_and_eval.py \
        --base /path/to/VITA-1.5 \
        --adapter /path/to/dpo-rlaif-v \
        --out /path/to/VITA-1.5-dpo
"""
import argparse
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="base VITA-1.5 checkpoint dir")
    ap.add_argument("--adapter", required=True, help="dir holding adapter_model.safetensors")
    ap.add_argument("--out", required=True, help="where to write the merged checkpoint")
    ap.add_argument("--model-type", default="qwen2p5_instruct")
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.out, "config.json")):
        print(f"{args.out} already has a config.json; refusing to overwrite. "
              "Delete it first if you meant to re-merge.")
        return

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import torch
    from peft import PeftModel
    from vita.model.builder import load_pretrained_model
    from vita.util.mm_utils import get_model_name_from_path

    print(f"loading base: {args.base}")
    tokenizer, model, _, _ = load_pretrained_model(
        args.base, None, get_model_name_from_path(args.base),
        model_type=args.model_type, device_map="cpu",
    )

    print(f"applying adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter, torch_dtype=torch.bfloat16)
    print("merging...")
    model = model.merge_and_unload()

    os.makedirs(args.out, exist_ok=True)
    print(f"saving merged checkpoint -> {args.out}")
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)

    # The wrapper loads the vision tower and audio encoder by the paths
    # recorded in the base config, and reads a few non-weight files straight
    # from the checkpoint directory. Copy anything the merge did not write.
    for name in os.listdir(args.base):
        src = os.path.join(args.base, name)
        dst = os.path.join(args.out, name)
        if os.path.exists(dst):
            continue
        if os.path.isdir(src):
            print(f"  copying dir {name}")
            shutil.copytree(src, dst)
        elif not name.endswith((".bin", ".safetensors")):
            print(f"  copying {name}")
            shutil.copy2(src, dst)

    print(f"\ndone: {args.out}")
    print("evaluate with:")
    print(f"  VITA_CKPT={args.out} python run.py --data MME --model vita_qwen2 ...")


if __name__ == "__main__":
    main()
