#!/usr/bin/env python
"""Inspect a configured dataset without a GPU or the 7B checkpoint.

Use this after wiring up a new dataset in vita/config/dataset_config.py and
before spending 8 GPUs on a training run. It loads the real tokenizer and
image/audio processors but never the language model, so it finishes in
seconds on CPU and tells you the things that actually go wrong:

  - does the dataset registry resolve, and are the files where it thinks
  - how long are the sequences, and do they fit model_max_length
  - which span of text is actually supervised (labels != IGNORE_INDEX)
  - does the collator batch the samples without a shape mismatch
  - how many samples hit the silent "tokenization mismatch" path that voids
    their labels (see ARCHITECTURE.md 9.3) -- this is the one to watch, it
    prints a warning and otherwise looks completely normal

Usage:
    export VITA_WEIGHTS=/path/to/weights
    export VITA_SMOKE_DATA_DIR=/path/to/smoke_data   # if using SmokeTest
    PYTHONPATH=./ python tools/inspect_dataset.py --dataset-use SmokeTest

Do not name a copy of this file inspect.py -- that shadows the standard
library module and breaks the torch import.
"""
import argparse
import os
import sys

import transformers
import yaml

from vita import conversation as conversation_lib
from vita.constants import IGNORE_INDEX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-use", default="SmokeTest",
                    help="key into DataConfig (vita/config/__init__.py)")
    ap.add_argument("--weights", default=os.environ.get("VITA_WEIGHTS"),
                    help="directory holding VITA-1.5/ and InternViT-300M-448px/")
    ap.add_argument("--version", default="qwen2p5_instruct",
                    help="conversation template; must match the training script")
    ap.add_argument("--num-samples", type=int, default=4, help="how many to inspect")
    ap.add_argument("--max-length", type=int, default=6200)
    args = ap.parse_args()

    if not args.weights:
        sys.exit("error: set VITA_WEIGHTS or pass --weights")

    model_path = os.path.join(args.weights, "VITA-1.5")
    vision_path = os.path.join(args.weights, "InternViT-300M-448px")
    audio_path = os.path.join(
        model_path, "audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning"
    )
    for p in (model_path, vision_path, audio_path):
        if not os.path.isdir(p):
            sys.exit(f"error: not a directory: {p}")

    # The data pipeline dispatches on the *global* default conversation, not on
    # anything passed in. Leaving it at the default sends qwen2p5 data through
    # preprocess_mixtral_two, which contains live pdb.set_trace() calls.
    if args.version not in conversation_lib.conv_templates:
        sys.exit(f"error: unknown --version {args.version}")
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]

    # Imported late: reads DataConfig at import time, which reads env vars.
    from vita.util.data_utils_video_audio_neg_patch import (
        DataArguments,
        make_supervised_data_module,
    )
    from vita.model.multimodal_encoder.internvit.internvit_encoder import (
        InternViTVisionTower,
    )
    from vita.model.multimodal_encoder.whale.init_model import audioEncoderProcessor

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, model_max_length=args.max_length, padding_side="right", use_fast=True
    )

    data_args = DataArguments()
    data_args.dataset_use = args.dataset_use
    data_args.image_aspect_ratio = "square"
    data_args.is_multimodal = True
    data_args.lazy_preprocess = True

    # Only the processors are needed, not the encoder weights -- but
    # InternViTVisionTower loads its model in __init__, so this does read the
    # 300M vision checkpoint. It stays on CPU.
    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.mm_vision_tower = vision_path
    data_args.image_processor = InternViTVisionTower(vision_path, args=cfg).image_processor

    whale_cfg = yaml.safe_load(open(os.path.join(audio_path, "train.yaml")))
    data_args.audio_processor = audioEncoderProcessor(dataset_conf=whale_cfg["dataset_conf"])

    module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    dataset = module["train_dataset"]
    collator = module["data_collator"]

    print(f"\n=== dataset '{args.dataset_use}': {len(dataset)} samples ===\n")

    n = min(args.num_samples, len(dataset))
    voided = 0
    for i in range(n):
        s = dataset[i]
        seq_len = s["input_ids"].shape[0]
        supervised = (s["labels"] != IGNORE_INDEX).sum().item()
        if supervised == 0:
            voided += 1
        modalities = [k for k in ("image", "audio") if k in s]
        print(f"[{i}] len={seq_len:5d}  supervised={supervised:5d}  has={modalities}")
        if supervised:
            text = tokenizer.decode(s["labels"][s["labels"] != IGNORE_INDEX])
            print(f"     supervised text: {text[:90]!r}")
        else:
            print("     *** no supervised tokens -- labels were voided ***")
        if seq_len > args.max_length:
            print(f"     *** exceeds model_max_length ({args.max_length}), will be truncated ***")

    batch = collator([dataset[i] for i in range(n)])
    print(f"\n=== collated batch of {n} ===")
    for k, v in sorted(batch.items()):
        if hasattr(v, "shape"):
            print(f"  {k:15} {tuple(v.shape)}")
        elif isinstance(v, dict):
            for k2, v2 in sorted(v.items()):
                shape = tuple(v2.shape) if hasattr(v2, "shape") else v2
                print(f"  {k+'.'+k2:15} {shape}")
        else:
            print(f"  {k:15} {type(v).__name__} (len {len(v)})")

    if voided:
        print(
            f"\nWARNING: {voided}/{n} inspected samples have no supervised tokens.\n"
            "Those samples still run forward and backward but contribute nothing\n"
            "to the loss. Check the 'tokenization mismatch' warnings above and\n"
            "verify --version matches the template your data was built for."
        )
    else:
        print(f"\nAll {n} inspected samples have supervised tokens.")


if __name__ == "__main__":
    main()
