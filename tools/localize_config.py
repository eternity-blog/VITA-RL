#!/usr/bin/env python
"""Point a VITA-1.5 checkpoint's config.json at locally downloaded encoders.

The released checkpoint ships HuggingFace repo IDs in `mm_vision_tower` and
`mm_audio_encoder`, so loading it reaches out to the network. On an air-gapped
or proxied machine that either fails or is slow. This rewrites both fields to
local paths and keeps a backup at config.json.orig.

Usage:
    python tools/localize_config.py \
        --model-path /path/to/weights/VITA-1.5 \
        --vision-tower /path/to/weights/InternViT-300M-448px

The audio encoder ships inside the VITA-1.5 repository, so it is derived from
--model-path unless --audio-encoder is given explicitly.
"""
import argparse
import json
import os
import shutil

AUDIO_SUBDIR = "audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="downloaded VITA-1.5 directory")
    ap.add_argument("--vision-tower", required=True, help="downloaded InternViT-300M-448px directory")
    ap.add_argument("--audio-encoder", default=None, help="defaults to <model-path>/" + AUDIO_SUBDIR)
    ap.add_argument("--restore", action="store_true", help="restore config.json from config.json.orig")
    args = ap.parse_args()

    cfg_path = os.path.join(args.model_path, "config.json")
    backup = cfg_path + ".orig"

    if args.restore:
        if not os.path.exists(backup):
            raise SystemExit(f"no backup to restore: {backup}")
        shutil.copy2(backup, cfg_path)
        print(f"restored {cfg_path} from {backup}")
        return

    audio = args.audio_encoder or os.path.join(args.model_path, AUDIO_SUBDIR)

    for label, path in (("vision tower", args.vision_tower), ("audio encoder", audio)):
        if not os.path.isdir(path):
            raise SystemExit(f"{label} directory not found: {path}")

    if not os.path.exists(cfg_path):
        raise SystemExit(f"config.json not found: {cfg_path}")

    # Back up once, so re-running does not overwrite the pristine copy.
    if not os.path.exists(backup):
        shutil.copy2(cfg_path, backup)
        print(f"backed up original to {backup}")

    with open(cfg_path) as f:
        cfg = json.load(f)

    cfg["mm_vision_tower"] = os.path.abspath(args.vision_tower)
    cfg["mm_audio_encoder"] = os.path.abspath(audio)

    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"mm_vision_tower  -> {cfg['mm_vision_tower']}")
    print(f"mm_audio_encoder -> {cfg['mm_audio_encoder']}")


if __name__ == "__main__":
    main()
