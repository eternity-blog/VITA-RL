#!/usr/bin/env python
"""Generate synthetic image+text prompts to smoke-test the multimodal GRPO path.

Mirrors make_grpo_smoke_data.py but each record carries an "image" field, so
the trainer exercises the vision-fusion code path that the text smoke does
not: encode_images -> mm_projector -> prepare_inputs_labels_for_multimodal
splices the tile features into the token embeddings once per batch, then
_rollout samples G completions and _sequence_logps scores them against the
shared fused prompt.

Not real training data. Each prompt ships the reward_meta its rules need --
keywords grounded in what the image actually depicts, so a rollout that
describes the image correctly scores higher than one that hallucinates or
rambles. That spread is what GRPO ranks on; without it every rollout in a
group scores the same and the group is degenerate (zero advantage, no
gradient). The keywords are picked to be words a correct English description
would plausibly contain, not prompts the model is led toward, so the reward
is a real check on groundedness.

    python tools/make_grpo_mm_smoke_data.py --out-dir /path/to/grpo_mm_smoke_data
    export VITA_GRPO_MM_DATA_DIR=/path/to/grpo_mm_smoke_data
"""
import argparse
import json
import os
import shutil

# (file, keywords a correct description would mention, [min_len, max_len],
#  questions to ask about it). The keywords are lowercase-matched by
# keyword_reward, so casing in the response does not matter; "VITA" matches
# because both sides are .lower()-ed.
IMAGES = [
    ("vita_newlog.jpg", ["VITA", "logo"], [30, 200],
     ["Describe this image.", "What does this logo say?"]),
    ("vita_15_audio_2.jpg", ["table", "error rate"], [40, 250],
     ["Describe this image.", "What is shown in this table?"]),
    ("vita_mllm_performance.png", ["chart", "benchmark"], [40, 250],
     ["Describe this image.", "What does this chart show?"]),
    ("vita_15_audio_training.png", ["audio", "image"], [40, 250],
     ["Describe this image.", "What does this diagram show?"]),
]


def build(repeat: int):
    data = []
    idx = 0
    for r in range(repeat):
        for img, keywords, length, questions in IMAGES:
            for q in questions:
                data.append(
                    {
                        "set": "grpo_mm_smoke",
                        "id": f"mm_{idx:04d}",
                        "conversations": [
                            {"from": "human", "value": f"<image>\n{q}"}
                        ],
                        "image": img,
                        "reward_meta": {
                            "keywords": keywords,
                            "target_len": length,
                            # VITA prefixes every reply with a state token;
                            # the image system prompt should yield the same, so
                            # keeping this lets the same --reward_fns spec as the
                            # text smoke run unchanged.
                            "state": "left",
                        },
                    }
                )
                idx += 1
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repeat", type=int, default=1, help="passes over the image list")
    ap.add_argument("--asset-dir", default="asset", help="source images (repo's own)")
    args = ap.parse_args()

    if not os.path.isdir(args.asset_dir):
        raise SystemExit(
            f"asset directory not found: {args.asset_dir}\n"
            "Run this from the repository root, or pass --asset-dir."
        )

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    for name, _kw, _len, _qs in IMAGES:
        src = os.path.join(args.asset_dir, name)
        if not os.path.exists(src):
            raise SystemExit(f"missing asset: {src}")
        shutil.copy2(src, os.path.join(img_dir, name))

    data = build(args.repeat)
    out = os.path.join(args.out_dir, "grpo_mm_train.json")
    with open(out, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

    print(f"wrote {len(data)} image prompts to {out}")
    for name, kw, _, _ in IMAGES:
        print(f"  {name:30} keywords={kw}")
    print(f"\nnow run:\n  export VITA_GRPO_MM_DATA_DIR={os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
