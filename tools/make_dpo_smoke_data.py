#!/usr/bin/env python
"""Generate a tiny synthetic preference set to smoke-test the DPO pipeline.

This is NOT real preference data. It exercises the code path end to end:
pair encoding, the 2B collation, the reference-model pass, and the loss. The
preferences are trivially separable on purpose -- a correct implementation
should drive the reward margin positive within a handful of steps, which is
the signal we are looking for.

Each record pairs an accurate description (chosen) with one of three kinds of
degraded response (rejected):

  1. generic  -- fluent but says nothing about this particular image
  2. wrong    -- confidently describes something that is not there
  3. empty    -- technically responsive but contentless

Run from the repository root:
    python tools/make_dpo_smoke_data.py --out-dir /path/to/dpo_smoke_data
    export VITA_DPO_DATA_DIR=/path/to/dpo_smoke_data
"""
import argparse
import json
import os
import shutil

# (file, accurate description)
IMAGES = [
    ("vita_newlog.jpg", "A blue logo reading 'VITA' with the text 'Open-Source Interactive MLLM' beneath it."),
    ("vita_15_audio_2.jpg", "A table comparing automatic speech recognition word error rates across several models."),
    ("vita_mllm_performance.png", "A chart of benchmark scores across multimodal evaluation suites."),
    ("vita_15_audio_training.png", "A diagram showing how adding the audio modality affects image understanding."),
]

GENERIC = [
    "This is an interesting image with a lot of detail worth discussing.",
    "The picture shows various elements arranged in a visually pleasing way.",
    "There are several things going on here that are worth pointing out.",
]

WRONG = [
    "A photograph of three cats sleeping on a red sofa next to a window.",
    "An aerial view of a coastal city at sunset with boats in the harbour.",
    "A plate of pasta with tomato sauce and fresh basil on a wooden table.",
]

EMPTY = [
    "An image.",
    "It is a picture.",
    "Something is shown.",
]

QUESTIONS = [
    "Describe this image.",
    "What does this image show?",
    "Please describe what you see.",
]


def build(n_per_type: int):
    data = []
    idx = 0
    for kind, pool in (("generic", GENERIC), ("wrong", WRONG), ("empty", EMPTY)):
        for i in range(n_per_type):
            img, good = IMAGES[i % len(IMAGES)]
            data.append(
                {
                    "set": "dpo_smoke",
                    "id": f"{kind}_{idx:04d}",
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{QUESTIONS[i % len(QUESTIONS)]}"},
                        {"from": "gpt", "value": good},
                    ],
                    "rejected": pool[i % len(pool)],
                    "image": img,
                }
            )
            idx += 1
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-type", type=int, default=8, help="records per rejection kind")
    ap.add_argument("--asset-dir", default="asset", help="source images (repo's own)")
    args = ap.parse_args()

    if not os.path.isdir(args.asset_dir):
        raise SystemExit(
            f"asset directory not found: {args.asset_dir}\n"
            "Run this from the repository root, or pass --asset-dir."
        )

    img_dir = os.path.join(args.out_dir, "images")
    audio_dir = os.path.join(args.out_dir, "audio")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    for name, _ in IMAGES:
        src = os.path.join(args.asset_dir, name)
        if not os.path.exists(src):
            raise SystemExit(f"missing asset: {src}")
        shutil.copy2(src, os.path.join(img_dir, name))

    # The loader always resolves an audio path even for image-only samples,
    # so keep the directory populated to match make_smoke_data.py.
    for name in ("q1.wav", "q2.wav"):
        src = os.path.join(args.asset_dir, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(audio_dir, name))

    data = build(args.n_per_type)
    out_json = os.path.join(args.out_dir, "dpo_train.json")
    with open(out_json, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

    print(f"wrote {len(data)} preference pairs to {out_json}")
    for kind in ("generic", "wrong", "empty"):
        print(f"  {kind:8} {sum(1 for d in data if d['id'].startswith(kind))}")
    print(f"\nnow run:\n  export VITA_DPO_DATA_DIR={os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
