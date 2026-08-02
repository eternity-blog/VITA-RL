#!/usr/bin/env python
"""Generate a tiny synthetic training set to smoke-test the training pipeline.

This is NOT real training data. It exists only to exercise the code path
end-to-end: dataset loading, multimodal token expansion, the collator, and a
few optimizer steps. Images and audio are the repository's own assets.

The generated set covers the three sample shapes the data loader branches on
in vita/util/data_utils_video_audio_neg_patch.py:

  1. image only            -> "image" present, "audio" absent
  2. image + audio         -> both present (the main VITA-1.5 case)
  3. image + audio + negative sample, via "inserted_id"

Usage:
    python tools/make_smoke_data.py --out-dir /path/to/smoke_data
"""
import argparse
import json
import os

# Descriptions are deliberately short; content quality is irrelevant for a
# smoke test, only the shape of the data matters.
IMAGES = [
    ("vita_newlog.jpg", "A blue logo with the text 'Open-Source Interactive MLLM' beneath it."),
    ("vita_15_audio_2.jpg", "A table comparing ASR word error rates across several speech models."),
    ("vita_mllm_performance.png", "A chart of benchmark scores across multimodal evaluation suites."),
    ("vita_15_audio_training.png", "A diagram showing how adding audio affects image understanding."),
]

AUDIO = ["q1.wav", "q2.wav"]


def build(n_per_type: int):
    data = []
    idx = 0

    # 1. image-only samples (text query about an image)
    for i in range(n_per_type):
        img, desc = IMAGES[i % len(IMAGES)]
        data.append({
            "set": "smoke",
            "id": f"img_{idx:04d}",
            "conversations": [
                {"from": "human", "value": "<image>\nDescribe this image briefly."},
                {"from": "gpt", "value": desc},
            ],
            "image": img,
        })
        idx += 1

    # 2. image + audio samples (spoken query about an image)
    for i in range(n_per_type):
        img, desc = IMAGES[i % len(IMAGES)]
        data.append({
            "set": "smoke",
            "id": f"imgaud_{idx:04d}",
            "conversations": [
                {"from": "human", "value": "<image>\n<audio>\n"},
                {"from": "gpt", "value": desc},
            ],
            "image": img,
            "audio": [AUDIO[i % len(AUDIO)]],
        })
        idx += 1

    # 3. image + audio with a negative (noisy-audio) turn.
    #    "inserted_id" indexes the gpt turn that should be marked with the
    #    negative state token; preprocess_multimodal asserts that turn is
    #    from "gpt", so it must be an odd index in this human/gpt layout.
    for i in range(n_per_type):
        img, desc = IMAGES[i % len(IMAGES)]
        data.append({
            "set": "smoke",
            "id": f"neg_{idx:04d}",
            "conversations": [
                {"from": "human", "value": "<image>\n<audio>\n"},
                {"from": "gpt", "value": desc},
                {"from": "human", "value": "<audio>\n"},
                {"from": "gpt", "value": "Sorry, I did not catch that."},
            ],
            "image": img,
            "audio": [AUDIO[0], AUDIO[1]],
            "inserted_id": 3,
        })
        idx += 1

    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="directory holding images/ and audio/")
    ap.add_argument("--n-per-type", type=int, default=8)
    args = ap.parse_args()

    data = build(args.n_per_type)
    out = os.path.join(args.out_dir, "smoke_train.json")
    with open(out, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"wrote {len(data)} samples to {out}")
    for kind in ("img_", "imgaud_", "neg_"):
        print(f"  {kind:9s} {sum(1 for d in data if d['id'].startswith(kind))}")


if __name__ == "__main__":
    main()
