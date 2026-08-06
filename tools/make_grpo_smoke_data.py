#!/usr/bin/env python
"""Generate synthetic text-only prompts to smoke-test the GRPO pipeline.

Not real training data. Each prompt ships the reward_meta its rules need --
keywords the answer should mention and a sensible length window -- so the
reward function can grade a rollout without any human labels.

The prompts are chosen so that sampling produces a spread of quality. That
matters more than it sounds: GRPO's advantage is (r - mean) / std within a
group, so if every rollout for a prompt scores identically the group is
degenerate and contributes no gradient. Open-ended questions with a few
checkable keywords give the reward something to rank.

    python tools/make_grpo_smoke_data.py --out-dir /path/to/grpo_smoke_data
    export VITA_GRPO_DATA_DIR=/path/to/grpo_smoke_data
"""
import argparse
import json
import os

# (question, keywords the answer ought to mention, [min_len, max_len])
PROMPTS = [
    ("Describe a cat in one sentence.", ["cat"], [20, 120]),
    ("What is the capital of France, and one thing it is known for?", ["Paris"], [20, 150]),
    ("Explain what a neural network is, briefly.", ["network", "learn"], [40, 200]),
    ("Name three primary colours.", ["red", "blue"], [10, 100]),
    ("What does the sun provide to Earth?", ["light", "heat"], [20, 150]),
    ("Describe the ocean in one sentence.", ["water"], [20, 120]),
    ("What is photosynthesis?", ["plant", "light"], [30, 180]),
    ("Explain why the sky appears blue.", ["light", "blue"], [30, 180]),
    ("What is a computer used for?", ["computer"], [20, 150]),
    ("Describe winter weather.", ["cold"], [20, 120]),
    ("What do birds use their wings for?", ["fly"], [15, 120]),
    ("Explain what water is made of.", ["hydrogen", "oxygen"], [20, 150]),
]


def build(repeat: int):
    data = []
    for r in range(repeat):
        for i, (question, keywords, length) in enumerate(PROMPTS):
            data.append(
                {
                    "set": "grpo_smoke",
                    "id": f"p_{r * len(PROMPTS) + i:04d}",
                    "conversations": [{"from": "human", "value": question}],
                    "reward_meta": {
                        "keywords": keywords,
                        "target_len": length,
                        # VITA prefixes text queries with the 'left' token.
                        "state": "left",
                    },
                }
            )
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repeat", type=int, default=1, help="passes over the prompt list")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = build(args.repeat)
    out = os.path.join(args.out_dir, "grpo_train.json")
    with open(out, "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)

    print(f"wrote {len(data)} prompts to {out}")
    print(f"\nnow run:\n  export VITA_GRPO_DATA_DIR={os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
