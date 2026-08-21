"""SuperCLEVR test200 -> GRPO-eval-format json (OOD counting eval).

R1-V's out-of-distribution eval set: 200 counting questions over SuperCLEVR
images (vehicles with parts/textures -- a visual domain CLEVR training never
shows). Source: the jigsaw-r1/super_clevr HF dataset, which packages the same
200 problems R1-V ships as superclevr_test200_counting_problems.jsonl.

Output records use the same shape as make_clevr_grpo_data.py so
tools/eval_grpo_heldout.py consumes them unchanged (reward_meta["answer"]
drives answer_accuracy). The prompt gets the same FORMAT_SUFFIX the models
were trained/evaluated with in-distribution -- the instruction is held
constant, only the image domain shifts.

Usage:
  python tools/make_superclevr_eval_data.py \
      --parquet /path/superclevr_raw/test.parquet --out-dir /path/superclevr_eval
"""

import argparse
import io
import json
import os
import sys

REQUIRED = ("image", "question", "ground_truth")

# Byte-identical to make_clevr_grpo_data.py.
FORMAT_SUFFIX = (
    " Output the thinking process in <think> </think> tags and the final "
    "answer (a single number) in <answer> </answer> tags."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is required: pip install --only-binary=:all: pyarrow")
    from PIL import Image

    pf = pq.ParquetFile(args.parquet)
    missing = [c for c in REQUIRED if c not in pf.schema_arrow.names]
    if missing:
        sys.exit(f"missing column(s) {missing}; found {pf.schema_arrow.names}")

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    records = []
    for batch in pf.iter_batches(batch_size=32):
        for row in batch.to_pylist():
            question = (row.get("question") or "").strip()
            gold = str(row.get("ground_truth") or "").strip()
            if not question or not gold:
                continue
            raw = row["image"]
            if isinstance(raw, dict):
                raw = raw["bytes"]
            base = os.path.splitext(str(row.get("image_id") or f"sc_{len(records):04d}"))[0]
            fname = base + ".jpg"
            with Image.open(io.BytesIO(raw)) as im:
                im.convert("RGB").save(os.path.join(img_dir, fname), "JPEG", quality=95)
            records.append(
                {
                    "set": "superclevr",
                    "id": f"superclevr_{len(records):04d}",
                    "conversations": [
                        {"from": "human", "value": "<image>\n" + question + FORMAT_SUFFIX},
                    ],
                    "image": fname,
                    "reward_meta": {"answer": gold},
                }
            )

    out_json = os.path.join(args.out_dir, "superclevr_test.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=1)

    from collections import Counter

    dist = Counter(r["reward_meta"]["answer"] for r in records)
    print(f"wrote {len(records)} OOD eval prompts -> {out_json}")
    print("answer distribution:", dict(sorted(dist.items(), key=lambda kv: kv[0])))


if __name__ == "__main__":
    main()
