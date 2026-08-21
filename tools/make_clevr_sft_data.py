"""CLEVR counting as SFT: the control arm for the GRPO run.

Reads the same clevr_cogen_a_train parquet shards as make_clevr_grpo_data.py,
applies the *same* filters in the *same* order (so record indices line up with
the GRPO conversion), drops the tail records that became the GRPO held-out
set, and samples N prompts to pair with their gold `solution` as a supervised
target.

The point of the control: GRPO saw N distinct prompts and got 8 one-bit
pass/fail signals per prompt; SFT sees the SAME number of distinct prompts
and gets the full gold answer handed to it. Same data budget, different
supervision channel. The dataset's solutions are direct answers
("<answer> 3 </answer>", no <think> chain), so SFT teaches the answer format
directly -- it cannot teach reasoning it never sees, which is exactly the
asymmetry the comparison is about.

Images are NOT re-extracted: pass --image-dir at the GRPO conversion's
images/ folder and the script just re-derives each record's sha1 filename
and verifies it exists.

Usage:
  python tools/make_clevr_sft_data.py \
      --parquet '/path/clevr_cogen_a_train/data/*.parquet' \
      --image-dir /path/clevr_grpo/images \
      --out-dir /path/clevr_sft --take 6400 --skip-tail 500
"""

import argparse
import glob
import hashlib
import json
import os
import random
import sys

REQUIRED = ("image", "problem", "solution")

# Must stay byte-identical to make_clevr_grpo_data.py so the prompts match
# what the GRPO policy trained on.
FORMAT_SUFFIX = (
    " Output the thinking process in <think> </think> tags and the final "
    "answer (a single number) in <answer> </answer> tags."
)


def _image_bytes(cell):
    if isinstance(cell, dict):
        data = cell.get("bytes")
        if data is None:
            raise ValueError(f"image dict has no 'bytes': keys={list(cell)}")
        return data
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    raise TypeError(f"unsupported image cell type: {type(cell).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, nargs="+")
    ap.add_argument("--image-dir", required=True,
                    help="existing images/ dir from the GRPO conversion")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--take", type=int, default=6400,
                    help="prompts to sample -- match the GRPO run's distinct "
                    "prompt count (steps x effective batch)")
    ap.add_argument("--skip-tail", type=int, default=500,
                    help="records at the end reserved as GRPO held-out; "
                    "never allowed into SFT training")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is required: pip install --only-binary=:all: pyarrow")

    paths = []
    for p in args.parquet:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])

    records = []
    stats = {"read": 0, "empty": 0, "no_solution": 0, "missing_image": 0}

    for path in paths:
        pf = pq.ParquetFile(path)
        missing = [c for c in REQUIRED if c not in pf.schema_arrow.names]
        if missing:
            sys.exit(f"{path}: missing column(s) {missing}; found {pf.schema_arrow.names}")
        for batch in pf.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                stats["read"] += 1
                problem = (row.get("problem") or "").strip()
                if not problem:
                    stats["empty"] += 1
                    continue
                solution = (row.get("solution") or "").strip()
                if not solution:
                    stats["no_solution"] += 1
                    continue
                raw = _image_bytes(row["image"])
                fname = hashlib.sha1(raw).hexdigest() + ".jpg"
                if not os.path.exists(os.path.join(args.image_dir, fname)):
                    stats["missing_image"] += 1
                    continue
                records.append((problem, solution, fname))

    if args.skip_tail:
        records = records[: -args.skip_tail]
    if args.take > len(records):
        sys.exit(f"--take {args.take} > available train records {len(records)}")

    sampled = random.Random(args.seed).sample(records, args.take)

    out = []
    for problem, solution, fname in sampled:
        out.append(
            {
                "set": "clevr_sft",
                "id": f"clevr_sft_{len(out):06d}",
                "conversations": [
                    {"from": "human", "value": "<image>\n" + problem + FORMAT_SUFFIX},
                    {"from": "gpt", "value": solution},
                ],
                "image": fname,
            }
        )

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "audio"), exist_ok=True)
    out_json = os.path.join(args.out_dir, "clevr_sft_train.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    print(f"wrote {len(out)} SFT records -> {out_json}")
    print("filtered:", ", ".join(f"{k}={v}" for k, v in stats.items()))
    print("NOTE: link or copy the GRPO images dir to <out-dir>/images "
          "before training.")


if __name__ == "__main__":
    main()
