"""Stage-2 data for the SFT-vs-GRPO continuation experiment (R5).

Draws ONE sample of fresh prompts and writes it in BOTH formats, so the two
continuation arms train on literally the same records and differ only in the
supervision channel:

  <out-root>/clevr_sft_s2/clevr_sft_train.json    gold solution as SFT target
  <out-root>/clevr_grpo_s2/clevr_grpo_train.json  prompt-only + reward_meta

Sampling is made disjoint from both the stage-1 SFT sample and the GRPO
held-out set by reconstruction: the script re-reads the parquets with the
same filters as make_clevr_grpo_data.py / make_clevr_sft_data.py, re-derives
the stage-1 sample (same seed 42 over the same train slice), removes it and
the held-out tail, then samples the stage-2 records with a different seed.

Images are reused from the original GRPO conversion (sha1 filenames);
symlink or copy that images/ dir into each output dir before training.

Usage:
  python tools/make_clevr_stage2_data.py \
      --parquet '/path/clevr_cogen_a_train/data/*.parquet' \
      --image-dir /path/clevr_grpo/images \
      --out-root /path \
      --take 6400 --stage1-take 6400 --skip-tail 500
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys

REQUIRED = ("image", "problem", "solution")

# Byte-identical to make_clevr_grpo_data.py.
FORMAT_SUFFIX = (
    " Output the thinking process in <think> </think> tags and the final "
    "answer (a single number) in <answer> </answer> tags."
)

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def gold_from_solution(solution: str) -> str:
    m = _ANSWER_TAG.search(solution or "")
    text = m.group(1).strip() if m else (solution or "").strip()
    nums = _NUMBER.findall(text)
    return nums[-1] if nums else text


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
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--take", type=int, default=6400)
    ap.add_argument("--stage1-take", type=int, default=6400,
                    help="size of the stage-1 SFT sample to reconstruct and exclude")
    ap.add_argument("--stage1-seed", type=int, default=42)
    ap.add_argument("--skip-tail", type=int, default=500)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is required: pip install --only-binary=:all: pyarrow")

    paths = []
    for p in args.parquet:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])

    records = []
    for path in paths:
        pf = pq.ParquetFile(path)
        missing = [c for c in REQUIRED if c not in pf.schema_arrow.names]
        if missing:
            sys.exit(f"{path}: missing column(s) {missing}")
        for batch in pf.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                problem = (row.get("problem") or "").strip()
                if not problem:
                    continue
                solution = (row.get("solution") or "").strip()
                if not solution:
                    continue
                raw = _image_bytes(row["image"])
                fname = hashlib.sha1(raw).hexdigest() + ".jpg"
                if not os.path.exists(os.path.join(args.image_dir, fname)):
                    continue
                records.append((problem, solution, fname))

    train = records[: -args.skip_tail] if args.skip_tail else records

    # Reconstruct the stage-1 sample exactly (same seed, same slice, same
    # sampling call as make_clevr_sft_data.py) and exclude it by identity.
    stage1 = random.Random(args.stage1_seed).sample(train, args.stage1_take)
    stage1_keys = {(p, f) for p, _, f in stage1}
    pool = [r for r in train if (r[0], r[2]) not in stage1_keys]
    print(f"train={len(train)}  stage1 excluded={len(train) - len(pool)}  pool={len(pool)}")

    if args.take > len(pool):
        sys.exit(f"--take {args.take} > remaining pool {len(pool)}")
    sampled = random.Random(args.seed).sample(pool, args.take)

    sft_dir = os.path.join(args.out_root, "clevr_sft_s2")
    grpo_dir = os.path.join(args.out_root, "clevr_grpo_s2")
    for d in (sft_dir, grpo_dir):
        os.makedirs(os.path.join(d, "audio"), exist_ok=True)

    sft_out, grpo_out = [], []
    for problem, solution, fname in sampled:
        human = {"from": "human", "value": "<image>\n" + problem + FORMAT_SUFFIX}
        sft_out.append(
            {
                "set": "clevr_sft",
                "id": f"clevr_sft_s2_{len(sft_out):06d}",
                "conversations": [human, {"from": "gpt", "value": solution}],
                "image": fname,
            }
        )
        grpo_out.append(
            {
                "set": "clevr_grpo",
                "id": f"clevr_s2_{len(grpo_out):06d}",
                "conversations": [human],
                "image": fname,
                "reward_meta": {"answer": gold_from_solution(solution)},
            }
        )

    sft_json = os.path.join(sft_dir, "clevr_sft_train.json")
    grpo_json = os.path.join(grpo_dir, "clevr_grpo_train.json")
    with open(sft_json, "w", encoding="utf-8") as fh:
        json.dump(sft_out, fh, ensure_ascii=False, indent=1)
    with open(grpo_json, "w", encoding="utf-8") as fh:
        json.dump(grpo_out, fh, ensure_ascii=False, indent=1)

    print(f"wrote {len(sft_out)} SFT records  -> {sft_json}")
    print(f"wrote {len(grpo_out)} GRPO records -> {grpo_json}")
    print("NOTE: link the original images/ dir into both output dirs.")


if __name__ == "__main__":
    main()
