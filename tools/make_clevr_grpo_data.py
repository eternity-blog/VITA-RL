"""Convert R1-V's CLEVR-70k counting set into GRPO prompt-only records.

This is the verifiable-reward counterpart to make_rlaif_v_grpo_data.py.
RLAIF-V's rewards (keyword overlap / LLM judge against a model-written gold)
are proxies: within a group of 8 decent open-ended descriptions they rank
stylistic variation, so the advantage signal is mostly noise. Counting has a
ground truth. The record ships reward_meta={"answer": "3"} and the binary
`answer` reward (rewards.py) marks each rollout right or wrong -- the
within-group pass/fail variance IS the GRPO signal, exactly the R1 recipe
(R1-V took Qwen2-VL-2B from 48% to 82.5% OOD counting in ~100 steps on this
very dataset).

Source: leonardPKU/clevr_cogen_a_train (HF), parquet with embedded images.
Columns: image (dict with bytes), problem (the counting question), solution
("<answer> 3 </answer>" style).

The prompt appends the R1-style format instruction so the `format` reward
has something to grade and the final answer is machine-extractable.

Usage:
    python tools/make_clevr_grpo_data.py \
        --parquet /path/data/*.parquet \
        --out-dir $WEIGHTS_ROOT/clevr_grpo --heldout 500
"""
import argparse
import glob
import hashlib
import io
import json
import os
import re
import sys

REQUIRED = ("image", "problem", "solution")

# R1-V's instruction, adapted: VITA-1.5 is not a reasoning-tuned model, so
# the tags are taught by the format reward rather than assumed.
FORMAT_SUFFIX = (
    " Output the thinking process in <think> </think> tags and the final "
    "answer (a single number) in <answer> </answer> tags."
)

_ANSWER_TAG = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def gold_from_solution(solution: str) -> str:
    """The verifiable answer out of the solution cell."""
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
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=999999)
    ap.add_argument(
        "--heldout",
        type=int,
        default=500,
        help="records reserved for the held-out accuracy eval "
        "(clevr_grpo_heldout.json, never trained on)",
    )
    args = ap.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is required: pip install --only-binary=:all: pyarrow")
    from PIL import Image

    paths = []
    for p in args.parquet:
        paths.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "audio"), exist_ok=True)

    records = []
    seen_images = {}
    stats = {"read": 0, "empty": 0, "no_answer": 0, "bad_image": 0}

    for path in paths:
        if len(records) >= args.limit:
            break
        pf = pq.ParquetFile(path)
        missing = [c for c in REQUIRED if c not in pf.schema_arrow.names]
        if missing:
            sys.exit(f"{path}: missing column(s) {missing}; found {pf.schema_arrow.names}")

        for batch in pf.iter_batches(batch_size=64):
            if len(records) >= args.limit:
                break
            for row in batch.to_pylist():
                if len(records) >= args.limit:
                    break
                stats["read"] += 1

                problem = (row.get("problem") or "").strip()
                if not problem:
                    stats["empty"] += 1
                    continue
                gold = gold_from_solution(row.get("solution") or "")
                if not gold:
                    stats["no_answer"] += 1
                    continue

                try:
                    raw = _image_bytes(row["image"])
                    digest = hashlib.sha1(raw).hexdigest()
                    fname = seen_images.get(digest)
                    if fname is None:
                        fname = f"{digest}.jpg"
                        dest = os.path.join(img_dir, fname)
                        if not os.path.exists(dest):
                            with Image.open(io.BytesIO(raw)) as im:
                                im.convert("RGB").save(dest, "JPEG", quality=95)
                        seen_images[digest] = fname
                except Exception as exc:  # noqa: BLE001
                    stats["bad_image"] += 1
                    if stats["bad_image"] <= 3:
                        print(f"  skipping unreadable image: {exc}")
                    continue

                records.append(
                    {
                        "set": "clevr_grpo",
                        "id": f"clevr_{len(records):06d}",
                        "conversations": [
                            {
                                "from": "human",
                                "value": "<image>\n" + problem + FORMAT_SUFFIX,
                            },
                        ],
                        "image": fname,
                        "reward_meta": {"answer": gold},
                    }
                )

    if args.heldout >= len(records):
        sys.exit(f"--heldout {args.heldout} >= total records {len(records)}")
    heldout = records[-args.heldout :] if args.heldout else []
    train = records[: len(records) - args.heldout]

    out_json = os.path.join(args.out_dir, "clevr_grpo_train.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(train, fh, ensure_ascii=False, indent=1)
    if heldout:
        held_json = os.path.join(args.out_dir, "clevr_grpo_heldout.json")
        with open(held_json, "w", encoding="utf-8") as fh:
            json.dump(heldout, fh, ensure_ascii=False, indent=1)
        print(f"held out {len(heldout)} -> {held_json}")

    print(f"wrote {len(train)} GRPO prompts -> {out_json}")
    print(f"unique images: {len(seen_images)} (in {img_dir})")
    print("filtered:", ", ".join(f"{k}={v}" for k, v in stats.items()))
    if train:
        from collections import Counter

        dist = Counter(r["reward_meta"]["answer"] for r in train)
        print("answer distribution:", dict(sorted(dist.items(), key=lambda kv: kv[0])))


if __name__ == "__main__":
    main()
