"""Convert RLAIF-V into an SFT dataset from the `chosen` responses alone.

Why this exists
---------------
Three rounds of DPO on RLAIF-V moved nothing, and a fourth round of on-policy
self-sampling did not raise separability either (54.8% against 53.6%). The
binding constraint turned out to be the *judge*: deciding which of two
responses is better needs a signal this pipeline does not have, and a
token-overlap proxy inverts about one pair in ten.

SFT sidesteps the problem entirely. It needs no ranking -- only good answers.
So the same data becomes usable by throwing the `rejected` side away and
training on `chosen` as an ordinary supervised target.

There is a real reason to expect this to work where DPO did not. The probe
says the base model cannot separate these pairs, which is another way of
saying it does not model this distribution well. SFT raises in-distribution
capability directly rather than trying to sharpen a preference the model does
not yet hold. If it works, a later DPO round starts from a base that finds the
same pairs easier -- separability is a property of the model, not just of the
data.

What this deliberately does not do
----------------------------------
No benchmark data is touched. The images and questions come from RLAIF-V,
whose sources are VQA *train* splits (VQAv2, OK-VQA, GQA, TextVQA, COCO), so
training on them leaves MME / MMStar / MMBench / AI2D / POPE / HallusionBench
uncontaminated. That matters because every before/after number in this project
depends on those six being untouched.

Usage:
    python tools/make_rlaif_v_sft_data.py \\
        --parquet shard000.parquet shard001.parquet \\
        --out-dir out/rlaif_v_sft --limit 20000
"""
import argparse
import hashlib
import io
import json
import os
import sys

REQUIRED = ("image", "question", "chosen")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--min-chars", type=int, default=16,
                    help="drop targets shorter than this; a two-word answer "
                         "teaches length, not content")
    ap.add_argument("--max-chars", type=int, default=1200,
                    help="drop very long targets so a handful of outliers do "
                         "not set the sequence length for the whole run")
    args = ap.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is required: pip install --only-binary=:all: pyarrow")
    from PIL import Image

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    # LazySupervisedDataset joins AudioFolder with a literal "audio" segment
    # even when no record has audio, and errors if the directory is missing.
    os.makedirs(os.path.join(args.out_dir, "audio"), exist_ok=True)

    records, seen = [], {}
    stats = {"read": 0, "empty": 0, "too_short": 0, "too_long": 0, "bad_image": 0}

    for path in args.parquet:
        if len(records) >= args.limit:
            break
        pf = pq.ParquetFile(path)
        missing = [c for c in REQUIRED if c not in pf.schema_arrow.names]
        if missing:
            sys.exit(f"{path}: missing column(s) {missing}")

        for batch in pf.iter_batches(batch_size=64):
            if len(records) >= args.limit:
                break
            for row in batch.to_pylist():
                if len(records) >= args.limit:
                    break
                stats["read"] += 1

                question = (row.get("question") or "").strip()
                answer = (row.get("chosen") or "").strip()
                if not question or not answer:
                    stats["empty"] += 1
                    continue
                if len(answer) < args.min_chars:
                    stats["too_short"] += 1
                    continue
                if len(answer) > args.max_chars:
                    stats["too_long"] += 1
                    continue

                try:
                    cell = row["image"]
                    raw = cell["bytes"] if isinstance(cell, dict) else cell
                    digest = hashlib.sha1(raw).hexdigest()
                    fname = seen.get(digest)
                    if fname is None:
                        fname = f"{digest}.jpg"
                        dest = os.path.join(img_dir, fname)
                        if not os.path.exists(dest):
                            with Image.open(io.BytesIO(raw)) as im:
                                im.convert("RGB").save(dest, "JPEG", quality=95)
                        seen[digest] = fname
                except Exception as exc:  # noqa: BLE001
                    stats["bad_image"] += 1
                    if stats["bad_image"] <= 3:
                        print(f"  skipping unreadable image: {exc}")
                    continue

                records.append({
                    "set": "rlaif_v_sft",
                    "id": f"sft_{len(records):06d}",
                    "conversations": [
                        # The <image> token is what splices vision features
                        # into the sequence; without it this trains text-only.
                        {"from": "human", "value": "<image>\n" + question},
                        {"from": "gpt", "value": answer},
                    ],
                    "image": fname,
                })

    out_json = os.path.join(args.out_dir, "rlaif_v_sft_train.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=1)

    print(f"\nwrote {len(records)} samples -> {out_json}")
    print(f"unique images: {len(seen)} (in {img_dir})")
    print("filtered:", ", ".join(f"{k}={v}" for k, v in stats.items()))
    if records:
        import statistics as st
        lens = [len(r["conversations"][1]["value"]) for r in records]
        print(f"target length: mean={st.mean(lens):.0f} "
              f"median={st.median(lens):.0f} max={max(lens)}")


if __name__ == "__main__":
    main()
