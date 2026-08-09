"""Convert RLAIF-V-Dataset parquet shards into the DPO record format.

RLAIF-V (Yu et al., 2024) is open-source AI feedback for multimodal models:
each record is one image, one question, and two responses that a stronger
model judged as better and worse. That is exactly a DPO preference pair, so
the conversion is mostly mechanical -- the work is in the details below.

    https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset

Three things this script has to get right:

1. **Images are embedded, not referenced.** The parquet carries raw bytes per
   row, and the same picture recurs across records. Writing one file per row
   would duplicate them, so images are content-addressed: the SHA-1 of the
   bytes becomes the filename and identical images collapse to one file.

2. **The prompt needs the <image> token.** LazySupervisedDataset splices
   vision features in wherever that literal appears, and a record without it
   silently trains on text alone. Prepended here rather than trusted to the
   source question.

3. **Degenerate pairs are dropped.** A pair whose two responses are identical
   after normalisation contributes a DPO logit of exactly zero -- no gradient,
   just wasted forward passes. Same for empty responses.

Usage:
    python tools/make_rlaif_v_data.py \
        --parquet /path/to/RLAIF-V-Dataset_000.parquet \
        --out-dir /path/to/rlaif_v_dpo \
        --limit 2000
"""
import argparse
import hashlib
import io
import json
import os
import sys

# Columns as published on the dataset card. Checked explicitly so a schema
# change surfaces as a clear error rather than a KeyError deep in the loop.
REQUIRED = ("image", "question", "chosen", "rejected")


def _image_bytes(cell):
    """Pull raw bytes out of a parquet image cell.

    HuggingFace writes an Image feature as {'bytes': ..., 'path': ...}, but
    a plain bytes column shows up in some exports. Accept both.
    """
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
    ap.add_argument("--parquet", required=True, nargs="+",
                    help="one or more RLAIF-V parquet shards")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=2000,
                    help="stop after this many usable pairs")
    ap.add_argument("--min-chars", type=int, default=8,
                    help="drop responses shorter than this")
    ap.add_argument("--max-chars", type=int, default=1200,
                    help="drop pairs whose chosen response exceeds this, so a "
                         "few very long samples do not set the sequence length "
                         "for the whole run")
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

    records = []
    seen_images = {}
    stats = {"read": 0, "no_image": 0, "identical": 0, "too_short": 0,
             "too_long": 0, "bad_image": 0}

    for path in args.parquet:
        if len(records) >= args.limit:
            break
        pf = pq.ParquetFile(path)
        missing = [c for c in REQUIRED if c not in pf.schema_arrow.names]
        if missing:
            sys.exit(f"{path}: missing column(s) {missing}; "
                     f"found {pf.schema_arrow.names}")

        for batch in pf.iter_batches(batch_size=64):
            if len(records) >= args.limit:
                break
            for row in batch.to_pylist():
                if len(records) >= args.limit:
                    break
                stats["read"] += 1

                question = (row.get("question") or "").strip()
                chosen = (row.get("chosen") or "").strip()
                rejected = (row.get("rejected") or "").strip()

                if not question or not chosen or not rejected:
                    stats["no_image"] += 1
                    continue
                # A pair the objective cannot separate: policy_delta and
                # ref_delta cancel, the loss is exactly -log(0.5) forever.
                if chosen == rejected:
                    stats["identical"] += 1
                    continue
                if min(len(chosen), len(rejected)) < args.min_chars:
                    stats["too_short"] += 1
                    continue
                if len(chosen) > args.max_chars or len(rejected) > args.max_chars:
                    stats["too_long"] += 1
                    continue

                try:
                    raw = _image_bytes(row["image"])
                    digest = hashlib.sha1(raw).hexdigest()
                    fname = seen_images.get(digest)
                    if fname is None:
                        fname = f"{digest}.jpg"
                        dest = os.path.join(img_dir, fname)
                        if not os.path.exists(dest):
                            # Normalise to RGB JPEG: the source mixes PNG and
                            # RGBA, and the vision tower wants three channels.
                            with Image.open(io.BytesIO(raw)) as im:
                                im.convert("RGB").save(dest, "JPEG", quality=95)
                        seen_images[digest] = fname
                except Exception as exc:  # noqa: BLE001 - report and skip
                    stats["bad_image"] += 1
                    if stats["bad_image"] <= 3:
                        print(f"  skipping unreadable image: {exc}")
                    continue

                records.append({
                    "set": "rlaif_v",
                    "id": f"rlaif_{len(records):06d}",
                    "conversations": [
                        # The <image> token is what splices vision features
                        # into the sequence; without it this trains text-only.
                        {"from": "human", "value": "<image>\n" + question},
                        {"from": "gpt", "value": chosen},
                    ],
                    "rejected": rejected,
                    "image": fname,
                })

    out_json = os.path.join(args.out_dir, "rlaif_v_train.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=1)

    print(f"\nwrote {len(records)} pairs -> {out_json}")
    print(f"unique images: {len(seen_images)} (in {img_dir})")
    print("filtered:", ", ".join(f"{k}={v}" for k, v in stats.items()))
    if records:
        avg_c = sum(len(r["conversations"][1]["value"]) for r in records) / len(records)
        avg_r = sum(len(r["rejected"]) for r in records) / len(records)
        print(f"mean chars: chosen={avg_c:.0f} rejected={avg_r:.0f}")


if __name__ == "__main__":
    main()
