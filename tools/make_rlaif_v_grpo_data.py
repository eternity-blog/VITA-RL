"""Convert RLAIF-V-Dataset into GRPO prompt-only records (image+text).

GRPO generates its own completions and scores them with a reward function,
so unlike the DPO converter (make_rlaif_v_data.py) this one keeps no
chosen/rejected responses -- only the image and the question. The reward
signal has to come from somewhere, though, and for grounded VQA the natural
ground truth is the `chosen` answer: a rollout that mentions more of the
content words the gold answer uses is more likely to be describing the right
thing. So the chosen answer is mined for keywords and shipped in reward_meta
for the `keyword` reward to check.

This is a proxy, not a judge. It rewards surface overlap with the gold
answer, which is exactly right for "name the object in the image" style VQA
and only rough for open-ended description. The `judge` reward (a small
instruct model scoring 1-5, see rewards.py) is the stronger signal when a
judge model is available; this rule-based default needs no extra model and
keeps the per-step cost at zero.

Record shape matches the multimodal GRPO smoke set
(tools/make_grpo_mm_smoke_data.py) so the same --reward_fns spec and the
same GRPOPromptDataset image path handle both.

Usage:
    python tools/make_rlaif_v_grpo_data.py \\
        --parquet shard000.parquet shard001.parquet \\
        --out-dir $WEIGHTS_ROOT/rlaif_v_grpo --limit 8000
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys

REQUIRED = ("image", "question", "chosen")

# A small English stopword set so the keyword reward is not dominated by
# "the", "a", "is". Kept inline rather than depending on nltk.
STOP = frozenset(
    """
a an the is are was were be been being to of in on at for with and or but
not this that these those it its as by from has have had do does did will
would can could should may might must shall you he she they we i me him her
them us my your his their our there here about above below over under into
onto upon among between within without across through during before after
since until while because if though although unless than then so very too
also just only more most some any all each both few many much such no nor
""".split()
)

KEYWORD_CAP = 8


def keywords_from(text):
    """Content words from a gold answer, for the keyword reward.

    Lowercased, stopword- and short-word-filtered, order-preserving dedup,
    capped so one long answer cannot dominate the reward.
    """
    toks = re.findall(r"[A-Za-z]{3,}", text.lower())
    seen, out = set(), []
    for t in toks:
        if t in STOP or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= KEYWORD_CAP:
            break
    return out


def _image_bytes(cell):
    """Pull raw bytes out of a parquet image cell (dict or plain bytes)."""
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
    ap.add_argument(
        "--parquet",
        required=True,
        nargs="+",
        help="one or more RLAIF-V parquet shards",
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--limit",
        type=int,
        default=8000,
        help="stop after this many usable prompts",
    )
    ap.add_argument(
        "--min-chars",
        type=int,
        default=4,
        help="drop questions whose gold answer is shorter than this; a "
        "one-token answer ('Yes') yields no keywords and a degenerate group",
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=600,
        help="drop prompts whose gold answer exceeds this, so a few long "
        "descriptions do not set the rollout length",
    )
    args = ap.parse_args()

    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("pyarrow is required: pip install --only-binary=:all: pyarrow")
    from PIL import Image

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    # GRPOPromptCollator does not touch audio, but the SFT/DPO loaders the
    # repo ships do; keep the sibling dir so the folder is interchangeable.
    os.makedirs(os.path.join(args.out_dir, "audio"), exist_ok=True)

    records = []
    seen_images = {}
    stats = {
        "read": 0,
        "empty": 0,
        "too_short": 0,
        "too_long": 0,
        "no_keywords": 0,
        "bad_image": 0,
    }

    for path in args.parquet:
        if len(records) >= args.limit:
            break
        pf = pq.ParquetFile(path)
        missing = [c for c in REQUIRED if c not in pf.schema_arrow.names]
        if missing:
            sys.exit(
                f"{path}: missing column(s) {missing}; "
                f"found {pf.schema_arrow.names}"
            )

        for batch in pf.iter_batches(batch_size=64):
            if len(records) >= args.limit:
                break
            for row in batch.to_pylist():
                if len(records) >= args.limit:
                    break
                stats["read"] += 1

                question = (row.get("question") or "").strip()
                chosen = (row.get("chosen") or "").strip()
                if not question or not chosen:
                    stats["empty"] += 1
                    continue
                if len(chosen) < args.min_chars:
                    stats["too_short"] += 1
                    continue
                if len(chosen) > args.max_chars:
                    stats["too_long"] += 1
                    continue

                kws = keywords_from(chosen)
                if len(kws) < 1:
                    # No content words to reward on -> the group would be
                    # degenerate (keyword_reward returns 0 for everyone).
                    stats["no_keywords"] += 1
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

                # Length window around the gold answer: reward answers in a
                # 0.6x..1.8x band, decaying outside (see length_reward).
                lo = max(5, int(0.6 * len(chosen)))
                hi = max(lo + 10, min(400, int(1.8 * len(chosen))))

                records.append(
                    {
                        "set": "rlaif_v_grpo",
                        "id": f"grpo_{len(records):06d}",
                        "conversations": [
                            {"from": "human", "value": "<image>\n" + question},
                        ],
                        "image": fname,
                        "reward_meta": {
                            "keywords": kws,
                            "target_len": [lo, hi],
                            "state": "left",
                            # Full gold answer for model-based rewards: the
                            # judge scores against the reference instead of
                            # guessing what a good answer looks like.
                            "gold": chosen,
                        },
                    }
                )

    out_json = os.path.join(args.out_dir, "rlaif_v_grpo_train.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=1)

    print(f"\nwrote {len(records)} GRPO prompts -> {out_json}")
    print(f"unique images: {len(seen_images)} (in {img_dir})")
    print("filtered:", ", ".join(f"{k}={v}" for k, v in stats.items()))
    if records:
        import statistics as st

        nk = [len(r["reward_meta"]["keywords"]) for r in records]
        print(
            f"keywords/prompt: mean={st.mean(nk):.1f} "
            f"median={st.median(nk)} max={max(nk)}"
        )


if __name__ == "__main__":
    main()
