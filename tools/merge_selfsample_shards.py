"""Merge sharded self-sampling output into one dataset directory.

tools/make_selfsample_data.py runs one worker per GPU, each writing its own
out-dir. This concatenates them: records are renumbered, images are hard-linked
into a single folder (same inode, so no extra disk), and duplicate image names
collide harmlessly because they are already content-addressed by SHA-1.

Usage:
    python tools/merge_selfsample_shards.py \\
        --shards out/w0 out/w1 ... --out out/merged
"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img_out = os.path.join(args.out, "images")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "audio"), exist_ok=True)

    merged, linked, skipped = [], 0, 0
    for shard in args.shards:
        jpath = os.path.join(shard, "rlaif_v_train.json")
        if not os.path.exists(jpath):
            print(f"  skip {shard}: no rlaif_v_train.json")
            continue
        recs = json.load(open(jpath))
        for r in recs:
            src = os.path.join(shard, "images", r["image"])
            dst = os.path.join(img_out, r["image"])
            if not os.path.exists(dst):
                if not os.path.exists(src):
                    skipped += 1
                    continue
                try:
                    # Hard link rather than copy: the file is identical and
                    # this dataset is mostly image bytes.
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
                linked += 1
            r["id"] = f"ss_{len(merged):06d}"
            merged.append(r)
        print(f"  {shard}: +{len(recs)} records")

    out_json = os.path.join(args.out, "rlaif_v_train.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, ensure_ascii=False, indent=1)

    print(f"\nmerged {len(merged)} pairs -> {out_json}")
    print(f"unique images: {linked}" + (f"  ({skipped} records dropped, image missing)" if skipped else ""))
    if merged:
        import statistics as st
        cl = [len(r["conversations"][1]["value"]) for r in merged]
        rl = [len(r["rejected"]) for r in merged]
        longer = sum(1 for a, b in zip(cl, rl) if a > b)
        sc = [r["sim_chosen"] for r in merged if "sim_chosen" in r]
        sr = [r["sim_rejected"] for r in merged if "sim_rejected" in r]
        print(f"mean chars: chosen={st.mean(cl):.0f} rejected={st.mean(rl):.0f}")
        print(f"chosen longer in {longer/len(merged):.1%} (50% = no length bias)")
        if sc:
            print(f"similarity to reference: chosen={st.mean(sc):.3f} "
                  f"rejected={st.mean(sr):.3f}  gap={st.mean(sc)-st.mean(sr):.3f}")


if __name__ == "__main__":
    main()
