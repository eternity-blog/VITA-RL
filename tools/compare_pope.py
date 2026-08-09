"""Compare POPE runs on the hallucination axis specifically.

POPE asks "is object X in the image?" and most of its questions are about
objects that are not there. The failure mode it exists to measure is the
model saying Yes anyway -- a false positive on an absent object.

VLMEvalKit's score CSV reports F1, accuracy, precision and recall, none of
which is that rate directly. This reads the per-question auxmatch sheet and
computes it, plus a McNemar test on the questions where two runs disagree,
which is the right test for two models scored on the same items: it ignores
the questions both got right or both got wrong and asks whether the flips
are lopsided.

The numbers here will not match the score CSV, and the difference is not a
bug in either. POPE tags a question with one or more of random / popular /
adversarial, and VLMEvalKit explodes those tags into separate rows before
computing the overall figures -- so a question tagged with all three counts
three times. That is defensible for a per-split table but it weights
questions unevenly in the aggregate. This script counts each of the 5127
questions once, which is the denominator a hallucination rate needs.

Usage:
    python tools/compare_pope.py \
        --before .../baseline/vita_qwen2 \
        --after  .../dpo/vita_qwen2 [--label-after "DPO lr5e-6"]
"""
import argparse
import glob
import math
import os
import sys


def _load(run_dir):
    import pandas as pd
    hits = glob.glob(os.path.join(run_dir, "*POPE_auxmatch.xlsx"))
    if not hits:
        sys.exit(f"no *POPE_auxmatch.xlsx in {run_dir}")
    df = pd.read_excel(hits[0])
    need = {"answer", "extracted", "index"}
    if not need.issubset(df.columns):
        sys.exit(f"{hits[0]} lacks {need - set(df.columns)}")
    return df.set_index("index").sort_index()


def _rates(df):
    yes_gt, yes_pred = df.answer == "Yes", df.extracted == "Yes"
    tp = int((yes_gt & yes_pred).sum())
    fp = int((~yes_gt & yes_pred).sum())
    fn = int((yes_gt & ~yes_pred).sum())
    tn = int((~yes_gt & ~yes_pred).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        # The headline: how often the model claims an absent object is present.
        "halluc_rate": fp / (fp + tn) if fp + tn else 0.0,
        "precision": prec, "recall": rec, "f1": f1,
        "acc": (tp + tn) / max(len(df), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--label-before", default="before")
    ap.add_argument("--label-after", default="after")
    args = ap.parse_args()

    b_df, a_df = _load(args.before), _load(args.after)
    shared = b_df.index.intersection(a_df.index)
    if len(shared) != len(b_df) or len(shared) != len(a_df):
        print(f"note: comparing {len(shared)} shared questions "
              f"({len(b_df)} before, {len(a_df)} after)")
    b_df, a_df = b_df.loc[shared], a_df.loc[shared]

    b, a = _rates(b_df), _rates(a_df)
    lb, la = args.label_before, args.label_after

    print(f"\n{'metric':<22}{lb:>14}{la:>14}{'delta':>12}")
    print("-" * 62)
    for key, name, pct in [
        ("halluc_rate", "hallucination rate", True),
        ("precision", "precision", True),
        ("recall", "recall", True),
        ("f1", "F1", True),
        ("acc", "accuracy", True),
    ]:
        scale = 100 if pct else 1
        print(f"{name:<22}{b[key]*scale:>14.2f}{a[key]*scale:>14.2f}"
              f"{(a[key]-b[key])*scale:>+12.2f}")
    print(f"\n{'false Yes (absent obj)':<22}{b['fp']:>14d}{a['fp']:>14d}{a['fp']-b['fp']:>+12d}")
    print(f"{'missed Yes (present)':<22}{b['fn']:>14d}{a['fn']:>14d}{a['fn']-b['fn']:>+12d}")

    # McNemar on the discordant pairs. b01 = before wrong / after right,
    # b10 = before right / after wrong. Under the null the two are equally
    # likely, so a binomial test on b01 out of (b01 + b10) is exact.
    b_ok = (b_df.answer == b_df.extracted).values
    a_ok = (a_df.answer == a_df.extracted).values
    b01 = int((~b_ok & a_ok).sum())
    b10 = int((b_ok & ~a_ok).sum())
    n = b01 + b10
    print(f"\ndisagreements: {n} of {len(shared)} questions")
    print(f"  after fixed:  {b01}")
    print(f"  after broke:  {b10}")
    if n:
        # Exact two-sided binomial p at q=0.5.
        tail = sum(math.comb(n, k) for k in range(0, min(b01, b10) + 1))
        pv = min(1.0, 2 * tail / (2 ** n))
        print(f"  McNemar exact two-sided p = {pv:.4f}"
              f"  -> {'significant' if pv < 0.05 else 'not significant'}")
    else:
        print("  the two runs answered every question identically")


if __name__ == "__main__":
    main()
