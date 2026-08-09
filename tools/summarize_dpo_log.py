"""Summarise a DPO training log into a trend rather than a wall of scalars.

Every step prints rewards/margin and rewards/accuracy, but on real preference
data the per-step values swing between positive and negative -- batch size is
one pair, so a single hard example flips the sign. Reading the raw log gives
no sense of whether training is working.

This bins the run and prints the mean per bin, which is the thing to look at:
accuracy should climb from 0.5 toward something above it, and the margin
should trend positive. Neither will look like the synthetic smoke test, where
the two responses are trivially separable and accuracy hits 1.0 immediately.

Usage:
    python tools/summarize_dpo_log.py /path/to/dpo-rlaif-v/log.txt [--bins 10]
"""
import argparse
import ast
import re
import sys

# The trainer logs two kinds of dict line: the HF one carrying 'loss' and the
# custom one carrying the DPO diagnostics. Both are Python dict literals.
DICT_RE = re.compile(r"\{'(?:loss|rewards/chosen)'.*?\}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    losses, margins, accs = [], [], []
    with open(args.log, encoding="utf-8", errors="replace") as fh:
        # tqdm writes carriage returns, so a "line" can hold many records.
        for chunk in fh.read().split("\r"):
            for m in DICT_RE.finditer(chunk):
                try:
                    d = ast.literal_eval(m.group(0))
                except (ValueError, SyntaxError):
                    continue
                if "loss" in d:
                    losses.append(d["loss"])
                if "rewards/margin" in d:
                    margins.append(d["rewards/margin"])
                    accs.append(d["rewards/accuracy"])

    if not margins:
        sys.exit(f"no DPO metric lines found in {args.log}")

    print(f"optimizer steps logged: {len(losses)}")
    print(f"metric records:         {len(margins)}")

    def bin_report(name, xs, fmt="{:+.4f}"):
        n = len(xs)
        if n < args.bins:
            return
        size = n // args.bins
        print(f"\n{name} over {args.bins} bins of {size}:")
        for i in range(args.bins):
            part = xs[i * size:(i + 1) * size] if i < args.bins - 1 else xs[i * size:]
            mean = sum(part) / len(part)
            bar = "#" * max(0, int(abs(mean) * (40 if max(map(abs, xs)) < 2 else 2)))
            print(f"  [{i*size:>5}-{i*size+len(part):>5}] " + fmt.format(mean) + f"  {bar}")

    bin_report("loss", losses, "{:.4f}")
    bin_report("rewards/margin", margins)
    bin_report("rewards/accuracy", accs, "{:.3f}")

    half = len(margins) // 2
    print(f"\nfirst half vs second half:")
    print(f"  margin   {sum(margins[:half])/half:+.4f} -> {sum(margins[half:])/(len(margins)-half):+.4f}")
    print(f"  accuracy {sum(accs[:half])/half:.3f} -> {sum(accs[half:])/(len(accs)-half):.3f}")
    if losses:
        h = len(losses) // 2
        print(f"  loss     {sum(losses[:h])/h:.4f} -> {sum(losses[h:])/(len(losses)-h):.4f}")


if __name__ == "__main__":
    main()
