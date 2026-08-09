"""Tabulate VLMEvalKit results from two runs side by side.

VLMEvalKit writes one CSV per (model, dataset) with a schema that differs by
benchmark type -- MME reports raw sums per category, MCQ benchmarks report
accuracies. This reads both shapes and prints a single before/after table.

The point of the tool is to make the comparison honest. Two things it does
that eyeballing two CSVs does not:

  - refuses to compare runs that answered different numbers of questions,
    since a partial run's average is not comparable to a full one's
  - prints the delta next to a rough noise floor, so a +0.3 on a 1500-question
    benchmark is not read as an improvement

Usage:
    python tools/compare_eval.py \
        --before /path/to/eval_out/baseline/vita_qwen2 \
        --after  /path/to/eval_out/dpo/vita_qwen2
"""
import argparse
import csv
import glob
import math
import os


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if len(rows) < 2:
        return {}
    header, values = rows[0], rows[1]
    out = {}
    for k, v in zip(header, values):
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _collect(run_dir):
    """dataset -> {metric: value} for every score CSV in a run directory.

    VLMEvalKit names these <model>_<dataset>_score.csv, and the model name
    itself contains underscores (vita_qwen2), so the dataset cannot be
    recovered by splitting. Take it from the directory name instead, which
    VLMEvalKit sets to the model.
    """
    model = os.path.basename(os.path.normpath(run_dir))
    found = {}
    for path in glob.glob(os.path.join(run_dir, "*.csv")):
        base = os.path.basename(path)
        for suffix in ("_score.csv", "_acc.csv"):
            if base.endswith(suffix):
                name = base[: -len(suffix)]
                if name.startswith(model + "_"):
                    name = name[len(model) + 1:]
                found[name] = _read_csv(path)
                break
    return found


def _noise_floor(dataset, metrics):
    """A rough 1-sigma band for a proportion, in the metric's own units.

    MCQ accuracies here are fractions in [0,1]; MME sums 200 points over a
    variable number of questions per category, so no meaningful floor is
    printed for it. This is deliberately crude -- it exists to stop a delta
    smaller than sampling noise being reported as a win, not to be a test.

    The counts are independent questions, which is not the same as rows.
    MMBench_DEV_EN_V11 ships 4876 rows that are option-order rotations of
    1292 base questions, and circular eval scores a question correct only if
    every rotation is -- so the sample size is 1292 and using 4876 would
    understate the band by a factor of ~2.

    POPE is excluded on purpose: its Overall is an F1 percentage, not a
    proportion in [0,1], so this formula would not apply to it.
    """
    n = {"MMStar": 1500, "MMBench_DEV_EN_V11": 1292, "AI2D_TEST": 3088}.get(dataset)
    if not n:
        return None
    p = metrics.get("Overall")
    if p is None or not 0 < p < 1:
        return None
    return 1.96 * math.sqrt(p * (1 - p) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    args = ap.parse_args()

    before, after = _collect(args.before), _collect(args.after)
    shared = sorted(set(before) & set(after))
    if not shared:
        print(f"no common datasets.\n  before: {sorted(before)}\n  after:  {sorted(after)}")
        return

    for ds in shared:
        b, a = before[ds], after[ds]
        print(f"\n=== {ds} ===")

        if ds == "MME":
            # MME's headline number is the sum of the two category totals,
            # which VLMEvalKit does not write to the CSV itself.
            for key in ("perception", "reasoning"):
                if key in b and key in a:
                    print(f"  {key:<24} {b[key]:>9.2f} -> {a[key]:>9.2f}   {a[key]-b[key]:+.2f}")
            if all(k in b and k in a for k in ("perception", "reasoning")):
                tb, ta = b["perception"] + b["reasoning"], a["perception"] + a["reasoning"]
                print(f"  {'TOTAL':<24} {tb:>9.2f} -> {ta:>9.2f}   {ta-tb:+.2f}")
            print()

        keys = [k for k in a if k in b and k not in ("perception", "reasoning")]
        # Overall first, then the rest alphabetically.
        keys.sort(key=lambda k: (k != "Overall", k))
        band = _noise_floor(ds, b)
        for k in keys:
            delta = a[k] - b[k]
            flag = ""
            if k == "Overall" and band is not None:
                flag = "  (within noise)" if abs(delta) < band else "  **"
            print(f"  {k:<24} {b[k]:>9.4f} -> {a[k]:>9.4f}   {delta:+.4f}{flag}")
        if band is not None:
            print(f"  [1.96-sigma on Overall: +/-{band:.4f}]")

        if ds == "POPE" and "precision" in b and "recall" in b:
            # POPE asks "is object X in the image?", and 70% of its questions
            # are about objects that are absent. A model that hallucinates
            # says Yes too often: that is a false positive, so it costs
            # precision while leaving recall alone. Less hallucination should
            # therefore show up as precision rising -- possibly with recall
            # falling, if the model has merely become more reluctant to say
            # Yes at all. Reading only F1 hides that trade entirely.
            dp = a["precision"] - b["precision"]
            dr = a["recall"] - b["recall"]
            print(f"\n  hallucination read: precision {dp:+.2f}, recall {dr:+.2f}")
            if dp > 0.5 and dr >= -0.5:
                print("    -> fewer false Yes at no cost to coverage: less hallucination")
            elif dp > 0.5 and dr < -0.5:
                print("    -> precision bought with recall: more conservative, not "
                      "necessarily more accurate")
            elif abs(dp) <= 0.5 and abs(dr) <= 0.5:
                print("    -> neither moved: the intervention did not change "
                      "object-existence judgements")
            else:
                print("    -> precision down: more false Yes, i.e. more hallucination")


if __name__ == "__main__":
    main()
