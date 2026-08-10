"""Tabulate every model in this experiment across all six benchmarks.

Five configurations were trained and evaluated (baseline, three DPO rounds, an
SFT, and SFT followed by DPO), each scored on six benchmarks that live in
several result directories. Assembling that by hand invites transcription
errors, and every number in EXPERIMENT_LOG is supposed to come from the CSVs
rather than from notes.

MME is reported two ways on purpose. Its published total is accuracy plus
acc_plus, where acc_plus requires both questions about an image to be right, so
one flipped answer is charged twice and a 3.4% move in the total can be a 0.4%
move in the answers. The per-question accuracy is the honest scale for
comparing models, and it is only available from the auxmatch sheet.

Usage:
    python tools/summarize_all_runs.py
"""
import csv
import glob
import os
import sys

ROOT = os.environ.get("EVAL_OUT", "/usr/local/kai/lx/eval_out")

# model label -> directories that hold its results, in any order
MODELS = [
    ("baseline",    ["baseline"]),
    ("A 3k/bs16",   ["dpo"]),
    ("B 3k/bs16-4x", ["dpo_lr2e5"]),
    ("C 15k/bs63",  ["dpo_large", "dpo_large_pope", "dpo_large_mcq"]),
    ("SFT",         ["sft", "sft_pope", "sft_mcq"]),
    ("SFT+DPO",     ["sftdpo", "sftdpo_pope", "sftdpo_mcq"]),
]

ROWS = ["MME_total", "MME_acc%", "MMStar", "MMBench", "AI2D",
        "POPE_halluc%", "Hallu_aAcc", "Hallu_fAcc"]


def _csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        r = list(csv.reader(fh))
    return dict(zip(r[0], r[1])) if len(r) > 1 else {}


def collect(dirs):
    out = {}
    for d in dirs:
        base = os.path.join(ROOT, d, "vita_qwen2")
        for f in glob.glob(os.path.join(base, "*.csv")):
            b = os.path.basename(f)
            v = _csv(f)
            try:
                if "MME_score" in b:
                    out["MME_total"] = float(v["perception"]) + float(v["reasoning"])
                elif "MMStar_acc" in b:
                    out["MMStar"] = float(v["Overall"]) * 100
                elif "MMBench" in b:
                    out["MMBench"] = float(v["Overall"]) * 100
                elif "AI2D" in b:
                    out["AI2D"] = float(v["Overall"]) * 100
                elif "HallusionBench_score" in b:
                    out["Hallu_aAcc"] = float(v["aAcc"])
                    out["Hallu_fAcc"] = float(v["fAcc"])
            except (KeyError, ValueError):
                continue
        # Per-question figures need the sheet, not the summary CSV.
        for pat, key in (("MME", "MME_acc%"), ("POPE", "POPE_halluc%")):
            hits = glob.glob(os.path.join(base, f"*{pat}*auxmatch.xlsx"))
            if not hits or key in out:
                continue
            try:
                import pandas as pd
                x = pd.read_excel(hits[0])
                if key == "MME_acc%":
                    out[key] = x["score"].astype(bool).mean() * 100
                else:
                    neg = x["answer"] == "No"
                    out[key] = (x.loc[neg, "extracted"] == "Yes").mean() * 100
            except Exception:  # noqa: BLE001 - a missing sheet is not fatal
                pass
    return out


def main():
    data = {label: collect(dirs) for label, dirs in MODELS}
    labels = [m[0] for m in MODELS]

    print(f"{'benchmark':<15}" + "".join(f"{l:>14}" for l in labels))
    print("-" * (15 + 14 * len(labels)))
    for row in ROWS:
        cells = []
        for l in labels:
            v = data[l].get(row)
            cells.append(f"{v:>14.2f}" if v is not None else f"{'-':>14}")
        print(f"{row:<15}" + "".join(cells))

    print(f"\ndelta vs baseline ({'higher is better except POPE_halluc%'}):")
    print(f"{'benchmark':<15}" + "".join(f"{l:>14}" for l in labels[1:]))
    print("-" * (15 + 14 * (len(labels) - 1)))
    for row in ROWS:
        b = data["baseline"].get(row)
        if b is None:
            continue
        cells = []
        for l in labels[1:]:
            v = data[l].get(row)
            cells.append(f"{v - b:>+14.2f}" if v is not None else f"{'-':>14}")
        print(f"{row:<15}" + "".join(cells))


if __name__ == "__main__":
    main()
