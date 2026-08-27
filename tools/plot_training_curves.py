#!/usr/bin/env python3
"""Regenerate asset/grpo_training_curves.png from artifacts/train/*.

The figure tells the GRPO story in four panels:
  1. R4 verifiable reward takes off (answer 0.59->0.89, format saturates first)
  2. R1 proxy reward also rises -- but benchmarks never moved (the trap)
  3. KL: R4 (beta=0.04) vs R6 (beta=0, KL measured but not penalized)
  4. R4 degenerate-group fraction climbs as headroom is exhausted

Data source is trainer_state.json log_history, so the plot is reproducible
from the repo alone (no wandb access needed).

Usage: python tools/plot_training_curves.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "artifacts", "train")
OUT = os.path.join(ROOT, "asset", "grpo_training_curves.png")


def series(run, key):
    hist = json.load(open(os.path.join(TRAIN, run, "trainer_state.json")))[
        "log_history"
    ]
    pts = [(h["step"], h[key]) for h in hist if key in h and "step" in h]
    # multiple log entries can share a step (per-accumulation logging); average them
    agg = {}
    for s, v in pts:
        agg.setdefault(s, []).append(v)
    steps = sorted(agg)
    return steps, [sum(agg[s]) / len(agg[s]) for s in steps]


def smooth(vals, w=15):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - w + 1)
        out.append(sum(vals[lo : i + 1]) / (i + 1 - lo))
    return out


def panel(ax, curves, title, ylabel, annotate=None):
    for run, key, label, color in curves:
        s, v = series(run, key)
        ax.plot(s, v, color=color, alpha=0.25, lw=0.8)
        ax.plot(s, smooth(v), color=color, lw=1.8, label=label)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    if annotate:
        ax.text(
            0.98,
            0.05,
            annotate,
            transform=ax.transAxes,
            ha="right",
            fontsize=8,
            style="italic",
            color="dimgray",
        )


fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

panel(
    axes[0][0],
    [
        ("r4_grpo_clevr", "reward/answer", "answer (exact match)", "tab:blue"),
        ("r4_grpo_clevr", "reward/format", "format (graded)", "tab:orange"),
    ],
    "R4 - CLEVR, verifiable reward: the takeoff",
    "reward",
    annotate="held-out 44.6% -> 77.4%",
)
panel(
    axes[0][1],
    [("r1_grpo_rlaif", "reward/mean", "reward/mean (proxy mix)", "tab:red")],
    "R1 - RLAIF-V, proxy reward: rises, benchmarks flat",
    "reward",
    annotate="MME/POPE/MMBench: no movement",
)
panel(
    axes[1][0],
    [
        ("r4_grpo_clevr", "grpo/kl", "R4 (beta=0.04)", "tab:blue"),
        ("r6_beta0", "grpo/kl", "R6 (beta=0, measured only)", "tab:green"),
    ],
    "KL divergence: penalized vs free",
    "grpo/kl",
    annotate="same accuracy either way (R6 ablation)",
)
panel(
    axes[1][1],
    [
        (
            "r4_grpo_clevr",
            "groups/degenerate_frac",
            "degenerate groups (all-same reward)",
            "tab:purple",
        )
    ],
    "R4 - degenerate group fraction: headroom runs out",
    "fraction",
    annotate="4% -> 44%: groups become all-correct",
)

fig.suptitle(
    "GRPO training curves, replotted from artifacts/train/*/trainer_state.json",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
