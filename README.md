# VITA-RL: Extending VITA-1.5 with Reinforcement Learning

> [!IMPORTANT]
> **This is not the official VITA repository.**
>
> This project is a fork of and an extension to [**VITA-MLLM/VITA**](https://github.com/VITA-MLLM/VITA)
> (VITA-1.5: *Towards GPT-4o Level Real-Time Vision and Speech Interaction*).
>
> - **Upstream repository**: https://github.com/VITA-MLLM/VITA
> - **Baseline commit**: [`35d064a`](https://github.com/VITA-MLLM/VITA/commit/35d064a6542a5d812136fcd66fa93d9beb27b03c) (2025-03-28)
> - **Upstream paper**: [VITA-1.5 (arXiv:2501.01957)](https://arxiv.org/pdf/2501.01957)
>
> All model architecture, training recipes, benchmark numbers, and pretrained weights
> described below are the work of the original **VITA team (Tencent Youtu Lab et al.)**.
> This repository adds no claim over them. Please cite the
> [original papers](#️-citation) and respect the [original license](./License.txt),
> which restricts use to **academic, research and educational purposes only**.

> Language: **English** | [中文](./README_zh-CN.md)

---

## 🎯 About This Fork

The goal of this repository is to **reproduce VITA-1.5 end-to-end**, and then to
**extend it with a reinforcement learning stage**, which upstream does not provide
(the original codebase contains only supervised fine-tuning).

> **Scope note.** This fork's RL work targets the **text + image/video**
> modalities only. VITA-1.5's audio encoder is kept as a frozen component
> throughout (it ships with the checkpoint and inference still supports it),
> but no audio training or audio RL happens here.

### Roadmap

| Stage | Status | Description |
|---|---|---|
| 1. Reproduce inference | ✅ Done | Text, audio and noisy-audio queries all run against the released VITA-1.5 checkpoint |
| 2. Verify training pipeline | ✅ Done | Runs end to end on 8×H800 with a synthetic dataset; checkpoints save and reload |
| 3. Benchmark baseline | ✅ Done | **MME 2353.5, MMStar 59.8, MMBench 77.8, AI2D 79.2 — all within 1.2 points of the paper.** See [BENCHMARKS.md §2.6](./docs/04-evaluation/BENCHMARKS.md) |
| 4. Train on real data | ✅ Done | 3000 RLAIF-V preference pairs, one epoch of LoRA DPO, first-step loss exactly `-log(0.5)` |
| 5. Add RL | ✅ Done | DPO: POPE hallucination 10.97% → 8.82% (McNemar p<1e-4) via SFT-then-DPO — see [EXPERIMENT_LOG.md](./docs/03-experiments/EXPERIMENT_LOG.md). GRPO: multimodal extension trained on CLEVR counting with a verifiable reward, **held-out accuracy 44.6% → 77.4% in 400 steps** (win rate 0.977), zero regression on MME/POPE/MMBench, plus matched SFT-control and OOD experiments that map the method's boundary — see [GRPO_DEEP_DIVE.md](./docs/03-experiments/GRPO_DEEP_DIVE.md) |

**Read [EXPERIMENT_LOG.md](./docs/03-experiments/EXPERIMENT_LOG.md) for the whole story of stages 3-5**:
three rounds of DPO on RLAIF-V, none of which moved the benchmarks beyond
noise, with the reason measured rather than guessed — the base model separates
those preference pairs at 53.6% (95% CI [50.3%, 56.8%], n=900), a
signal-to-noise ratio of 0.055–0.11. Raising the effective batch from 16 to 63
made DPO genuinely learn on this data (loss below 0.69, 18x the reward margin)
and the benchmarks still did not improve — POPE changed 24 of 5127 answers,
12 fixed against 12 broken. `tools/probe_preference_separability.py` predicts
this in eight minutes, before the four hours behind it. The endgame is
SFT-then-DPO (POPE hallucination 10.97% → 8.82%).

**Read [GRPO_DEEP_DIVE.md](./docs/03-experiments/GRPO_DEEP_DIVE.md) for the GRPO arc**: the same
lesson in a different costume. Two rounds on RLAIF-V with proxy rewards
(keyword overlap, LLM judge) moved the KL 6x without moving any content
reward — within a group of 8 open-ended descriptions, proxy scores rank
stylistic luck, so the group-normalised advantage was ranking noise.
Switching to a verifiable reward (CLEVR counting, binary exact-match) made
the curve take off: held-out accuracy 44.6% → 77.4% in 400 steps. The
follow-up controls then mapped the boundary honestly: zero regression on
MME/POPE/MMBench, OOD transfer to SuperCLEVR (+17pt), and a data-matched
SFT control that ties GRPO in-distribution at 1/7.5 the cost and beats it
OOD — with the stage-2 experiment (same SFT start, same fresh prompts,
SFT-vs-GRPO continuation) showing a ~77–78% task ceiling either way. The
empirical rule that falls out: SFT while its loss still falls; GRPO only
when loss floors and residual errors still have pass@G > pass@1.

**New to this codebase?** Start with [PRIMER.md](./docs/00-background/PRIMER.md) — the background
you need before the other documents make sense: the negative-index placeholder
mechanism, measured token budgets, the three encoders, and the traps that cost
the most time. Its [reading order](./docs/00-background/PRIMER.md#12-建议的阅读顺序) lays out a
four-stage path through the code, with timings and which parts need a GPU.

### Documentation map (organized as a pipeline)

Everything under `docs/` is numbered in the order you would work through the
project — background, setup, data, training, evaluation, review:

| Stage | Directory | What is inside |
|---|---|---|
| 0 · Background | [docs/00-background/](./docs/00-background/) | [PRIMER](./docs/00-background/PRIMER.md) (**read first** — prerequisites for everything else) · [ARCHITECTURE](./docs/00-background/ARCHITECTURE.md) ([中文](./docs/00-background/ARCHITECTURE_zh-CN.md)) — why the code is the way it is · [CODEMAP](./docs/00-background/CODEMAP.md) — clickable jump table for reading on GitHub |
| 1 · Setup | [docs/01-setup/](./docs/01-setup/) | [ENVIRONMENT](./docs/01-setup/ENVIRONMENT.md) (**conda rebuild + download links for every weight/dataset/benchmark**) · [REPRODUCE](./docs/01-setup/REPRODUCE.md) ([中文](./docs/01-setup/REPRODUCE_zh-CN.md)) — install order and why each pin exists · [HANDBOOK](./docs/01-setup/HANDBOOK.md) — commands, landmines, troubleshooting · [MIGRATION](./docs/01-setup/MIGRATION.md) ([中文](./docs/01-setup/MIGRATION_zh-CN.md)) |
| 2 · Data | [docs/02-data/](./docs/02-data/) | [DATASETS](./docs/02-data/DATASETS.md) — what the paper used, what is downloadable, what this fork actually trained on (§3.3) |
| 3 · Training & experiments | [docs/03-experiments/](./docs/03-experiments/) | [EXPERIMENT_LOG](./docs/03-experiments/EXPERIMENT_LOG.md) (**the DPO line end to end** + GRPO summary) · [SFT_DPO_DEEP_DIVE](./docs/03-experiments/SFT_DPO_DEEP_DIVE.md) (mechanisms, memory math, 21 interview Q&As) · [GRPO_DEEP_DIVE](./docs/03-experiments/GRPO_DEEP_DIVE.md) (**the GRPO line end to end**: math, rewards, metric handbook, six rounds + controls, DAPO/GSPO/vLLM, interview Q&A) |
| 4 · Evaluation | [docs/04-evaluation/](./docs/04-evaluation/) | [BENCHMARKS](./docs/04-evaluation/BENCHMARKS.md) — measured numbers, noise bands, timings, memory |
| 5 · Review | [docs/05-review/](./docs/05-review/) | [KNOWLEDGE](./docs/05-review/KNOWLEDGE.md) (**master index for interview prep**: one-line takeaway + exact source per knowledge point) · [PROJECT_SUMMARY](./docs/05-review/PROJECT_SUMMARY.md) — one-page summary + reading route |

Working on this repo with an AI coding agent? [AGENTS.md](./AGENTS.md) is the
machine-oriented entry point: environment-rebuild constraints, verification
commands ordered by cost, and repo conventions.

Raw run artifacts (per-run training logs, step-by-step trainer state, raw
evaluation outputs) salvaged from the now-decommissioned dev machine live in
[artifacts/](./artifacts/README.md) — every number in the documents above can
be traced back to a file there. The five GRPO-era LoRA adapters are hosted at
[lee31221/VITA-RL](https://huggingface.co/lee31221/VITA-RL) on Hugging Face;
merging one into the base model with `tools/merge_and_eval.py` exactly
reproduces the corresponding evaluated checkpoint. The DPO-era weights were
lost with an earlier dev machine — the full record and reproduction path
survive in [EXPERIMENT_LOG.md §13.3](./docs/03-experiments/EXPERIMENT_LOG.md).

The two walkthroughs worth knowing about: [ARCHITECTURE.md
§5](./docs/00-background/ARCHITECTURE.md#5-prepare_inputs_labels_for_multimodal-the-heart-of-the-model)
dissects the function that makes this model work, and [§14](./docs/00-background/ARCHITECTURE.md#14-the-rl-stack-dpo-and-grpo)
covers the RL stack this fork added.

See [REPRODUCE.md](./docs/01-setup/REPRODUCE.md) for the full log: working dependency set,
the code fixes required, and how to run the training smoke test. See
[ARCHITECTURE.md](./docs/00-background/ARCHITECTURE.md) for how the model and codebase actually
work — the modality-fusion mechanism, the three encoders, the inference and
training paths, and how the RL stack (DPO + GRPO) attaches. See
[DATASETS.md](./docs/02-data/DATASETS.md) for the training-data survey: what the paper used,
what is still downloadable as of August 2026, and three plans sized to
available disk.

### Reproducing this

The install and quick-start instructions in the upstream README do not work
as written on every machine (see [REPRODUCE.md](./docs/01-setup/REPRODUCE.md) for why).
Start here instead:

```bash
export VITA_REPO=$(pwd) VITA_WEIGHTS=/path/to/weights   # ~25 GB free
conda create -n vita python=3.10 -y && conda activate vita
# staged dependency install: REPRODUCE.md#install-order-order-matters
# weights + config localization: REPRODUCE.md#weights
python tools/localize_config.py \
    --model-path "$VITA_WEIGHTS/VITA-1.5" \
    --vision-tower "$VITA_WEIGHTS/InternViT-300M-448px"
```

Exact resolved versions are in
[`requirements-lock.txt`](./requirements-lock.txt) — read it as a record of a
known-good set, not as an install path.

Rebuilding on a fresh machine? See [MIGRATION.md](./docs/01-setup/MIGRATION.md) — git holds
only code (~11 MB); the weights and conda environment must be re-acquired.

### Changes relative to upstream

- Added a `.gitignore` (upstream has none) covering training outputs, model weights and secrets.
- Rewrote this README to attribute the work to the upstream project and to document the fork's goals.
- **Fixed `cache_position` under the pinned `transformers==4.41.1`** — upstream's
  `vita_qwen2.py` cannot generate on the version its own `requirements.txt`
  pins. See [REPRODUCE.md](./docs/01-setup/REPRODUCE.md#code-fixes-required).
- **Added the missing `DataConfig` keys** (`Pretrain_video0`, `Pretrain_audio`)
  that several upstream training scripts pass but that were never defined.
- **Made `audios` optional in `prepare_inputs_labels_for_multimodal`** — the
  `None` branch existed but was unreachable, so every text-only or image-only
  forward pass had to be fed a dummy waveform and ran the 341M-parameter audio
  encoder for nothing. See [ARCHITECTURE.md](./docs/00-background/ARCHITECTURE.md#12-known-defects-and-rough-edges).
- **Added GRPO** on top of DPO: the policy samples its own completions and a
  pluggable reward scores them during training, with the group as the
  baseline instead of a critic. `vita/train/{rewards,grpo_loss,grpo_data,grpo_trainer}.py`
  and `train_grpo.py`, plus `tools/test_grpo_loss.py` (39 checks) and
  `tools/test_rewards.py` (44 checks). Extended from text-only to
  **image+text** (vision features are fused into the prompt embeddings once,
  then reused by all G rollouts), with PPO-style sample reuse
  (`--grpo_num_iterations`) and verifiable rewards (`answer` exact match +
  graded `format`). Trained for real on CLEVR counting:
  `tools/make_clevr_grpo_data.py`, `script/train/grpo_clevr.sh`,
  `tools/eval_grpo_heldout.py` — held-out accuracy 44.6% → 77.4% in 400
  steps, zero regression on general benchmarks. Then bounded honestly with
  controls: a data-matched SFT arm (`tools/make_clevr_sft_data.py`,
  `script/train/sft_clevr.sh`), a SuperCLEVR OOD eval
  (`tools/make_superclevr_eval_data.py`), and a stage-2 continuation
  experiment (`tools/make_clevr_stage2_data.py`) that pins the task ceiling
  at ~77–78% regardless of channel. See
  [GRPO_DEEP_DIVE.md](./docs/03-experiments/GRPO_DEEP_DIVE.md) and
  [HANDBOOK.md §9](./docs/01-setup/HANDBOOK.md#9-grpo组相对策略优化).
- **Added offline DPO**, the first RL-family objective in this codebase
  (upstream has only SFT). `vita/train/dpo_{loss,data,trainer}.py` and
  `train_dpo.py`, with `tools/test_dpo_loss.py` (19 CPU checks) and
  `script/train/dpo_smoke_test.sh`. The reference policy is the same weights
  with the LoRA adapter disabled, so it costs no extra memory. See
  [HANDBOOK.md §8](./docs/01-setup/HANDBOOK.md#8-dpo离线偏好优化).
- Added `encode_images_deduped` to `vita_arch.py`: when a batch repeats the
  same media across sequences (DPO's chosen/rejected pair, later GRPO's
  rollout group), the vision tower encodes one repetition and the features
  are tiled. Bit-identical, asserted with `torch.equal` in
  `tools/test_image_dedup.py`; 44-46% off the vision-tower forward. Opt-in
  via `image_group_size`, so SFT is untouched.
- Generalised `train()` in `vita/train/train.py` to accept optional
  argument-class, data-module and trainer factories, so DPO reuses the ~230
  lines of model construction instead of copying them. Calling it with no
  arguments behaves exactly as before.
- **Made LoRA usable.** `find_all_linear_names` did not exclude
  `audio_encoder`, and whale contains two `nn.Linear`s whose leaf name is the
  digit `"0"`; peft matches by suffix, so that matched `layers.0` — a whole
  `Qwen2DecoderLayer` — and `--lora_enable True` always failed. Excluded
  `audio_encoder` and skipped numeric leaf names. Single-GPU LoRA now peaks
  at 23.3 GB.
- Added `script/train/smoke_test_lora.sh`, the first script that exercises
  the LoRA path.
- Added `tools/make_smoke_data.py` and `script/train/smoke_test_qwen.sh`, a
  synthetic-data smoke test for the training pipeline.
- Added `tools/test_audio_optional.py`, a CPU unit test for the fix above
  (stubbed encoders, no weights needed).
- Added `tools/inspect_dataset.py`, which loads a configured dataset on CPU
  and reports sequence lengths, which span is supervised, collated shapes,
  and how many samples had their labels silently voided — run it after
  wiring up a dataset and before spending GPUs on it.
- Added `tools/localize_config.py`, which rewrites the checkpoint's
  `mm_vision_tower` / `mm_audio_encoder` from HuggingFace repo IDs to local
  paths so that loading does not require network access.
- Added `PRIMER.md` (prerequisite knowledge), `HANDBOOK.md` (hands-on guide),
  `REPRODUCE.md` (operational log),
  `ARCHITECTURE.md` (code walkthrough), `DATASETS.md` (training-data survey),
  `EXPERIMENT_LOG.md` (the DPO six-round record),
  `GRPO_DEEP_DIVE.md` (the GRPO four-round record and deep dive),
  `SFT_DPO_DEEP_DIVE.md`, `PROJECT_SUMMARY.md`
  and `requirements-lock.txt`.

Any further deviation from upstream will be recorded in this section.

> **Note on reproduction.** The upstream scripts contain hard-coded absolute paths from the
> original authors' cluster (`/mnt/cfs/lhj/...`), hard-coded multi-node addresses, and an
> empty dataset registry. These must be adapted locally before anything will run — see
> [Reproduction Notes](#-reproduction-notes).

---

## 📚 Upstream (VITA-1.5) resources

This README intentionally covers only the fork. For the upstream model
introduction, paper results, official training recipe, inference quick-start,
web demos and benchmark-evaluation instructions, see the
[upstream README](https://github.com/VITA-MLLM/VITA#readme) and the
[VITA-1.5 paper](https://arxiv.org/pdf/2501.01957)
([ModelScope demo](https://modelscope.cn/studios/modelscope/VITA1.5_demo)).
This fork's own measured numbers live in
[BENCHMARKS.md](./docs/04-evaluation/BENCHMARKS.md),
[EXPERIMENT_LOG.md](./docs/03-experiments/EXPERIMENT_LOG.md) and
[GRPO_DEEP_DIVE.md](./docs/03-experiments/GRPO_DEEP_DIVE.md).

## 🛠 Reproduction Notes

*This section is specific to this fork and is not part of the upstream README.*

The upstream code was released as-is from the authors' internal cluster. The following
must be adapted before anything will run — none of these are bugs, they are simply
environment-specific values that were never parameterised:

1. **Hard-coded absolute paths.** Every script under `script/train/` references
   `/mnt/cfs/lhj/...`, `/mnt/cfs2/lhj/...` or `/mnt/shared/data1/lhj/...` for model
   weights and outputs. `GLOBAL_WEIGHTS_PATH` in
   [`vita/constants.py`](./vita/constants.py) is still the placeholder
   `/path/to/model_weights`.

2. **Hard-coded multi-node settings.** The `*_nodes.sh` scripts pin `INDEX` (node rank)
   and `MASTER_ADDR` to the authors' cluster, e.g. `INDEX=3` and
   `MASTER_ADDR="10.206.0.199"` in `finetuneTaskNeg_qwen_nodes.sh`. Each node needs a
   distinct `INDEX`. NCCL variables (`NCCL_SOCKET_IFNAME=eth0`, `NCCL_IB_GID_INDEX=3`)
   assume a specific interconnect.

3. **Empty dataset registry.** [`vita/config/dataset_config.py`](./vita/config/dataset_config.py)
   ships with empty strings for `AudioFolder`, `FolderDict` and `chat_path`. In addition,
   `DataConfig` in [`vita/config/__init__.py`](./vita/config/__init__.py) only defines the
   key `Pretrain_video`, while several scripts pass `--dataset_use Pretrain_video0` or
   `Pretrain_audio`; those keys must be added or the run fails with a `KeyError`.

4. **The data pipeline is selected in source, not on the CLI.** `train.py` imports one of
   seven `data_utils_*` variants via commented-out import lines near the top of
   [`vita/train/train.py`](./vita/train/train.py). The default (`..._neg_patch`) matches
   the documented continual-training recipe.

5. **Pinned, older dependencies.** `torch==2.3.1` and `transformers==4.41.1`.
   `vita/model/language_model/vita_qwen2.py` monkey-patches `Qwen2ForCausalLM.forward`,
   which couples it tightly to that `transformers` version — upgrading is likely to break it.

6. **`command.sh` (upstream) has been removed from this fork.** It was the original
   authors' scratch command history, not a build script, and referenced files that no
   longer exist. It remains available in git history if ever needed.

## ✒️ Citation

**This fork introduces no new publication.** If you use this code, please cite the original
VITA papers — all credit for the model and method belongs to the upstream authors.

```bibtex
@article{fu2025vita,
  title={VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction},
  author={Fu, Chaoyou and Lin, Haojia and Wang, Xiong and Zhang, Yi-Fan and Shen, Yunhang and Liu, Xiaoyu and Li, Yangze and Long, Zuwei and Gao, Heting and Li, Ke and others},
  journal={arXiv preprint arXiv:2501.01957},
  year={2025}
}

@article{fu2024vita,
  title={Vita: Towards open-source interactive omni multimodal llm},
  author={Fu, Chaoyou and Lin, Haojia and Long, Zuwei and Shen, Yunhang and Zhao, Meng and Zhang, Yifan and Dong, Shaoqi and Wang, Xiong and Yin, Di and Ma, Long and others},
  journal={arXiv preprint arXiv:2408.05211},
  year={2024}
}
```


## &#x1F4E3; Statement

**The following statement is inherited from the upstream project and applies equally here:**

**VITA is trained on large-scale open-source corpus, and its output has randomness. Any content generated by VITA does not represent the views of the model developers. We are not responsible for any problems arising from the use, misuse, and dissemination of VITA, including but not limited to public opinion risks and data security issues.**

Additionally: this fork is an unofficial, research-only extension. It is not endorsed by,
affiliated with, or supported by the original VITA authors. Use of the code and the
upstream weights remains subject to [`License.txt`](./License.txt), which permits
**academic, research and educational use only** and prohibits commercial or production use.


## 👍 Acknowledgement

First and foremost, this repository is derived entirely from
[**VITA-MLLM/VITA**](https://github.com/VITA-MLLM/VITA) — thanks to the VITA team for
open-sourcing their work.

VITA itself is built with reference to the following outstanding works: [LLaVA-1.5](https://github.com/haotian-liu/LLaVA), [Bunny](https://github.com/BAAI-DCAI/Bunny), [ChatUnivi](https://github.com/PKU-YuanGroup/Chat-UniVi), [InternVL](https://github.com/OpenGVLab/InternVL), [InternViT](https://huggingface.co/OpenGVLab/InternViT-300M-448px), [Qwen-2.5](https://github.com/QwenLM/Qwen2.5), [VLMEvalkit](https://github.com/open-compass/VLMEvalKit), and [Mixtral 8*7B](https://mistral.ai/news/mixtral-of-experts/).
Thanks！

