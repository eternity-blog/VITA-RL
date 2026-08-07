# Migrating to Another Machine

For rebuilding this project from scratch on a fresh host.

> Language: **English** | [中文](./MIGRATION_zh-CN.md)

## TL;DR

Git holds only code and documentation (~11 MB). **The weights (19.6 GB) and the
conda environment (6.6 GB) are not in git** and must be re-acquired on the new
machine — that is where the time goes.

```bash
git clone https://github.com/eternity-blog/VITA-RL.git
cd VITA-RL
# then follow "Quick start" in REPRODUCE.md
```

## 1. What is and is not in git

| | Content | Size | How to get it on the new host |
|---|---|---|---|
| ✅ in git | All code, 6 documents, tool scripts, `requirements-lock.txt` | ~11 MB | `git clone` |
| ❌ not in git | VITA-1.5 weights | 19.6 GB | re-download from HuggingFace |
| ❌ not in git | InternViT-300M-448px | 0.65 GB | re-download from HuggingFace |
| ❌ not in git | conda env `vita` | 6.6 GB | reinstall per the docs (~10–20 min) |
| ❌ not in git | synthetic smoke data | tiny | `python tools/make_smoke_data.py` |

This is deliberate: `.gitignore` excludes weights and training outputs, which
would otherwise push the repository into the tens of GB.

## 2. Rebuild steps

### 1. Clone

```bash
git clone https://github.com/eternity-blog/VITA-RL.git
cd VITA-RL
export VITA_REPO=$(pwd)
export VITA_WEIGHTS=/path/to/weights     # needs ~25 GB free
```

If the new host also lacks direct internet egress, set a proxy first — clone,
pip and HuggingFace all need it:

```bash
export http_proxy=http://<host>:<port> https_proxy=http://<host>:<port>
export no_proxy=localhost,127.0.0.1
```

### 2. Environment

**Do not** run `pip install -r requirements.txt` directly — the upstream file
does not install cleanly, see
[REPRODUCE.md](./REPRODUCE.md#deviations-from-upstream-requirementstxt).
Follow the [staged install](./REPRODUCE.md#install-order-order-matters) instead.

Then sanity-check before committing to a 20 GB download:

```bash
python -c "
import torch, numpy, transformers, flash_attn
from flash_attn import flash_attn_func
print('torch', torch.__version__, '| cuda', torch.version.cuda)
print('numpy', numpy.__version__, '| transformers', transformers.__version__)
print('GPUs', torch.cuda.device_count(), '| bf16', torch.cuda.is_bf16_supported())
q = torch.randn(2,64,8,64, dtype=torch.bfloat16, device='cuda')
print('flash-attn kernel ok:', tuple(flash_attn_func(q,q,q).shape))
import torchaudio; print('audio backends:', torchaudio.list_audio_backends())"
```

### 3. Weights

```bash
python -c "
from huggingface_hub import snapshot_download
import os
w = os.environ['VITA_WEIGHTS']
snapshot_download('VITA-MLLM/VITA-1.5', local_dir=f'{w}/VITA-1.5')
snapshot_download('OpenGVLab/InternViT-300M-448px', local_dir=f'{w}/InternViT-300M-448px')"
```

Resumable — just re-run if the connection drops.

### 4. Localize the config (important)

```bash
python tools/localize_config.py \
    --model-path   "$VITA_WEIGHTS/VITA-1.5" \
    --vision-tower "$VITA_WEIGHTS/InternViT-300M-448px"
```

⚠️ **This must be redone on the new machine.** It writes absolute paths, and the
old machine's paths are meaningless here. If you copied the weights directory
across rather than re-downloading, `config.json` still points at the **old**
paths and this step is mandatory (optionally `--restore` first).

### 5. Verify

```bash
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
    --model_path "$VITA_WEIGHTS/VITA-1.5" \
    --image_path asset/vita_newlog.jpg \
    --model_type qwen2p5_instruct --conv_mode qwen2p5_instruct \
    --question "Describe this image."
```

A coherent description of the VITA logo means success.

## 3. Common pitfalls

| Pitfall | Notes |
|---|---|
| **Copying the conda env directory** | Unreliable — it contains compiled extensions and baked-in absolute paths. Reinstall from the docs. |
| **Copying weights but not re-running localize_config** | `config.json` holds the old machine's absolute paths; loading fails or silently falls back to the network. |
| **Mismatched flash-attn wheel** | The pinned wheel is `cp310 / torch2.3 / cxx11abiFALSE`. On a different Python or torch, pick another; check ABI with `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"`. |
| **Too few GPUs** | One is fine for inference. **Full-parameter training needs 8**; a single GPU OOMs allocating AdamW state. |
| **Forgetting the proxy** | Without egress, `git clone`, `pip` and `snapshot_download` all hang. |
| **numpy upgraded to 2.x** | torch 2.3.1 is incompatible. Re-pin `numpy==1.26.4` after installing numba/librosa/opencv. |
| **Forgetting `export PYTHONPATH=./`** | Yields `ModuleNotFoundError: No module named 'vita'`. |

## 4. Copying weights instead of re-downloading

Copying is faster than re-downloading, but **step 4 is still mandatory**:

```bash
# old machine
tar -cf - -C /path/to weights | ssh newhost 'tar -xf - -C /path/to'

# new machine: repoint the config
python tools/localize_config.py \
    --model-path   "$VITA_WEIGHTS/VITA-1.5" \
    --vision-tower "$VITA_WEIGHTS/InternViT-300M-448px"
```

## 5. Post-migration checklist

- [ ] `git log --oneline -1` matches the latest commit on GitHub
- [ ] `git status` is clean
- [ ] the environment self-check passes and `audio backends` is **non-empty**
- [ ] `python -c "import json;print(json.load(open('$VITA_WEIGHTS/VITA-1.5/config.json'))['mm_vision_tower'])"` shows the **new** machine's path
- [ ] text inference works
- [ ] audio inference works (`q1.wav` replies start with `☞`, `q2.wav` with `☟`)
- [ ] the five CPU test suites pass (no GPU, no weights, seconds):
      `for t in dpo_loss grpo_loss rewards image_dedup audio_optional; do python tools/test_$t.py; done`
- [ ] if training: `bash script/train/smoke_test_qwen.sh /tmp/smoke_out 8` completes

Then check the numbers against [BENCHMARKS.md](./BENCHMARKS.md), which records
what this all measured on the original 8×H100 box. Two of those are identities
rather than measurements and **must** reproduce anywhere:

| Check | Expected |
|---|---|
| DPO first-step loss | exactly `0.6931` (`-log(0.5)`) |
| GRPO first-step `grpo/kl` | exactly `0.0` |

If the CPU suites pass but either of those is off, the problem is the
reference policy or the `mm_projector` freeze, not the port. Timings and peak
memory will differ on different hardware; those two will not.

## 6. Where to pick up

Current progress is in the [README roadmap](./README.md#roadmap): inference and
the training pipeline are both verified; RL is next. The technical path and its
five concrete obstacles are in
[ARCHITECTURE.md §13](./ARCHITECTURE.md#13-where-rl-would-attach).
