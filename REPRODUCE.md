# Reproduction Log

Environment setup and reproduction notes for this fork. Upstream ships a
`requirements.txt` that does not install cleanly on every machine; this file
records what actually worked, and why each deviation was necessary.

## Host

| Item | Value |
|---|---|
| GPU | 8 × NVIDIA H800 80GB |
| NVIDIA driver | 535.129.03 (supports CUDA ≤ 12.2) |
| System CUDA toolkit | 11.4 (`nvcc`) — **not used**, see below |
| System gcc | 4.8.5 — too old to compile C99/C++17 sources |
| conda | 4.14.0 |
| Network | No direct egress; HTTP proxy required for GitHub / HuggingFace |

Because the system toolchain is old, **everything is installed from prebuilt
wheels** (`--only-binary=:all:`). The wheels bundle their own CUDA 12.1
runtime, so the 11.4 system toolkit is irrelevant — only the driver version
matters, and 535 is new enough.

## Environment

```bash
conda create -n vita python=3.10 -y
conda activate vita
export PYTHONPATH=./
```

### Install order (order matters)

```bash
# 1. Torch stack first, with pillow pinned to a version that has a cp310 wheel
pip install --only-binary=:all: "pillow==10.4.0" \
    torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1

# 2. Core deps. xformers must be pinned or it drags in a newer torch.
pip install --only-binary=:all: "numpy<2" "xformers==0.0.27" \
    "transformers==4.41.1" accelerate decord Jinja2 ninja tqdm

# 3. Imports used by the code but absent from requirements.txt
pip install --only-binary=:all: timm einops PyYAML \
    "opencv-python-headless==4.10.0.84" "soundfile==0.12.1" librosa \
    sentencepiece protobuf six

# 4. Re-pin numpy: numba/librosa pull numpy 2.x back in
pip install --only-binary=:all: "numpy==1.26.4" "numba<0.61" "llvmlite<0.44"

# 5. Training only
pip install "deepspeed==0.14.4"

# 6. flash-attn: no wheel on our mirror, and gcc 4.8.5 cannot build it.
#    Use the official prebuilt wheel matching cp310 / torch2.3 / cxx11abiFALSE.
#    Check your ABI with: python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### Deviations from upstream `requirements.txt`

| Package | Upstream | Used here | Reason |
|---|---|---|---|
| `xformers` | unpinned | `0.0.27` | Unpinned resolves to 0.0.35, which requires `torch>=2.10` and conflicts with the pinned `torch==2.3.1`. The install aborts. |
| `pillow` | unpinned | `10.4.0` | Unpinned resolves to 12.3.0, which has no cp310 wheel here and fails to compile under gcc 4.8.5 (`'for' loop initial declarations are only allowed in C99 mode`). |
| `numpy` | unpinned | `1.26.4` | torch 2.3.1 predates the numpy 2.0 ABI break. `numba`/`librosa`/`opencv` will each try to pull numpy 2.x back in — re-pin after installing them. |
| `opencv-python-headless` | not listed | `4.10.0.84` | Imported by the code. Version 5.x hard-requires numpy>=2. |
| `timm`, `einops`, `PyYAML`, `soundfile`, `librosa`, `six`, `sentencepiece` | not listed | latest compatible | Imported by the code but missing from `requirements.txt`. `six` in particular is needed by `vita/model/multimodal_encoder/whale/module/encoder/encoder.py`. |
| `torchvision` | unpinned | `0.18.1` | The release matching torch 2.3.1. |
| `flash-attn` | build from source | prebuilt wheel | `train.py` hard-codes `attn_implementation="flash_attention_2"`, so this is mandatory for training. Source build needs a modern gcc. |

### Optional components (safe to ignore)

These print warnings on import but are not required:

- `apex` — `Please build and install Nvidia apex package...`
- `mamba_ssm` — `Please install mamba_ssm to use MambaSSM component.`

### Verification

```
torch 2.3.1+cu121   cuda 12.1   8 GPUs visible   bf16 supported
numpy 1.26.4        transformers 4.41.1
flash_attn 2.5.9.post1 — GPU kernel executes
vita.model imports: VITAQwen2ForCausalLM, VITAMixtralForCausalLM,
                    VITAMistralForCausalLM, VITAFOQwen2ForCausalLM  ✅
conv modes: default, llama, minicpm, mixtral_two, mixtral_zh,
            nemo, phi3, plain, qwen2p5_instruct
```

`pip check` reports only `decord 0.6.0 is not supported on this platform`,
which is a platform-metadata note rather than a dependency conflict; decord
imports and runs.

## Weights

Downloaded to `/usr/local/kai/lx/weights/` (outside the repo — see `.gitignore`):

| Weight | Source | Size |
|---|---|---|
| VITA-1.5 | [`VITA-MLLM/VITA-1.5`](https://huggingface.co/VITA-MLLM/VITA-1.5) | ~19.6 GB (includes the audio encoder under `audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning/`, ~1.5 GB) |
| InternViT-300M-448px | [`OpenGVLab/InternViT-300M-448px`](https://huggingface.co/OpenGVLab/InternViT-300M-448px) | ~0.3 GB |

The audio encoder ships inside the VITA-1.5 repository, so only two downloads
are needed rather than three.

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('VITA-MLLM/VITA-1.5', local_dir='/path/to/weights/VITA-1.5')
snapshot_download('OpenGVLab/InternViT-300M-448px', local_dir='/path/to/weights/InternViT-300M-448px')
"
```

## Code fixes required

### `cache_position` incompatibility with the pinned `transformers==4.41.1`

**Symptom.** `video_audio_demo.py` fails during `generate()`:

```
KeyError: 'cache_position'
    at vita/model/language_model/vita_qwen2.py:250
```

and, once that is worked around:

```
TypeError: Qwen2Model.forward() got an unexpected keyword argument 'cache_position'
    at vita/model/language_model/vita_qwen2.py:78
```

**Cause.** In `transformers==4.41.1` — the version upstream pins in
`requirements.txt` — `cache_position` had been added to the **Llama**
implementation but not yet to **Qwen2**:

| | 4.41.1 |
|---|---|
| `LlamaForCausalLM.prepare_inputs_for_generation` returns `cache_position` | yes |
| `Qwen2ForCausalLM.prepare_inputs_for_generation` returns `cache_position` | **no** |
| `Qwen2Model.forward` accepts `cache_position` | **no** |

Upstream added this code and pinned `transformers==4.41.1` in the same commit
(`fe4d74e` / `9a968b3`, both 2024-12-20), so the two were never consistent for the
Qwen2 path. `MistralModel` and `MixtralModel` in 4.41.1 likewise do not accept it.

**Fix applied** (`vita/model/language_model/vita_qwen2.py`):

1. In `prepare_inputs_for_generation`, read `cache_position` with `.get()` and, when
   absent, derive it from the cache length —
   `torch.arange(past_length, past_length + input_ids.shape[1])`, which is what
   `cache_position` means. The subsequent position-id adjustment is guarded so it
   is skipped when either tensor is unavailable.
2. In `custom_forward`, only pass `cache_position` to `self.model(...)` when the
   installed `Qwen2Model.forward` actually accepts it, detected once at import
   time via `inspect.signature`.

Both changes are version-conditional, so the file still behaves as the authors
intended on newer `transformers` releases where `cache_position` is supported.

`VITAFOQwen2ForCausalLM` calls `super().forward(...)`, which resolves to the
patched `custom_forward`, so it is covered by the same fix.

> **Not fixed:** `vita/model/language_model/vita_nemo.py` (lines 78, 178) passes
> `cache_position` to `MistralModel` the same way and will hit the identical
> `TypeError` under 4.41.1. It was left alone because the Nemo/Mistral path is not
> exercised by this reproduction and cannot be tested without those weights.

### `soundfile` missing its native library

**Symptom.** Audio queries fail with `cannot open asset/q1.wav!!!!!!!!!!!!!!!!`
followed by `UnboundLocalError: local variable 'sample_rate' referenced before
assignment` in `whale/init_model.py:40` — the loader swallows the real error and
then dereferences an unset variable.

**Cause.** `torchaudio.list_audio_backends()` returned `[]`. The latest
`soundfile` wheel did not ship `libsndfile.so`, so `import soundfile` raised
`OSError: cannot load library 'libsndfile.so'` and torchaudio silently registered
no backend.

**Fix.** Pin `soundfile==0.12.1`, whose wheel bundles the native library:

```bash
pip install --only-binary=:all: "soundfile==0.12.1"
```

After this, `torchaudio.list_audio_backends()` returns `['soundfile']`.

## Inference reproduction

Point the checkpoint's `config.json` at the local encoder paths first, so that
loading does not depend on network access (upstream ships HuggingFace repo IDs
in these fields):

```bash
# in the downloaded VITA-1.5 directory
cp config.json config.json.orig
python -c "
import json; c=json.load(open('config.json'))
c['mm_vision_tower']='/path/to/weights/InternViT-300M-448px'
c['mm_audio_encoder']='/path/to/weights/VITA-1.5/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning'
json.dump(c,open('config.json','w'),indent=2)"
```

All three quick-start modes from the README were run on a single H800:

| Mode | Command | Result |
|---|---|---|
| Text query | `--question "Describe this image."` | ✅ Coherent description of the VITA logo, 7.4 s |
| Audio query | `--audio_path asset/q1.wav` | ✅ Chinese description prefixed with `☞`, 2.3 s |
| Noisy audio | `--audio_path asset/q2.wav` | ✅ Response prefixed with `☟`, 1.9 s |

```bash
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
    --model_path /path/to/weights/VITA-1.5 \
    --image_path asset/vita_newlog.jpg \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --question "Describe this image."
```

The `☞` / `☟` prefixes are the state tokens described in
`vita/util/data_utils_video_audio_neg_patch.py` (`preprocess_multimodal`):
`☞` marks a reply to an audio query, `☜` a reply to a text query, and `☟` a reply
in the negative/noisy-audio condition. Seeing `☞` on `q1.wav` and `☟` on `q2.wav`
confirms both the audio encoder and the negative-sample behaviour are working.

> **README inaccuracy:** the audio examples reference `asset/vita_newlog.png`,
> which does not exist in the repository — the file is `asset/vita_newlog.jpg`.

## Training pipeline smoke test

Upstream cannot ship its training data, and `vita/config/dataset_config.py` is
checked in with empty paths. To verify the training path itself works, this fork
adds a tiny **synthetic** dataset built from the repository's own assets.

> This is a pipeline check, **not** a reproduction of VITA-1.5 training. It says
> nothing about model quality — only that data loading, multimodal token
> expansion, collation, forward, backward, and checkpointing all work.

### Dataset registry fixes

Two upstream defects had to be fixed before any training script could start:

1. **Missing `DataConfig` keys.** `vita/config/__init__.py` defined only
   `Pretrain_video`, but `pretrain_mlp_qwen.sh`, `finetune_qwen.sh` and others
   pass `--dataset_use Pretrain_video0`, and `pretrain_audio_mlp_qwen.sh` passes
   `Pretrain_audio`. Both raise `KeyError` at
   `data_utils_video_audio_neg_patch.py:833`. They are now defined.

2. **Audio path layout is undocumented.** The loader resolves audio as
   `os.path.join(AudioFolder, "audio", file)` — note the hard-coded `audio`
   segment. So `AudioFolder` must be the *parent* of the `audio/` directory, not
   the directory itself.

Both the smoke dataset and the registry entry are opt-in via the
`VITA_SMOKE_DATA_DIR` environment variable, so the default configuration remains
identical to upstream.

### Running it

```bash
python tools/make_smoke_data.py --out-dir /path/to/smoke_data
# place images under <dir>/images and wavs under <dir>/audio
export VITA_SMOKE_DATA_DIR=/path/to/smoke_data
bash script/train/smoke_test_qwen.sh /path/to/output 8
```

The dataset covers the three sample shapes the loader branches on, and each
produces the expected state token:

| Sample type | Fields | State token |
|---|---|---|
| image only | `image` | `☜` (reply to a text query) |
| image + audio | `image`, `audio` | `☞` (reply to an audio query) |
| image + audio + negative | plus `inserted_id` | `☞` and `☟` on the marked turn |

### Result

```
{'loss': 3.1885, 'grad_norm': 45.93, 'learning_rate': 1e-06, 'epoch': 0.33}
{'loss': 3.7144, 'grad_norm': 55.69, 'learning_rate': 5e-07, 'epoch': 0.67}
{'loss': 2.7051, 'grad_norm': 38.45, 'learning_rate': 0.0,   'epoch': 1.0}
{'train_runtime': 14.68, 'train_loss': 3.2026, 'epoch': 1.0}
```

Losses are finite and gradients flow. With only 3 steps at `lr=1e-6` on 24
synthetic samples, the loss trajectory is meaningless — that is intentional, the
learning rate is deliberately tiny so the run cannot be mistaken for training.

Checkpoint saving was verified separately (`--save_strategy steps --save_steps 2`,
which `smoke_test_qwen.sh` disables by default). `VITATrainer._save_checkpoint`
wrote a complete 4-shard model plus ZeRO-3 optimizer state, and the resulting
checkpoint **loads and generates correctly** with `video_audio_demo.py` — closing
the loop from training back to inference.

> A saved full checkpoint is ~16 GB, and a run that keeps optimizer state can
> reach ~130 GB. Point `--output_dir` outside the repository; `.gitignore`
> already excludes `outputs/` and `checkpoint-*/` in case it is not.

### Memory note

Full-parameter 7B training does **not** fit on one 80GB H800. A single-GPU run
completes forward and backward, then fails allocating AdamW state:

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.84 GiB
    at torch/optim/adamw.py:128, state["exp_avg_sq"] = torch.zeros_like(...)
```

AdamW keeps two fp32 moments per parameter (~56 GB for 7B) plus fp32 master
weights (~28 GB). ZeRO-3 shards these across ranks, so **8 GPUs are required**;
the same run succeeds on 8. Use LoRA (`--lora_enable True`) if fewer GPUs are
available — note that code path is inherited from LLaVA and is not exercised by
any upstream script.

## Status

- [x] conda environment + all dependencies
- [x] flash-attn working on GPU
- [x] `vita.model` imports cleanly
- [x] Weights downloaded (VITA-1.5 19.6 GB + InternViT 0.3 GB)
- [x] Inference reproduced — text, audio, and noisy-audio queries
- [x] Training pipeline verified end to end on synthetic data (8 × H800)
- [ ] Training on real data (requires a dataset upstream does not provide)
