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
    "opencv-python-headless==4.10.0.84" soundfile librosa \
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

## Status

- [x] conda environment + all dependencies
- [x] flash-attn working on GPU
- [x] `vita.model` imports cleanly
- [ ] Weights downloaded
- [ ] Inference reproduced (`video_audio_demo.py`)
- [ ] Training reproduced
