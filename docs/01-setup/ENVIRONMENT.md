# 环境复现与资源下载

> 开发机已回收，这份文档回答两个问题：**conda 环境怎么原样搭回来**、
> **权重/数据集/评测集从哪里下载**。
>
> 配套：[REPRODUCE.md](./REPRODUCE.md) 是安装顺序的权威出处（每个 pin 的
> 原因、上游 requirements.txt 的偏离表）；本文是执行清单 + 资源链接总表。

## 目录

- [1. 已验证的机器](#1-已验证的机器)
- [2. conda 环境复现](#2-conda-环境复现)
- [3. 两个 lock 文件](#3-两个-lock-文件)
- [4. 装完怎么验证](#4-装完怎么验证)
- [5. 资源下载总表](#5-资源下载总表)

---

## 1. 已验证的机器

| | DPO 时代（§EXPERIMENT_LOG 3–9） | GRPO 时代（§EXPERIMENT_LOG 14，artifacts/ 的出处） |
|---|---|---|
| GPU | 8× H100 80GB | 8× H800 80GB |
| 驱动 | 535.129.03 | 595.58.03 |
| Python | 3.10.18 | 3.10.20 |
| torch | 2.3.1+cu121 | 2.3.1+cu121 |
| lock 文件 | `requirements-lock.txt` | `requirements-lock-grpo.txt` |

torch wheel 自带 CUDA 12.1 运行时，系统 CUDA toolkit 版本无关紧要，
只要驱动支持 CUDA ≥ 12.1。推理最低 1× 80GB；全参训练需要 8 卡。

## 2. conda 环境复现

**不要**直接 `pip install -r` lock 文件——安装顺序有依赖
（numpy 会被 numba/librosa/opencv 拉回 2.x，必须重 pin；
xformers 不 pin 会拖走 torch）。按下面的顺序装：

```bash
conda create -n vita python=3.10 -y && conda activate vita

# 1. torch 栈（pillow 先 pin，否则 cp310 无 wheel 会走源码编译）
pip install --only-binary=:all: "pillow==10.4.0" \
    torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1

# 2. 核心依赖（xformers 必须 pin）
pip install --only-binary=:all: "numpy<2" "xformers==0.0.27" \
    "transformers==4.41.1" accelerate decord Jinja2 ninja tqdm

# 3. 代码 import 了但上游 requirements.txt 没写的
pip install --only-binary=:all: timm einops PyYAML \
    "opencv-python-headless==4.10.0.84" "soundfile==0.12.1" librosa \
    sentencepiece protobuf six

# 4. 重新 pin numpy（numba/librosa 会拉回 2.x）
pip install --only-binary=:all: "numpy==1.26.4" "numba<0.61" "llvmlite<0.44"

# 5. 训练
pip install "deepspeed==0.14.4" "peft==0.11.1"

# 6. flash-attn：用官方预编译 wheel（cp310 / torch2.3 / cxx11abiFALSE）
#    ABI 自查：python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# 7. 指标与评测（GRPO 时代新增；VLMEvalKit 依赖必须 --no-deps，
#    否则它会升级 transformers/torch）
pip install wandb tensorboard
pip install --no-deps pandas openpyxl xlsxwriter rich portalocker validators
pip install "moviepy==2.2.1"   # VLMEvalKit import moviepy.editor 需要 shim，见 HANDBOOK.md §5
```

装完 `pip list --format=freeze` 与对应 lock 文件 diff，确认落在同一组版本上。

## 3. 两个 lock 文件

| 文件 | 记录的是 | 何时用 |
|---|---|---|
| [`requirements-lock.txt`](../../requirements-lock.txt) | DPO 时代 H100 机器的已知可用组合 | 复现 DPO 六轮 |
| [`requirements-lock-grpo.txt`](../../requirements-lock-grpo.txt) | GRPO 时代 H800 机器（`artifacts/` 所有 run 的出处），比前者多 wandb/tensorboard/VLMEvalKit 依赖 | 复现 GRPO R1–R6 |

两者核心 pin（torch/transformers/deepspeed/peft/flash-attn）完全一致，
装哪个都能跑全部代码路径。

## 4. 装完怎么验证

```bash
# 内核与关键 import
python - <<'PY'
import torch, numpy, transformers, flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 8, 4, 64, dtype=torch.bfloat16, device='cuda')
print('flash-attn kernel ok:', tuple(flash_attn_func(q, q, q).shape))
print(torch.__version__, numpy.__version__, transformers.__version__)
PY

# 训练链路 smoke（8 卡）：见 REPRODUCE.md「Quick start」第 5 步
# RL 链路首步恒等式（比任何单测都硬）：
#   DPO 首步 loss ≈ 0.6931 (-log 0.5)     见 SFT_DPO_DEEP_DIVE.md
#   GRPO 首步 kl == 0 且 ratio == 1        见 GRPO_DEEP_DIVE.md
```

## 5. 资源下载总表

### 模型权重（均为上游/第三方发布，HF 直接下载）

| 权重 | HF 仓库 | 用途 | 大小 |
|---|---|---|---|
| VITA-1.5 | [VITA-MLLM/VITA-1.5](https://huggingface.co/VITA-MLLM/VITA-1.5) | 基座（音频编码器在仓库内 `audio-encoder-Qwen2-7B-1107-…` 子目录，无需单独下载） | ~16 GB |
| InternViT-300M-448px | [OpenGVLab/InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px) | 视觉塔 | ~0.6 GB |
| Qwen2.5-3B-Instruct | [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) | GRPO R2 的 LLM Judge 奖励模型 | ~6 GB |

下载后必须跑 `tools/localize_config.py` 把 checkpoint 里的 HF repo ID
改成本地路径（离线加载），见 REPRODUCE.md「Quick start」第 3 步。

### 训练数据集

| 数据集 | HF 仓库 | 用途 | 转换脚本 |
|---|---|---|---|
| RLAIF-V | [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset) | DPO 偏好对 / SFT / GRPO R1–R3 | `tools/make_rlaif_v_data.py` 等 |
| CLEVR-70k 计数 | [leonardPKU/clevr_cogen_a_train](https://huggingface.co/datasets/leonardPKU/clevr_cogen_a_train) | GRPO R4–R6 可验证奖励训练（R1-V 同款） | `tools/make_clevr_grpo_data.py` / `make_clevr_sft_data.py` / `make_clevr_stage2_data.py` |

### 评测集

| 评测集 | 来源 | 用途 |
|---|---|---|
| SuperCLEVR test200 | [jigsaw-r1/super_clevr](https://huggingface.co/datasets/jigsaw-r1/super_clevr)（HF，R1-V 同款） | OOD 泛化评测，`tools/make_superclevr_eval_data.py` 转换 |
| MME / POPE / MMBench_DEV_EN_V11 | VLMEvalKit 首次评测时自动下载到 `~/LMUData/`（tsv；证书过期需 `curl -k` + MD5 校验的坑见 EXPERIMENT_LOG.md §11.6） | 通用基准回归 |
| CLEVR held-out 500 / stage-2 held-out | 从 CLEVR-70k 切分（脚本固定随机种子，可精确复现） | GRPO 专项评测 |

### 本仓库自产的模型

GRPO 时代的 5 个 LoRA adapter（R4 / SFT 对照 / R5 两臂 / R6-β0，每个 ~310 MB）
已上传 [lee31221/VITA-RL](https://huggingface.co/lee31221/VITA-RL)，
用 `tools/merge_and_eval.py` 合并进基座即可精确复原任何一轮的评测模型。
合并后的完整模型（~16 GB/个）未上传（adapter + 基座可复原，无需冗余）。
DPO 时代的 adapter 随上一台开发机丢失，只能按 REPRODUCE.md 重训。
