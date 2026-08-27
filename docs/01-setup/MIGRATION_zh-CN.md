# 迁移到新机器

本文面向"在一台全新机器上从零重建本项目"的场景。

> 语言：**中文** | [English](./MIGRATION.md)

## TL;DR

git 里只有代码和文档（约 11 MB）。**权重（19.6 GB）和 conda 环境（6.6 GB）都不在 git 里**，需要在新机器上重新获取——这是重建时的主要耗时。

```bash
git clone https://github.com/eternity-blog/VITA-RL.git
cd VITA-RL
# 然后照 REPRODUCE_zh-CN.md 的"快速开始"走一遍
```

## 一、git 里有什么、没有什么

| | 内容 | 体积 | 新机器如何获得 |
|---|---|---|---|
| ✅ 在 git 里 | 全部代码、整套文档（见 README 文档导览表）、工具脚本、`requirements-lock.txt` | ~11 MB | `git clone` |
| ❌ 不在 git 里 | VITA-1.5 权重 | 19.6 GB | 从 HuggingFace 重新下载 |
| ❌ 不在 git 里 | InternViT-300M-448px | 0.65 GB | 从 HuggingFace 重新下载 |
| ❌ 不在 git 里 | conda 环境 `vita` | 6.6 GB | 按文档重装（约 10-20 分钟） |
| ❌ 不在 git 里 | 合成 smoke 数据 | 极小 | `python tools/make_smoke_data.py` 一条命令重建 |

这是刻意的设计：`.gitignore` 排除了权重和训练产物，否则仓库会膨胀到几十 GB 且无法推送。

## 二、重建步骤

### 1. clone

```bash
git clone https://github.com/eternity-blog/VITA-RL.git
cd VITA-RL
export VITA_REPO=$(pwd)
export VITA_WEIGHTS=/path/to/weights     # 需 ~25 GB 可用空间
```

如果新机器也没有外网直连，先设代理（clone / pip / HuggingFace 都需要）：

```bash
export http_proxy=http://<host>:<port> https_proxy=http://<host>:<port>
export no_proxy=localhost,127.0.0.1
```

### 2. 环境

**不要**直接 `pip install -r requirements.txt`（上游那份装不上，原因见
[REPRODUCE_zh-CN.md](./REPRODUCE_zh-CN.md#与上游-requirementstxt-的偏离)）。
按 [REPRODUCE_zh-CN.md 的分步安装](./REPRODUCE_zh-CN.md#安装顺序顺序很重要)执行。

装完先自检，再决定要不要下 20 GB 权重：

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

### 3. 权重

```bash
python -c "
from huggingface_hub import snapshot_download
import os
w = os.environ['VITA_WEIGHTS']
snapshot_download('VITA-MLLM/VITA-1.5', local_dir=f'{w}/VITA-1.5')
snapshot_download('OpenGVLab/InternViT-300M-448px', local_dir=f'{w}/InternViT-300M-448px')"
```

支持断点续传，中断后重跑即可。

### 4. 本地化 config（重要）

```bash
python tools/localize_config.py \
    --model-path   "$VITA_WEIGHTS/VITA-1.5" \
    --vision-tower "$VITA_WEIGHTS/InternViT-300M-448px"
```

⚠️ **这一步必须在新机器上重做。** 它写的是绝对路径，旧机器的路径在新机器上无效。
如果你是把权重目录整个拷贝过去的，`config.json` 里仍然是**旧机器的路径**，
必须重新执行这条命令（或先 `--restore` 再执行）。

### 5. 验证

```bash
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
    --model_path "$VITA_WEIGHTS/VITA-1.5" \
    --image_path asset/vita_newlog.jpg \
    --model_type qwen2p5_instruct --conv_mode qwen2p5_instruct \
    --question "Describe this image."
```

看到对 VITA logo 的连贯描述即成功。

## 三、容易踩的坑

| 坑 | 说明 |
|---|---|
| **直接拷贝 conda 环境目录** | 不可靠。环境里有编译好的扩展和写死的绝对路径。请按文档重装。 |
| **拷贝权重后忘了改 config** | `config.json` 里是旧机器的绝对路径，模型会加载失败或回退到访问网络。重跑 `tools/localize_config.py`。 |
| **flash-attn wheel 不匹配** | 文档钉的是 `cp310 / torch2.3 / cxx11abiFALSE`。新机器若 Python 或 torch 版本不同，需换 wheel，用 `python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"` 查 ABI。 |
| **GPU 数量不足** | 推理 1 张卡够。**全参数训练需要 8 张**，单卡会在分配 AdamW 状态时 OOM。 |
| **忘了设代理** | 若新机器无外网直连，`git clone`、`pip`、`snapshot_download` 全都会卡住。 |
| **numpy 被升级到 2.x** | torch 2.3.1 不兼容。装完 numba/librosa/opencv 后要重新固定 `numpy==1.26.4`。 |
| **`PYTHONPATH=./` 忘了导** | 会 `ModuleNotFoundError: No module named 'vita'`。 |

## 四、如果想省下重新下载权重

权重可以直接拷贝（比重新下载快），但**拷完必须重做第 4 步**：

```bash
# 旧机器
tar -cf - -C /path/to weights | ssh newhost 'tar -xf - -C /path/to'

# 新机器：把 config 指向新路径
python tools/localize_config.py \
    --model-path   "$VITA_WEIGHTS/VITA-1.5" \
    --vision-tower "$VITA_WEIGHTS/InternViT-300M-448px"
```

## 五、迁移后自查清单

- [ ] `git log --oneline -1` 与 GitHub 上的最新 commit 一致
- [ ] `git status` 干净
- [ ] 环境自检脚本全部通过，`audio backends` **非空**
- [ ] `python -c "import json;print(json.load(open('$VITA_WEIGHTS/VITA-1.5/config.json'))['mm_vision_tower'])"` 显示的是**新机器**的路径
- [ ] 文本推理跑通
- [ ] 音频推理跑通（`q1.wav` 回复以 `☞` 开头，`q2.wav` 以 `☟` 开头）
- [ ] 如需训练：`bash script/train/smoke_test_qwen.sh /tmp/smoke_out 8` 跑通

## 六、后续工作的起点

项目当前进度见 [README_zh-CN.md 的路线图](../../README_zh-CN.md#路线图)：推理、训练链路
与两条 RL 线均已完成——DPO 端到端测量（POPE 幻觉率 10.97% → 8.82%，见
[EXPERIMENT_LOG.md](../03-experiments/EXPERIMENT_LOG.md)），多模态 GRPO 在可验证奖励任务上
训练成功（held-out 准确率 44.6% → 77.4%，见
[GRPO_DEEP_DIVE.md](../03-experiments/GRPO_DEEP_DIVE.md)）。RL 栈代码走读见
[ARCHITECTURE_zh-CN.md 第 14 章](../00-background/ARCHITECTURE_zh-CN.md)。
