# 复现日志

本 fork 的环境搭建与复现记录。上游的 `requirements.txt` 并不能在所有机器上顺利安装；本文记录**实际可用**的方案，以及每一处偏离上游的原因。

> 语言：[English](./REPRODUCE.md) | **中文**
>
> 配套文档：[README_zh-CN.md](./README_zh-CN.md) 是项目说明，[ARCHITECTURE_zh-CN.md](./ARCHITECTURE_zh-CN.md) 是代码走读。

以下内容均在下一节所述机器上验证通过。路径统一写作 `$VITA_WEIGHTS` / `$VITA_REPO`，因此在另一台机器上只需设置这两个变量即可直接粘贴执行：

```bash
export VITA_REPO=/path/to/VITA-RL          # 本仓库
export VITA_WEIGHTS=/path/to/weights       # 需约 25 GB 可用空间
cd "$VITA_REPO"
```

如果你的机器没有外网直连出口，请先设置代理——conda、pip、HuggingFace、git 各步骤都需要：

```bash
export http_proxy=http://<host>:<port> https_proxy=http://<host>:<port>
export no_proxy=localhost,127.0.0.1
```

## 验证环境

| 项目 | 取值 |
|---|---|
| GPU | 8 × NVIDIA H800 80GB |
| NVIDIA 驱动 | 535.129.03（支持 CUDA ≤ 12.2） |
| 系统 CUDA 工具链 | 11.4（`nvcc`）—— **未使用**，见下文 |
| 系统 gcc | 4.8.5 —— 过旧，无法编译 C99/C++17 源码 |
| conda | 4.14.0 |
| Python | 3.10.18 |
| 网络 | 无外网直连；GitHub / HuggingFace 需 HTTP 代理 |

由于系统工具链过旧，**所有依赖均从预编译 wheel 安装**（`--only-binary=:all:`）。wheel 自带 CUDA 12.1 运行时，因此系统那个 11.4 工具链无关紧要——**只有驱动版本重要**，535 已经足够新。

**运行本项目的最低要求**：推理需 1 张 80GB GPU；**全参数训练需 8 张**（见[显存说明](#显存说明)）。驱动需支持 CUDA ≥ 12.1。权重约占 25 GB 磁盘，每保存一个 checkpoint 再加约 16 GB。

## 快速开始

假设上述变量已设置，完整复现流程如下：

```bash
# 1. 环境（各处版本固定的原因见下一节）
conda create -n vita python=3.10 -y && conda activate vita
#    ... 分步安装，见下方"安装顺序" ...

# 2. 权重（约 20 GB）
python -c "
from huggingface_hub import snapshot_download
import os
w = os.environ['VITA_WEIGHTS']
snapshot_download('VITA-MLLM/VITA-1.5', local_dir=f'{w}/VITA-1.5')
snapshot_download('OpenGVLab/InternViT-300M-448px', local_dir=f'{w}/InternViT-300M-448px')"

# 3. 把 checkpoint 里的编码器路径指向本地，而非 HF repo ID
python tools/localize_config.py \
    --model-path  "$VITA_WEIGHTS/VITA-1.5" \
    --vision-tower "$VITA_WEIGHTS/InternViT-300M-448px"

# 4. 推理
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
    --model_path "$VITA_WEIGHTS/VITA-1.5" \
    --image_path asset/vita_newlog.jpg \
    --model_type qwen2p5_instruct --conv_mode qwen2p5_instruct \
    --question "Describe this image."

# 5. 训练链路 smoke test（需 8 卡）
python tools/make_smoke_data.py --out-dir "$VITA_WEIGHTS/smoke_data"
export VITA_SMOKE_DATA_DIR="$VITA_WEIGHTS/smoke_data"
WEIGHTS_ROOT="$VITA_WEIGHTS" bash script/train/smoke_test_qwen.sh /tmp/smoke_out 8
```

## 环境搭建

```bash
conda create -n vita python=3.10 -y
conda activate vita
export PYTHONPATH=./
```

### 安装顺序（顺序很重要）

```bash
# 1. 先装 torch 全家桶，并把 pillow 固定到有 cp310 wheel 的版本
pip install --only-binary=:all: "pillow==10.4.0" \
    torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1

# 2. 核心依赖。xformers 必须固定版本，否则会拖来更新的 torch
pip install --only-binary=:all: "numpy<2" "xformers==0.0.27" \
    "transformers==4.41.1" accelerate decord Jinja2 ninja tqdm

# 3. 代码实际 import 但 requirements.txt 里没列的包
pip install --only-binary=:all: timm einops PyYAML \
    "opencv-python-headless==4.10.0.84" "soundfile==0.12.1" librosa \
    sentencepiece protobuf six

# 4. 重新固定 numpy：numba/librosa 会把 numpy 2.x 拉回来
pip install --only-binary=:all: "numpy==1.26.4" "numba<0.61" "llvmlite<0.44"

# 5. 仅训练需要
pip install "deepspeed==0.14.4"

# 5b. 仅 LoRA（单卡训练）需要。peft 0.11.x 是与 transformers 4.41.1 同期的版本。
pip install --only-binary=:all: "peft==0.11.1"

# 6. flash-attn：内网源没有 wheel，而 gcc 4.8.5 编不了。
#    使用官方预编译 wheel，需匹配 cp310 / torch2.3 / cxx11abiFALSE。
#    用这条命令查你的 ABI：python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

### 与上游 `requirements.txt` 的偏离

| 包 | 上游 | 本项目 | 原因 |
|---|---|---|---|
| `xformers` | 未固定 | `0.0.27` | 不固定会解析到 0.0.35，它要求 `torch>=2.10`，与固定的 `torch==2.3.1` 冲突，安装直接中断。 |
| `pillow` | 未固定 | `10.4.0` | 不固定会解析到 12.3.0，此处无 cp310 wheel，转而源码编译并在 gcc 4.8.5 下失败（`'for' loop initial declarations are only allowed in C99 mode`）。 |
| `numpy` | 未固定 | `1.26.4` | torch 2.3.1 早于 numpy 2.0 的 ABI 断裂。`numba`/`librosa`/`opencv` 都会试图把 numpy 2.x 拉回来——装完它们后需重新固定。 |
| `opencv-python-headless` | 未列出 | `4.10.0.84` | 代码有 import。5.x 版本强制要求 numpy>=2。 |
| `timm`、`einops`、`PyYAML`、`soundfile`、`librosa`、`six`、`sentencepiece` | 未列出 | 最新兼容版 | 代码 import 了但 `requirements.txt` 里缺失。其中 `six` 是 `vita/model/multimodal_encoder/whale/module/encoder/encoder.py` 必需的。 |
| `torchvision` | 未固定 | `0.18.1` | 与 torch 2.3.1 对应的版本。 |
| `flash-attn` | 源码编译 | 预编译 wheel | `train.py` 硬编码了 `attn_implementation="flash_attention_2"`，训练必需。源码编译需要较新的 gcc。 |

### 可忽略的可选组件

以下组件在 import 时会打印警告，但并非必需：

- `apex` —— `Please build and install Nvidia apex package...`
- `mamba_ssm` —— `Please install mamba_ssm to use MambaSSM component.`

### 验证结果

```
torch 2.3.1+cu121   cuda 12.1   8 GPUs visible   bf16 supported
numpy 1.26.4        transformers 4.41.1
flash_attn 2.5.9.post1 —— GPU kernel 可执行
vita.model 导入成功：VITAQwen2ForCausalLM、VITAMixtralForCausalLM、
                     VITAMistralForCausalLM、VITAFOQwen2ForCausalLM  ✅
conv 模式：default、llama、minicpm、mixtral_two、mixtral_zh、
           nemo、phi3、plain、qwen2p5_instruct
```

`pip check` 仅报告 `decord 0.6.0 is not supported on this platform`，这是平台元数据提示而非依赖冲突；decord 实际可正常导入与运行。

精确解析出的版本记录在 [`requirements-lock.txt`](./requirements-lock.txt)。请把它当作**核对结果**用，而非安装路径——直接照它安装可能失败，因为上面的顺序很重要。

在下载 20 GB 权重之前，先做一次环境自检：

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

`audio backends` **必须非空**——空列表意味着音频查询会在后续步骤失败（见 [soundfile 一节](#soundfile-缺少原生库)）。

## 权重

下载到 `$VITA_WEIGHTS`（仓库之外——见 `.gitignore`）：

| 权重 | 来源 | 体积 |
|---|---|---|
| VITA-1.5 | [`VITA-MLLM/VITA-1.5`](https://huggingface.co/VITA-MLLM/VITA-1.5) | 约 19.6 GB（含 `audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning/` 下的音频编码器，约 1.5 GB） |
| InternViT-300M-448px | [`OpenGVLab/InternViT-300M-448px`](https://huggingface.co/OpenGVLab/InternViT-300M-448px) | 约 0.3 GB |

音频编码器就在 VITA-1.5 仓库内部，因此**只需下载两份**，而非上游 README 暗示的三份。两个仓库都是公开的，无需申请权限。

```bash
python -c "
from huggingface_hub import snapshot_download
import os
w = os.environ['VITA_WEIGHTS']
snapshot_download('VITA-MLLM/VITA-1.5', local_dir=f'{w}/VITA-1.5')
snapshot_download('OpenGVLab/InternViT-300M-448px', local_dir=f'{w}/InternViT-300M-448px')"
```

`snapshot_download` 支持断点续传，连接中断后重跑即可。校验完整性：

```bash
python -c "
import json, os
w = os.environ['VITA_WEIGHTS'] + '/VITA-1.5'
idx = json.load(open(f'{w}/model.safetensors.index.json'))
missing = [f for f in set(idx['weight_map'].values()) if not os.path.exists(f'{w}/{f}')]
print('missing shards:', missing or 'none')"
```

## 必须的代码修复

### `cache_position` 与固定的 `transformers==4.41.1` 不兼容

**现象。** `video_audio_demo.py` 在 `generate()` 阶段失败：

```
KeyError: 'cache_position'
    at vita/model/language_model/vita_qwen2.py:250
```

绕过这个之后，紧接着是：

```
TypeError: Qwen2Model.forward() got an unexpected keyword argument 'cache_position'
    at vita/model/language_model/vita_qwen2.py:78
```

（以上行号对应修复前的代码状态。）

**原因。** 在 `transformers==4.41.1`——也就是上游 `requirements.txt` 固定的那个版本——中，`cache_position` 已加入 **Llama** 实现，但**尚未**加入 **Qwen2**：

| | 4.41.1 |
|---|---|
| `LlamaForCausalLM.prepare_inputs_for_generation` 返回 `cache_position` | 是 |
| `Qwen2ForCausalLM.prepare_inputs_for_generation` 返回 `cache_position` | **否** |
| `Qwen2Model.forward` 接受 `cache_position` | **否** |

上游是在**同一天的两个提交**里同时加入这段代码并固定 `transformers==4.41.1` 的（`fe4d74e` / `9a968b3`，均为 2024-12-20），所以对 Qwen2 路径而言，两者从未一致过。4.41.1 中的 `MistralModel` 和 `MixtralModel` 同样不接受该参数。

**已实施的修复**（`vita/model/language_model/vita_qwen2.py`）：

1. 在 `prepare_inputs_for_generation` 中用 `.get()` 读取 `cache_position`，缺失时从 cache 长度推导——`torch.arange(past_length, past_length + input_ids.shape[1])`，这正是 `cache_position` 的语义。后续的 position id 调整加了保护，任一张量不可用时跳过。
2. 在 `custom_forward` 中，仅当已安装的 `Qwen2Model.forward` 确实接受该参数时才传入，通过 `inspect.signature` 在 import 时检测一次。

两处改动都是**版本条件化**的，因此在支持 `cache_position` 的更新版 `transformers` 上，该文件仍按作者原意工作。

`VITAFOQwen2ForCausalLM` 调用 `super().forward(...)`，会解析到被 patch 的 `custom_forward`，因此同样被这个修复覆盖。

> **未修复：** `vita/model/language_model/vita_nemo.py`（第 78、178 行）以同样方式把 `cache_position` 传给 `MistralModel`，在 4.41.1 下会触发完全相同的 `TypeError`。之所以不动，是因为本次复现没有走 Nemo/Mistral 路径，且缺少对应权重无法测试——改了等于交付未验证的代码。

### `soundfile` 缺少原生库

**现象。** 音频查询失败，先打印 `cannot open asset/q1.wav!!!!!!!!!!!!!!!!`，随后在 `whale/init_model.py:40` 抛出 `UnboundLocalError: local variable 'sample_rate' referenced before assignment`——加载器吞掉了真实异常，然后引用了一个未赋值的变量。

**原因。** `torchaudio.list_audio_backends()` 返回 `[]`。最新版 `soundfile` 的 wheel 没有打包 `libsndfile.so`，导致 `import soundfile` 抛出 `OSError: cannot load library 'libsndfile.so'`，而 torchaudio 静默地没有注册任何后端。

**修复。** 固定 `soundfile==0.12.1`，其 wheel 自带原生库：

```bash
pip install --only-binary=:all: "soundfile==0.12.1"
```

之后 `torchaudio.list_audio_backends()` 返回 `['soundfile']`。

## 推理复现

先把 checkpoint 的 `config.json` 指向本地编码器路径，使加载不依赖网络。上游在 `mm_vision_tower` / `mm_audio_encoder` 里存的是 HuggingFace repo ID，不改写的话每次加载都会访问 HF：

```bash
python tools/localize_config.py \
    --model-path   "$VITA_WEIGHTS/VITA-1.5" \
    --vision-tower "$VITA_WEIGHTS/InternViT-300M-448px"
# 会在 config.json.orig 留备份；--restore 可还原
```

README 里的三种快速开始模式均在单张 H800 上跑通：

| 模式 | 命令 | 结果 |
|---|---|---|
| 文本查询 | `--question "Describe this image."` | ✅ 对 VITA logo 的连贯描述，7.4 秒 |
| 音频查询 | `--audio_path asset/q1.wav` | ✅ 中文描述，前缀 `☞`，2.3 秒 |
| 噪声音频 | `--audio_path asset/q2.wav` | ✅ 回复前缀 `☟`，1.9 秒 |

```bash
export PYTHONPATH=./
CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
    --model_path "$VITA_WEIGHTS/VITA-1.5" \
    --image_path asset/vita_newlog.jpg \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --question "Describe this image."
```

`☞` / `☟` 前缀就是 `vita/util/data_utils_video_audio_neg_patch.py`（`preprocess_multimodal`）中定义的状态符号：`☞` 表示对音频查询的回复，`☜` 表示对文本查询的回复，`☟` 表示负样本/噪声音频条件下的回复。`q1.wav` 得到 `☞`、`q2.wav` 得到 `☟`，说明音频编码器与负样本行为都正常工作。

> **README 的一处错误：** 音频示例引用了 `asset/vita_newlog.png`，但仓库里并无此文件——实际是 `asset/vita_newlog.jpg`。

## 训练链路 smoke test

上游无法提供其训练数据，`vita/config/dataset_config.py` 提交上来时路径全为空。为验证训练路径本身可用，本 fork 增加了一个用仓库自带素材构造的**合成**微型数据集。

> 这是**链路检查**，**不是** VITA-1.5 训练的复现。它不能说明任何模型质量问题——只能说明数据加载、多模态 token 展开、collation、前向、反向、checkpoint 保存都能跑通。

### 数据集注册表修复

任何训练脚本能启动之前，必须先修两个上游缺陷：

1. **`DataConfig` 缺少 key。** `vita/config/__init__.py` 只定义了 `Pretrain_video`，但 `pretrain_mlp_qwen.sh`、`finetune_qwen.sh` 等传的是 `--dataset_use Pretrain_video0`，`pretrain_audio_mlp_qwen.sh` 传的是 `Pretrain_audio`。两者都会在 `data_utils_video_audio_neg_patch.py:833` 抛 `KeyError`。现已补上定义。

2. **音频路径布局没有文档。** 加载器解析音频的方式是 `os.path.join(AudioFolder, "audio", file)`——注意中间那个硬编码的 `audio` 段。因此 `AudioFolder` 必须指向 `audio/` 目录的**父目录**，而不是 `audio/` 本身。

合成数据集与注册表条目都通过 `VITA_SMOKE_DATA_DIR` 环境变量**按需启用**，因此不设该变量时，默认配置与上游完全一致。

### 运行方式

```bash
# 会把 asset/ 里的图片和 wav 拷入 <dir>/images 与 <dir>/audio，
# 然后写出 <dir>/smoke_train.json —— 需在仓库根目录执行
python tools/make_smoke_data.py --out-dir "$VITA_WEIGHTS/smoke_data"
export VITA_SMOKE_DATA_DIR="$VITA_WEIGHTS/smoke_data"

WEIGHTS_ROOT="$VITA_WEIGHTS" bash script/train/smoke_test_qwen.sh /tmp/smoke_out 8
```

`smoke_test_qwen.sh` 接收输出目录和 GPU 数量两个参数，并从 `WEIGHTS_ROOT`（或单独的 `MODEL_PATH` / `VISION_TOWER` / `AUDIO_ENCODER` 变量）读取权重位置，脚本自身不含任何绝对路径。

数据集覆盖了加载器分支处理的三种样本形态，各自产生预期的状态符号：

| 样本类型 | 字段 | 状态符号 |
|---|---|---|
| 纯图像 | `image` | `☜`（对文本查询的回复） |
| 图像 + 音频 | `image`、`audio` | `☞`（对音频查询的回复） |
| 图像 + 音频 + 负样本 | 额外加 `inserted_id` | `☞`，以及被标记轮次上的 `☟` |

### 运行结果

```
{'loss': 3.1885, 'grad_norm': 45.93, 'learning_rate': 1e-06, 'epoch': 0.33}
{'loss': 3.7144, 'grad_norm': 55.69, 'learning_rate': 5e-07, 'epoch': 0.67}
{'loss': 2.7051, 'grad_norm': 38.45, 'learning_rate': 0.0,   'epoch': 1.0}
{'train_runtime': 14.68, 'train_loss': 3.2026, 'epoch': 1.0}
```

loss 有限、梯度正常回传。但只有 3 步、`lr=1e-6`、24 条合成样本，**loss 轨迹本身没有任何意义**——这是故意的，学习率刻意设得极小，以免这次运行被误当成"训练过了"。

checkpoint 保存单独验证过（`--save_strategy steps --save_steps 2`，`smoke_test_qwen.sh` 默认关闭该项）。`VITATrainer._save_checkpoint` 正常写出完整的 4 分片模型加 ZeRO-3 优化器状态，且产出的 checkpoint 用 `video_audio_demo.py` **能正常加载并生成**——形成了从训练回到推理的闭环。

> 一个完整 checkpoint 约 16 GB，保留优化器状态的运行可达约 130 GB。请把 `--output_dir` 指向仓库之外；`.gitignore` 已排除 `outputs/` 和 `checkpoint-*/` 作为兜底。

### 显存说明

7B 全参数训练在单张 80GB H800 上**放不下**。单卡运行会完成前向和反向，然后在分配 AdamW 状态时失败：

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.84 GiB
    at torch/optim/adamw.py:128, state["exp_avg_sq"] = torch.zeros_like(...)
```

AdamW 为每个参数保存两个 fp32 动量（7B 约 56 GB），外加 fp32 主权重（约 28 GB）。ZeRO-3 会把这些切分到各 rank，因此**需要 8 张卡**；同样的运行在 8 卡上成功。卡数不足时可用 LoRA（`--lora_enable True`）——但请注意那条代码路径是从 LLaVA 继承来的，上游没有任何脚本验证过它。

## 进度

- [x] conda 环境 + 全部依赖
- [x] flash-attn 在 GPU 上可用
- [x] `vita.model` 可正常导入
- [x] 权重下载完成（VITA-1.5 19.6 GB + InternViT 0.3 GB）
- [x] 推理复现——文本、音频、噪声音频三种查询
- [x] 训练链路在合成数据上端到端验证（8 × H800）
- [ ] 真实数据训练（需要上游未提供的数据集）

## 在另一台机器上复现

以上内容与路径无关：设置 `VITA_REPO` 和 `VITA_WEIGHTS`，按需加代理，然后照[快速开始](#快速开始)执行即可。换机器时最可能出现分歧的点：

| 如果你的机器 | 那么 |
|---|---|
| Python 或 torch 版本不同 | 固定的 flash-attn wheel 不会匹配。请从 [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases) 挑选与你的 `cp3XX` / `torchX.Y` / ABI 对应的版本。 |
| gcc ≥ 7 且 CUDA 工具链匹配 | 可以源码编译 flash-attn（`pip install flash-attn --no-build-isolation`），不必用 wheel。 |
| GPU 少于 8 张 | 推理单卡即可。全参数训练不行——见[显存说明](#显存说明)。 |
| 有外网直连 | 跳过代理设置；但 `tools/localize_config.py` 仍值得执行，以免加载时访问网络。 |
| 基于 numpy 2 的技术栈 | 不要升级：torch 2.3.1 早于 numpy 2.0 的 ABI 断裂。 |

本机上两次独立运行——第二次完全按上文所写的命令驱动——得到 `loss = 3.1885, 3.7144, 2.705x`。得益于 `train.py` 中固定的随机种子（`set_random_seed(42)`），**前两步字节级一致**；第三步在小数第四位有差异（`2.7051` vs `2.7104`），这是分布式 bf16 归约顺序造成的常见非确定性。**应当预期数值接近，而非完全相同。**
