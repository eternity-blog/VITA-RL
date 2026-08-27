# 上手手册：本机实操

> [PRIMER.md](../00-background/PRIMER.md) 讲的是**原理**，这份讲的是**动手**：本机的确切路径、
> 可直接复制的命令、改代码时的自检流程，以及各条路径的实测状态。
>
> 所有命令都在本机（8×H100 / `/usr/local/kai/lx/VITA-RL`）验证过。
> 凡是我**没跑过**的，本文明确标注为未验证，不含糊过去。
>
> **范围说明**：本 fork 的 RL 工作只针对文本+图片/视频。§2.1 的语音推理
> 命令是上游能力的验证，音频编码器在本 fork 的所有训练里都是冻结组件。

## 目录

- [1. 五秒进入工作状态](#1-五秒进入工作状态)
- [2. 常用命令速查](#2-常用命令速查)
- [3. 改代码时的自检流程](#3-改代码时的自检流程)
- [4. 接入自己的数据集](#4-接入自己的数据集)
- [5. 各条路径的实测状态](#5-各条路径的实测状态)
- [6. 地雷区](#6-地雷区)
- [7. 故障排查](#7-故障排查)
- [8. DPO（离线偏好优化）](#8-dpo离线偏好优化)
- [9. GRPO（组相对策略优化）](#9-grpo组相对策略优化)
- [10. 本机资源现状](#10-本机资源现状)

---

## 1. 五秒进入工作状态

新开终端后，这段直接粘贴：

> ⚠️ **下面的路径是原机器（8×H100）的。** 换机器后把
> `VITA_REPO` 和 `VITA_WEIGHTS` 改成你自己的位置，其余不变。
> 重建步骤见 [MIGRATION.md](./MIGRATION.md)。

```bash
export VITA_REPO=/usr/local/kai/lx/VITA-RL
export VITA_WEIGHTS=/usr/local/kai/lx/vita_weights
export VITA_SMOKE_DATA_DIR=$VITA_WEIGHTS/smoke_data
export PATH=/root/anaconda3/envs/vita/bin:$PATH
export PYTHONPATH=$VITA_REPO
cd $VITA_REPO
```

建议存成 `~/vita_env.sh`，之后 `source ~/vita_env.sh` 即可。

**为什么不用 `conda activate`**：本机 conda 4.9.2 在非交互 shell 里需要先
`source conda.sh`，直接改 `PATH` 更省事，效果一样。

验证就位：

```bash
python -c "import torch; print('torch', torch.__version__, '| GPUs', torch.cuda.device_count())"
# 期望：torch 2.3.1+cu121 | GPUs 8
```

**`PYTHONPATH` 建议始终设置**。在仓库根目录下运行时不设也能 import（当前目录
恰好在搜索路径里），但一旦 `cd` 到别处就会 `ModuleNotFoundError: No module
named 'vita'`——这是最常见的第一个报错。

## 2. 常用命令速查

### 2.1 推理三种模式

```bash
# 文本提问（输出应以 ☜ 开头）
CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
    --model_path "$VITA_WEIGHTS/VITA-1.5" \
    --image_path asset/vita_newlog.jpg \
    --model_type qwen2p5_instruct --conv_mode qwen2p5_instruct \
    --question "Describe this image."

# 语音提问（输出应以 ☞ 开头）
CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
    --model_path "$VITA_WEIGHTS/VITA-1.5" \
    --image_path asset/vita_newlog.jpg \
    --model_type qwen2p5_instruct --conv_mode qwen2p5_instruct \
    --audio_path asset/q1.wav

# 噪声音频（输出应以 ☟ 开头 = 拒答）
#   把上面的 q1.wav 换成 q2.wav
```

⚠️ **`--question` 和 `--audio_path` 只能给一个**
（`video_audio_demo.py:145` 有断言），不能同时给。

实测耗时：首次约 8.7 s（含编译），之后 1.6–2.9 s。

### 2.2 检视数据（不用 GPU，秒级）

```bash
python tools/inspect_dataset.py --dataset-use SmokeTest --num-samples 3
```

输出会告诉你：序列长度、**哪段文本被监督**、状态 token 前缀、
collator 后的张量形状、以及有没有样本的 label 被静默作废。

**改完数据配置后先跑这个**，再考虑开 8 卡。

### 2.3 单元测试（不用 GPU，不用权重）

```bash
python tools/test_audio_optional.py     # 期望结尾 ALL CHECKS PASSED
```

### 2.4 8 卡训练冒烟

```bash
bash script/train/smoke_test_qwen.sh /tmp/smoke_out 8
# 需要 WEIGHTS_ROOT 或 VITA_WEIGHTS，以及 VITA_SMOKE_DATA_DIR
```

实测：3 步 15.4 s，每卡峰值约 18 GB，产出 16.6 GB checkpoint。
**跑完记得 `rm -rf /tmp/smoke_out`**，否则很快吃满磁盘。

### 2.4b 单卡 LoRA 训练

```bash
bash script/train/smoke_test_lora.sh /tmp/lora_out 1     # 末位是 GPU 编号
```

实测：24 步 10.4 s，峰值 **23.3 GB**（单卡即可），产出 308 MB adapter。
详见 §5。

### 2.4d 单卡 DPO 训练

```bash
python tools/make_dpo_smoke_data.py --out-dir $VITA_WEIGHTS/dpo_smoke_data
export VITA_DPO_DATA_DIR=$VITA_WEIGHTS/dpo_smoke_data
bash script/train/dpo_smoke_test.sh /tmp/dpo_out 1
```

**看首步 loss 是否等于 0.6931** —— 这是判断参考模型接对没有的最强信号，
见 §8.2。

### 2.4e 单卡 GRPO 训练

```bash
python tools/make_grpo_smoke_data.py --out-dir $VITA_WEIGHTS/grpo_smoke_data
export VITA_GRPO_DATA_DIR=$VITA_WEIGHTS/grpo_smoke_data
bash script/train/grpo_smoke_test.sh /tmp/grpo_out 1
```

**看首步 `grpo/kl` 是否为 0**（参考模型正确）、
**`reward/mean` 是否上升**（核心信号）、
**`groups/degenerate_frac` 是否接近 1.0**（接近则奖励无区分度）。见 §9。

### 2.4c 视频推理

```bash
CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
    --model_path "$VITA_WEIGHTS/VITA-1.5" \
    --video_path /path/to/video.mp4 \
    --model_type qwen2p5_instruct --conv_mode qwen2p5_instruct \
    --question "What is happening in this video?"
```

抽帧规则：`video_framerate=1` 即每秒 1 帧，上限 `MAX_IMAGE_LENGTH=16` 帧。
5 秒视频 → 5 帧 → 1280 token；超过 16 秒的会被降采样到 16 帧（4096 token，
占默认上下文 66%）。

### 2.5 加载训练产出的 checkpoint

产出的 config 会继承本地路径，**不需要**再跑 `localize_config.py`：

```bash
CUDA_VISIBLE_DEVICES=0 python video_audio_demo.py \
    --model_path /tmp/smoke_out/smoke-finetune_task_neg \
    --image_path asset/vita_newlog.jpg \
    --model_type qwen2p5_instruct --conv_mode qwen2p5_instruct \
    --question "Describe this image in one sentence."
```

### 2.6 看训练日志的关键行

```bash
# loss 曲线
grep -oE "\{'loss'[^}]*\}" /tmp/smoke_run.log

# 静默作废的样本数 —— 真实数据训练时务必看这个
grep -c "tokenization mismatch" /tmp/smoke_run.log

# 失败签名
grep -nE "Traceback|OutOfMemory|unused parameter|did not receive grad" /tmp/smoke_run.log
```

### 2.7 跑 benchmark 评测

四个必需的环境变量，缺一个都会以难懂的方式失败：

```bash
export PATH=/root/anaconda3/envs/vita/bin:$PATH
export PYTHONPATH=$VITA_REPO          # 否则 vlmeval 找不到 vita 包
export VITA_CKPT=$VITA_WEIGHTS/VITA-1.5   # config.py 读它来定位权重
export LMUData=/root/LMUData          # 见下：不设会重新下载数据集

cd $VITA_REPO/VLMEvalKit
CUDA_VISIBLE_DEVICES=0 python run.py --data MME --model vita_qwen2 \
    --work-dir /path/to/eval_out/baseline
```

**`LMUData` 是最坑的一个**。`vlmeval/smp/file.py:71` 把数据集根目录
硬编码成上游作者的 `/mnt/cfs/lhj`，本机没有，于是它**无视你已经
下好的 tsv、每次都试图重新下载**——而 opencompass 的证书又是过期的，
最后你看到的是一个 SSL 报错，跟真正的原因隔了三层。

**不需要 GPT judge 的 benchmark**（规则判分，能离线跑完）：
`MME`、`MMStar`、`MMBench_DEV_EN_V11`、`AI2D_TEST`、`OCRBench`。
`MMVet`、`MathVista`、`LLaVABench` 需要配 `.env` 里的 judge model。

对比两次评测：

```bash
python tools/compare_eval.py \
    --before /path/to/eval_out/baseline/vita_qwen2 \
    --after  /path/to/eval_out/dpo/vita_qwen2
```

它会在 Overall 旁边打印 1.96σ 噪声带。**MMStar 上这个带是 ±2.5 个点**——
比它小的变化不要当成提升。

评测一个 LoRA adapter 需要先合并，因为 VLMEvalKit 的 wrapper
只接受一个模型路径、没有 adapter 参数：

```bash
python tools/merge_and_eval.py \
    --base $VITA_WEIGHTS/VITA-1.5 \
    --adapter /path/to/dpo-rlaif-v \
    --out $VITA_WEIGHTS/VITA-1.5-dpo
```

## 3. 改代码时的自检流程

按代价从低到高，能在前面发现的问题就别拖到后面。

```
1. python tools/test_audio_optional.py          几秒，CPU
2. python tools/inspect_dataset.py              几十秒，CPU
3. 单卡推理一次                                  约 30 秒，1 GPU
4. bash script/train/smoke_test_qwen.sh ... 8   约 4 分钟，8 GPU
5. 真实训练
```

**改了 `vita_arch.py` / `vita_qwen2.py`（模型前向）之后**，第 3 步和第 4 步都要跑：
推理验证生成路径，冒烟验证训练路径。这两条路径在 `prepare_inputs_labels_for_multimodal`
里共用代码，但一个走 `generate()` 一个走 `forward()`。

**判断改动是否等价的方法**：冒烟测试的 loss 是可复现的。我修 `audios=None`
时的对照结果：

| 步 | 修复前 | 修复后 |
|---|---|---|
| 1 | 3.1885 | 3.1885 |
| 2 | 3.7144 | 3.7144 |
| 3 | 2.7124 | 2.7129 |

前两步逐位相同，第三步差 5e-4 —— 这是 bf16 多卡归约顺序的正常抖动，
不是行为改变。**如果你的改动让第一步 loss 就变了，那就是真的改了行为。**

## 4. 接入自己的数据集

四步，顺序不能乱。

### 步骤 1：数据转成 VITA 格式

```json
{
  "set": "mydata",
  "id": "sample_0001",
  "conversations": [
    {"from": "human", "value": "<image>\n描述这张图。"},
    {"from": "gpt",   "value": "这是一只猫。"}
  ],
  "image": "cats/001.jpg",
  "audio": ["wavs/q001.wav"],
  "inserted_id": 3
}
```

- `image` 路径是**相对于** `FolderDict[set]` 的
- `audio` 路径拼接方式是 `os.path.join(AudioFolder, "audio", file)`
  ——中间那个 `"audio"` **是硬编码的**，所以 `AudioFolder` 要指向
  `audio/` 的**父目录**
- `inserted_id` 可选，标记哪一轮 gpt 回复是负样本（会被打 `☟`）
- `"gpt"` 只是 LLaVA 沿用的标签，意思是「标准答案」，与模型无关

### 步骤 2：注册到 `vita/config/dataset_config.py`

```python
FolderDict = {
    "sharegpt4": "",
    "mydata": "/abs/path/to/images",      # 新增
}
MyData = {"chat_path": "/abs/path/to/mydata.json"}
```

### 步骤 3：加进 `vita/config/__init__.py`

```python
DataConfig = {
    ...
    "MyData": [MyData],       # 新增
}
```

### 步骤 4：验证后再训练

```bash
python tools/inspect_dataset.py --dataset-use MyData --num-samples 5
```

**重点看 `supervised=` 那一栏**。如果是 0，说明 label 被静默作废了
（见 §6），此时训练能跑但学不到任何东西。

确认无误后，把训练脚本的 `--dataset_use` 改成 `MyData`。

## 5. 各条路径的实测状态

诚实标注，避免你在未验证的路径上浪费时间。

| 路径 | 状态 | 说明 |
|---|---|---|
| 文本/语音/噪声推理 | ✅ **本机跑通** | 三种状态 token 正确区分 |
| 8 卡 ZeRO-3 训练 | ✅ **本机跑通** | 3 步，checkpoint 可重载 |
| checkpoint 重载推理 | ✅ **本机跑通** | 1.6 s |
| `tools/` 四个工具 | ✅ **本机跑通** | localize / make_smoke / inspect / test |
| **单卡 LoRA 训练** | ✅ **本机跑通（需先修复）** | 峰值 **23.3 GB**，24 步 10.4 s。上游此路径有致命 bug，本 fork 已修，见下 |
| **视频推理** | ✅ **本机跑通** | 每秒抽 1 帧，上限 16 帧 |
| **多模态 DPO（真实数据）** | ✅ **跑通到终局** | RLAIF-V 上 SFT→DPO，POPE 幻觉率 10.97%→8.82%（McNemar p<1e-4）。见 [EXPERIMENT_LOG.md](../03-experiments/EXPERIMENT_LOG.md) |
| **多模态 GRPO（真实数据）** | ✅ **跑通到终局** | CLEVR 计数 + 可验证奖励，400 步 held-out 准确率 44.6%→77.4%（win rate 0.977）。见 [GRPO_DEEP_DIVE.md](../03-experiments/GRPO_DEEP_DIVE.md) |
| **GRPO 对照与边界实验** | ✅ **闭环** | 通用回归（MME/POPE/MMBench **零退化**）、SuperCLEVR OOD（37.5%→54.5%）、配平 SFT 对照（26min 达 75.4%，OOD 63.0% 反超）、R5 阶段二对照（任务天花板 ~77–78%）。见 [GRPO_DEEP_DIVE.md §10](../03-experiments/GRPO_DEEP_DIVE.md) 与 [EXPERIMENT_LOG.md §14](../03-experiments/EXPERIMENT_LOG.md) |
| web demo | ❌ **不可用** | `flask`/`flask_socketio`/`vllm`/`onnxruntime` 均未装；还缺 `web_demo/wakeup_and_vad/resource/` 下的 `silero_vad.onnx`/`.jit` |
| VLMEvalKit 评测 | ✅ **已跑通** | omegaconf/antlr4 冲突已解（见下），基线与 DPO 对比数字已产出：MME 2353.5、MMBench 77.8、POPE 见上行。历史排障记录保留在下节 |
| Video-MME 评测 | ⚠️ 未测 | 需另下 Video-MME 数据集 |
| 语音输出（TTS） | ⚠️ 存疑 | 见 [PRIMER.md §9](../00-background/PRIMER.md#9-语音输出论文与代码不一致)——代码与论文描述不一致 |

### VLMEvalKit：进行到哪一步了

**已完成**：`VLMEvalKit/vlmeval/config.py` 改为读 `VITA_CKPT` 环境变量
（原本硬编码 `/path/to/model`），并补上缺失的 `import os`。用法：

```bash
export VITA_CKPT=$VITA_WEIGHTS/VITA-1.5
```

**依赖安装的关键原则**：**不要** `pip install -r VLMEvalKit/requirements.txt`。
那份需求没有固定 `transformers`/`torch`/`numpy` 版本，直接装会升级
transformers，从而废掉 `vita_qwen2.py` 对 4.41.1 的 monkey patch，
整个项目都跑不了。

正确做法是全程 `--no-deps`：

```bash
pip install --only-binary=:all: --no-deps \
  pandas openpyxl portalocker rich sty tabulate tiktoken validators \
  xlsxwriter omegaconf imageio matplotlib python-dotenv "openai==1.3.5"
pip install --no-deps timeout-decorator      # 只有源码包，需放开 --only-binary
# --no-deps 会漏掉传递依赖，补上：
pip install --only-binary=:all: --no-deps \
  pytz python-dateutil tzdata six markdown-it-py mdurl pygments \
  contourpy cycler fonttools kiwisolver pyparsing et-xmlfile \
  regex httpx distro annotated-types
```

装完务必验证核心栈没被动过：

```bash
python -c "import torch,transformers,numpy; print(torch.__version__, transformers.__version__, numpy.__version__)"
# 必须仍是 2.3.1+cu121 / 4.41.1 / 1.26.4
```

**已解决的坑**：`moviepy`。VLMEvalKit 的 `mvbench.py` 用
`from moviepy.editor import ...`，但 moviepy 2.x 移除了该模块，
而镜像上没有 1.0.3。麻烦在于 `mvbench.py` 被 `dataset/__init__.py`
**无条件 import**，所以一个视频数据集的依赖问题会让**所有图像基准**
都加载不了。处理方式是装 moviepy 2.x 再加一个 `.pth` 垫片，
把 `moviepy.editor` / `moviepy.config_defaults` 注册为 2.x 顶层符号的别名
（不改 VLMEvalKit 源码）。

**当时未解决、后来已解决的坑**：`omegaconf 2.3.1` 需要
`antlr4-python3-runtime 4.9.x`，而企业镜像最低只有 4.11，导致
`Exception: Could not deserialize ATN with version 3 (expected 4)`。
omegaconf 被 `vlmeval/vlm/vxverse.py` 在模块顶层 import，同样会拖垮整个包。
**最终解法**（2026-08-20，见 `script/dev/setup_vlmeval_deps_dev.sh`）：从源码装
antlr4 4.9.3（纯 Python，无需编译）。此后 VLMEvalKit 已完整跑通，
产出基线与 DPO/GRPO 对比数字（POPE / MME / MMBench）。

### 单卡 LoRA：上游的致命 bug（本 fork 已修）

装上 `peft` 后直接跑 `--lora_enable True`，**必然崩溃**：

```
ValueError: Target module Qwen2DecoderLayer(...) is not supported.
```

根因链（已逐环实证）：

1. `find_all_linear_names`（`train.py:157`）用 `names[-1]` 取叶子名
2. whale 里有 **2 个 `nn.Linear` 的叶子名是纯数字 `"0"`**——
   `encoder.enc.0.core.out.0` 和 `encoder.enc.1.embed.0`
   （它们直接位于 `nn.Sequential` 内）
3. 上游的排除列表只有 `mm_projector`/`vision_tower`/`vision_resampler`，
   **没有 `audio_encoder`**，所以 `"0"` 被收进 LoRA 目标集
4. peft 按**后缀**匹配目标模块，`"0"` 于是命中了 LLM 的 `layers.0`，
   即整个 `Qwen2DecoderLayer`
5. peft 拒绝适配非 Linear 模块 → 崩溃

**这说明上游的 LoRA 路径从未被运行过**——与显存无关，只要模型里有
`audio_encoder` 就必然失败。

本 fork 的修复：把 `audio_encoder` 加入排除列表（同时贯彻
`--freeze_audio_encoder` 的意图），并跳过纯数字叶子名。修复后实测：

| 项 | 值 |
|---|---|
| 目标模块 | `q/k/v/o_proj`、`gate/up/down_proj`（正好 7 个标准层） |
| 可训练参数 | 161.5M（占 7.6B 的 **2.12%**） |
| `mm_projector`（另行可训练） | 27.5M |
| **单卡峰值显存** | **23.3 GB** |
| 24 步耗时 | 10.4 s |

**23.3 GB 意味着一张 24 GB 的卡（3090/4090/A10）就能训。**
对比：全参需要约 98 GB（单卡必 OOM），8 卡 ZeRO-3 每卡 18 GB。

用法：

```bash
bash script/train/smoke_test_lora.sh /tmp/lora_out 1    # 最后的参数是 GPU 编号
```

⚠️ 注意 LoRA 用的学习率是 **2e-4**，比全参的 1e-6 高两个数量级——
这是 LoRA 的常规设定，别照抄全参脚本的学习率。

## 6. 地雷区

### 6.1 ⚠️ 上游代码里有 9 处活跃的 `pdb.set_trace()`

**这是最容易让人莫名其妙卡住的坑。** 实测位置：

| 文件 | 处数 | 所在函数 |
|---|---|---|
| `data_utils_video_audio_neg_patch.py` | 4 | `preprocess_mixtral_two` ×3、`preprocess_nemo` ×1 |
| `data_utils_video_audio_neg_patch_fo.py` | 4 | 同上 |
| `vita_tts/decoder/ticodec/vqvae_tester.py` | 1 | — |

**为什么现在没事**：VITA-1.5 走 `preprocess_qwen2p5_instruct` 分支，
实测该函数区间内没有活跃 pdb。

**什么时候会炸**：一旦你改 `--version`、换对话模板、或者忘记设置
`conversation_lib.default_conversation`，就会掉进 mixtral/nemo 分支，
**训练直接卡在调试器里**。多卡场景下的表现是「没有任何输出的挂起」，
极难诊断。

自查命令：

```bash
grep -rn "^\s*import pdb; pdb.set_trace()" --include=*.py vita/
```

### 6.2 分词长度不匹配会静默作废整条样本

`data_utils_video_audio_neg_patch.py:642`：

```python
if cur_len != total_len:
    target[:] = IGNORE_INDEX      # 整条 label 清空
    print(f"WARNING: tokenization mismatch: ... (ignored)")
```

样本仍进 batch、仍前向反向，**但对 loss 零贡献**。只有一行 print，没有计数器。

**真实数据上这是头号风险**：如果模板有细微偏差，可能大批样本静默失效，
而 loss 曲线看起来完全正常。冒烟测试实测为 0，接真实数据后务必统计。

### 6.3 `get_prompt()` 不幂等

同一个 conversation 对象调两次，system prompt 会塌成一个字母。
详见 [PRIMER.md §6.2](../00-background/PRIMER.md#62-get_prompt-不幂等未记录的缺陷)。
**写 RL rollout 循环时每次都要重新 `.copy()`。**

### 6.4 零长度切片不能删

`vita_arch.py` 里的 `cur_audio_features[0:0]` 看起来是废代码，
实际是把编码器挂在 autograd 图上防止 ZeRO 报未使用参数。
删了会 `did not receive grad`。

### 6.5 `adpter` 拼写错误不能改

音频 adapter 全库拼作 `adpter`，权重的 state_dict key 也是这个拼写。
改了会加载失败。

### 6.6 换数据管线要改源码

`train.py:17-21` 用注释切换七个 `data_utils_*` 变体，没有命令行参数。
**改任何数据相关代码前，先确认 `train.py` import 的是哪一个。**

## 7. 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'vita'` | 没设 `PYTHONPATH`，且当前目录不是仓库根 | `export PYTHONPATH=$VITA_REPO`（在仓库根目录下运行时恰好不会报，换目录就炸） |
| 训练无输出地挂起 | 掉进了 `pdb.set_trace()` | 见 §6.1，检查 `--version` 和 conv 模板 |
| `TypeError: 'NoneType' object is not subscriptable` | 老版本的 `audios=None` 缺陷 | 已修；确认代码是最新的 |
| `did not receive grad` | 删了零长度切片 | 见 §6.4 |
| loss 正常但模型学不到东西 | 样本 label 被静默作废 | 见 §6.2，跑 `inspect_dataset.py` |
| 单卡训练 OOM | 7B 全参需要约 98 GB | 用 LoRA（峰值 23.3 GB），见 §5 |
| 加载 checkpoint 时联网卡住 | config 里是 HF repo ID | 跑 `tools/localize_config.py` |
| `NameError: name '_C' is not defined` | 脚本命名为 `inspect.py` 遮蔽了标准库 | 换个文件名 |
| 磁盘满 | 每个 checkpoint 16 GB | `rm -rf /tmp/smoke_out*` |
| `pip check` 报 decord 平台不支持 | 元数据问题，非真实冲突 | 忽略，decord 能正常用 |
| `Address already in use ... 29500` | 同机跑第二个 deepspeed 任务 | 加 `--master_port 29555`，见下 |
| `Could not deserialize ATN with version 3` | omegaconf 需要 antlr4 4.9.x | `pip install antlr4-python3-runtime==4.9.3`，见 BENCHMARKS §6.5 |
| 评测时 SSL 证书报错 | 上游数据集服务器证书过期 + `LMUData` 未设 | 见 §2.7 与 BENCHMARKS §6.5.1 |

**同机并行多个训练任务**：deepspeed 默认监听 29500，第二个任务会
以 `DistNetworkError` 失败，而报错栈有 18 层深、最后才提到端口。
每个任务给一个不同的 `--master_port`：

```bash
deepspeed --include localhost:2 --master_port 29500 ... &
deepspeed --include localhost:3 --master_port 29555 ... &
```

`--include localhost:N` 只管分配哪张卡，不管通信端口——这两件事
容易混为一谈。

## 8. DPO（离线偏好优化）

这是本 fork 相对上游新增的两种训练能力之一（另一种是 [§9 的 GRPO](#9-grpo组相对策略优化)）——上游只有 SFT。

### 8.1 组成

| 文件 | 作用 |
|---|---|
| `vita/train/dpo_loss.py` | DPO 损失 + fp32 log-prob |
| `vita/train/dpo_data.py` | 偏好对数据集 + 2B 批 collator |
| `vita/train/dpo_trainer.py` | `VITADPOTrainer`，覆写 `compute_loss` |
| `vita/train/train_dpo.py` | 入口 |
| `tools/make_dpo_smoke_data.py` | 合成偏好数据 |
| `tools/test_dpo_loss.py` | 19 项 CPU 测试 |
| `script/train/dpo_smoke_test.sh` | 单卡运行 |

数据格式在 SFT 格式上只加一个 `rejected` 字段：

```json
{
  "set": "dpo_smoke",
  "conversations": [
    {"from": "human", "value": "<image>\nDescribe this image."},
    {"from": "gpt",   "value": "首选回答"}
  ],
  "rejected": "次选回答",
  "image": "vita_newlog.jpg"
}
```

### 8.2 首步 loss 必须等于 0.6931

初始时 LoRA 的 B 矩阵为 0，策略等于参考，DPO logit 为 0，
loss 恰为 `-log(0.5) = 0.6931`。

**这是最强的正确性检查**：若首步偏离这个值，说明参考模型没接对。
实测结果：

```
{'loss': 0.6931, ...}
{'rewards/chosen': 0.0, 'rewards/rejected': 0.0, 'rewards/margin': 0.0}
```

24 步实测趋势（batch=1，噪声大，看均值）：

| 指标 | 前半 | 后半 |
|---|---|---|
| `rewards/margin` | +0.00200 | **+0.01054** |
| `rewards/accuracy` | 0.58 | **0.67** |
| `loss` | 0.6922 | **0.6879** |

### 8.3 三个必须知道的设计点

**参考模型 = 关掉 adapter 的同一份权重**，靠 `disable_adapter()`。
额外显存为 0，不需要第二份 7B。所以 `--lora_enable True` 是**强制**的。

**`mm_projector` 必须冻结。** 这点极易踩：`train.py:388` 先应用 LoRA，
`:395` 的 `initialize_vision_modules` 又把 `mm_projector` 的 `requires_grad`
设回 `True`（`vita_arch.py:59-61`，注释写着 "In case it is frozen by LoRA"）。
而 `disable_adapter()` **管不到它** —— 它一更新，参考模型就不再是基座，
`-log(0.5)` 只在第 1 步成立，之后静默漂移而 loss 看起来毫无异常。

`train_dpo.py` 会显式冻结所有非 adapter 参数并打印：

```
[DPO] froze 27.5M non-adapter parameters so the reference policy stays fixed
```

产出的 `non_lora_trainables.bin` 应为 **0 参数**，可用来验证。

**log-prob 必须在 fp32 下算。** `vita_qwen2.py:96` 的
`logits = logits.float()` 被注释掉了，forward 返回 bf16。实测在
152k 词表、200 token 上：

| 计算方式 | 误差 |
|---|---|
| bf16 全程 | **6.29 nats** |
| bf16 输入 + fp32 内部 | **0.0061 nats** |

差 1000 倍。DPO 的 β 典型取 0.1，bf16 那点噪声会完全淹没偏好信号。
`batch_sequence_logps` 内部已做上转，不要绕过它。

### 8.4 冒烟测试为什么把 lora_dropout 设为 0

dropout 在 `model.train()` 下对 policy 生效，而参考模型关了 adapter
因此没有 dropout —— 两侧不对称，首步就不等于 0.6931，那条检查会失效。
真实训练想要正则化时再打开。

### 8.5 用真实偏好数据训练（RLAIF-V）

合成数据只验证链路。真正能训出东西的是 **RLAIF-V**——每条是
一张图 + 一个问题 + 强模型判过优劣的两个回答，正好是 DPO 的
一对偏好样本。

```bash
# 1. 取一个分片（约 1.1 GB，6814 对）。HF 单流慢，分 8 段并行快 4 倍
URL=https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset/resolve/main/RLAIF-V-Dataset_000.parquet
SIZE=1109434826; N=8; CHUNK=$(( (SIZE + N - 1) / N ))
for i in $(seq 0 $((N-1))); do
  s=$((i*CHUNK)); e=$((s+CHUNK-1)); [ $e -ge $SIZE ] && e=$((SIZE-1))
  curl -sL -r ${s}-${e} -o part_$i "$URL" &
done; wait; cat part_* > shard000.parquet && rm part_*

# 2. 转成 DPO 记录格式
python tools/make_rlaif_v_data.py \
    --parquet shard000.parquet \
    --out-dir $VITA_WEIGHTS/rlaif_v_dpo --limit 3000

# 3. 训练
export VITA_RLAIF_DATA_DIR=$VITA_WEIGHTS/rlaif_v_dpo
bash script/train/dpo_rlaif_v.sh /path/to/output 0
```

**转换脚本处理的三件事**，每一件不做都会静默出错：

1. **图像按内容寻址**。parquet 里图是内嵌字节且跨行重复，
   逐行写文件会存一堆副本。用 SHA-1 做文件名后
   3000 对只占 2408 张图。
2. **补 `<image>` token**。`LazySupervisedDataset` 靠这个字面量
   决定往哪儿插视觉特征，缺了就**静默退化成纯文本训练**——
   loss 照常下降，你不会发现。
3. **丢掉两侧相同的样本**。这种 pair 的 DPO logit 恒为 0，
   loss 永远是 `-log(0.5)`，白跑前向。实测 3082 条里有 2 条。

**首步 loss 仍然必须是 0.6931**。实测在真实数据上精确命中，
和合成数据一致——这说明参考模型在真实多模态输入下也接对了。

**数据本身值得看一眼**：chosen 平均 299 字符，rejected 298 字符。
**几乎相等是好事**——若 chosen 系统性更长，模型会学到「长 = 好」
这个捷径，而不是学质量判别。换数据集时先量这个数。

### 8.5.1 训练前先量一下数据可分性

**这是本项目踩过最值得记的一课。** 在 RLAIF-V 上跑 DPO，
`rewards/margin` 一直在 0 附近震荡、`rewards/accuracy` 稳在 0.5，
看起来像训练坏了。把学习率从 5e-6 提到 2e-5（4 倍），
漂移量只从 0.0227 变成 0.0245，准确率纹丝不动。

真正的原因用 `tools/probe_preference_separability.py` 一测就清楚：

```
n=400:  base accuracy 55.2%  (221/400)
        95% CI [50.4%, 60.1%]   vs 随机: z=2.10, p=0.036
        logp gap: mean=+3.89 sd=35.87   信噪比 0.11
```

**基座对这些样本的判别力虽然显著高于随机（p=0.036），但极其微弱**
——置信区间下界只有 50.4%，信噪比 0.11。

> 先用 n=100 测得 53%，据此判断「等同随机」是错的——样本量不足。
> n=400 才看清是「信号真实但太弱」。**探针至少跑 300–400 对**，
> 100 对的置信区间宽到无法区分「随机」和「弱信号」。

DPO 的梯度正比于「模型当前错得有多系统」。信噪比 0.11 意味着
每个 batch 的梯度方向几乎被噪声主导，187 步的预算下累积不出可见偏移。
**margin 平坦主要是数据可分性的问题，不是 trainer 的 bug**。

为什么这批数据这么难？转换脚本报的两个数字已经预告了：

| 指标 | 值 | 含义 |
|---|---|---|
| chosen 更长的比例 | 46.5% | 无长度偏置（50% 为完全无偏） |
| 两回答文本相似度 | 均值 0.29 | 是**两个不同的答案**，不是改一个词 |

没有长度捷径、没有格式捷径，两个回答都流畅切题、只在某个细节
是否幻觉上不同。**这正是 RLAIF-V 作为高质量偏好数据的设计目标**，
但也意味着它对 7B 基座是一个很陡的任务。

**所以：跑任何 DPO 之前先跑这个探针**，几分钟的事：

```bash
VITA_RLAIF_DATA_DIR=<dir> VITA_CKPT=<ckpt> \
python tools/probe_preference_separability.py --n 400
```

- **接近 50%、CI 跨过 50%** → 数据对这个基座不可分。换数据。
- **显著但信噪比 < 0.2** → 信号真实但很弱（本次即此）。
  几百步内别指望看到变化：要么大幅加数据量，要么改用
  贴近基座决策边界的样本对。
- **65%+** → 信号充足，此时 margin 还平坦才该去查
  trainer、学习率、beta。

### 8.5.2 已知限制

- **只支持 LoRA**。全参 DPO 需要第二份 7B + 8 卡，未实现。
- **状态 token 未特殊处理**。chosen/rejected 目前带相同的
  `☜`/`☞`/`☟` 前缀（由数据构造保证），若两侧状态不同会让模型
  学到区分状态而非区分质量。
- **RLAIF-V 的图来自 COCO/OK-VQA 等**，与 MME/MMStar 的图像分布
  不完全一致，跨分布的提升本来就比同分布难。

### 8.6 成对样本的图像只编码一次

chosen 和 rejected 看的是同一张图，但两条序列各带一份，
朴素实现会让 InternViT 把同一张图编码两次。

现在 collator 会算出 `image_group_size`（一半的图块数）并传给
`prepare_inputs_labels_for_multimodal`，后者调用
`encode_images_deduped` 只编码一份、再复制特征。
因为视觉编码器是确定性的，**结果逐位相同**——
`tools/test_image_dedup.py` 用 `torch.equal` 断言这点，
而非 `allclose`。

实测收益（H100，一对样本）：

| 每序列图块 | 优化前 | 优化后 | 节省 |
|---|---|---|---|
| 5 块 | 22.4 ms | 12.6 ms | 44% |
| 13 块 | 53.9 ms | 29.2 ms | 46% |

**但在整步里占比很小**：24 步端到端只从 20.8 s 降到 20.5 s，
因为瓶颈是 7B 的 LLM 前向（每步 4 次）而非视觉编码器。

真正的价值在 GRPO：一组 N 个 rollout 共享同一张图时，
节省是 `(N-1)/N`——N=8 时省 87.5%。
`encode_images_deduped` 已支持任意倍数，测试覆盖了 x2/x3/x4。

SFT 路径不受影响：`image_group_size` 默认为 `None`，
此时走原来的 `encode_images`。

## 9. GRPO（组相对策略优化）

比 DPO 更进一步：**回答由模型自己生成，奖励在训练中实时计算**。

### 9.1 与 DPO 的区别

| | DPO | GRPO |
|---|---|---|
| 数据 | 预标好的 chosen/rejected 对 | **只要 prompt** |
| 打分 | 数据里给定 | **训练中实时算** |
| 采样 | 无 | **每 prompt 生成 G 个 rollout** |
| 优势 | 隐式（两两相减） | **组内归一化** `(r-mean)/std` |
| critic | 无 | 无（这是 GRPO 相对 PPO 的优势） |

### 9.2 用法

```bash
python tools/make_grpo_smoke_data.py --out-dir $VITA_WEIGHTS/grpo_smoke_data
export VITA_GRPO_DATA_DIR=$VITA_WEIGHTS/grpo_smoke_data
bash script/train/grpo_smoke_test.sh /tmp/grpo_out 1
```

数据格式只要 prompt + 奖励依据：

```json
{
  "set": "grpo_smoke",
  "conversations": [{"from": "human", "value": "Describe a cat."}],
  "reward_meta": {"keywords": ["cat"], "target_len": [20, 120], "state": "left"}
}
```

### 9.3 奖励函数是可插拔的

`vita/train/rewards.py` 是一个注册表。内置四个规则奖励：

| 名称 | 作用 |
|---|---|
| `keyword` | 覆盖了多少个期望关键词（按比例给分，非二值） |
| `length` | 长度是否落在目标区间，两侧线性衰减 |
| `no_repeat` | 3-gram 去重率，惩罚复读 |
| `state_token` | 首字符是否是正确的 `☜`/`☞`/`☟` |

用 `--reward_fns keyword:1.0,length:0.5` 选择并加权。
加 `--judge_model_path` 可以再挂一个小模型打分
（如 `Qwen/Qwen2.5-0.5B-Instruct`，3.5 GB，Apache-2.0）——
它读模型给 "1".."5" 各 token 的概率算加权期望，
比解析文本稳健，且输出连续。

**写新规则时注意**：奖励必须能**区分同一组内的 rollout**。
GRPO 的优势是组内归一化，所有 rollout 得分相同的组
（`degenerate`）优势为 0、不产生梯度。二值规则若 8 个 rollout 全过，
等于没加。**优先写分级的而非通过/不通过的**。

### 9.4 实测结果

12 步、G=8、单卡，所有指标朝正确方向：

| 指标 | 前半 | 后半 | |
|---|---|---|---|
| **`reward/mean`** | 0.7891 | **0.8622** | ↑ 核心信号 |
| `reward/length` | 0.1816 | **0.4009** | ↑ |
| `reward/keyword` | 0.8854 | 0.9583 | ↑ |
| `completion/len` | 75.1 | **67.3** | ↓ |

最有说服力的是 **length 奖励翻倍的同时回答长度真的变短了** ——
两个数字互相印证，说明模型在响应奖励信号而非随机漂移。

健康指标：首步 `grpo/kl` **= 0**（参考模型正确，作用同 DPO 的 0.6931）、
`ratio` = 1、`advantage_std` 恒为 1.0、`degenerate_frac` **= 0%**。

### 9.5 三个关键实现点

**rollout 要绕过 `generate()`。** `vita_qwen2.py:209` 显式拒绝外部传入
`inputs_embeds`，但直接调 `Qwen2ForCausalLM.generate(model, inputs_embeds=...)`
可行。这样 prompt 只需编码一次，多模态版本还能省下重复的视觉编码。

**log-prob 复用缓存的 prompt embeds 重算。** 把采样 token 的 embedding
拼到缓存的 prompt embeds 后面再前向。实测与生成时的 scores 一致
（最大差 0.0042，bf16 正常范围）。这解决了
[ARCHITECTURE §13](../00-background/ARCHITECTURE.md#13-where-rl-would-attach)
列为主要障碍的「prompt token 序列不复原」。

**采样时必须 `model.eval()`。** 否则 LoRA dropout 会让采样策略与
计算 log-prob 的策略不一致。冒烟脚本另外把 `lora_dropout` 设为 0。

### 9.6 组内标准差为 0 的陷阱

优势是 `(r - mean) / std`，组内全同时 std=0，`0/0` 得 NaN 并静默污染梯度。
**规则奖励下这极其常见**（所有 rollout 都满足或都不满足某规则）。

`group_advantages` 在 `std < 1e-6` 时把该组优势置 0 并计数，
通过 `groups/degenerate_frac` 上报。**这个数接近 1.0 说明奖励没有区分度，
几乎什么都没在训练**——它看起来像「模型学不动」，实际是数据/奖励的问题。

### 9.7 多模态与真实训练（后来补齐的部分）

上面 9.1–9.6 写于纯文本冒烟阶段。之后 GRPO 补齐了：

- **多模态 prompt**：trainer 的 `_fuse` 每 batch 把视觉特征拼进 prompt
  embedding 一次（在 G 倍扩展之前），批量 `generate` 直接吃融合后的
  embedding。冒烟：`bash script/train/grpo_mm_smoke_test.sh`。
- **μ 步样本复用**：`--grpo_num_iterations μ`（脚本里 `NUM_ITER`）。
  μ>1 时同一批 rollout 连续用于 μ 个优化步，复用步只重算策略 log-prob，
  ratio 离开 1、clip 生效；观察 `grpo/clip_frac`。
- **可验证奖励**：`answer`（二值精确匹配）+ `format`（分级结构检查），
  见 `vita/train/rewards.py`。
- **真实训练全链路**（CLEVR 计数）：

```bash
python tools/make_clevr_grpo_data.py --parquet '<shards>' --out-dir <dir>   # 数据
export VITA_CLEVR_GRPO_DATA_DIR=<dir> WEIGHTS_ROOT=<weights>
bash script/train/grpo_clevr.sh <output_dir>                               # 训练（8 卡）
python tools/eval_grpo_heldout.py --before <base> --after <merged> ...     # held-out 评测
```

  结果：400 步 held-out 准确率 44.6% → 77.4%。后续对照实验补齐了边界：
  通用基准零退化、SuperCLEVR OOD +17pt、配平 SFT 对照分布内追平且 OOD
  反超（工具：`make_clevr_sft_data.py` / `make_superclevr_eval_data.py` /
  `make_clevr_stage2_data.py`，训练脚本 `sft_clevr.sh`）。全程记录、超参
  与指标手册见 [GRPO_DEEP_DIVE.md](../03-experiments/GRPO_DEEP_DIVE.md)。

仍然的限制：

- **只支持 LoRA**，同 DPO。
- **RL 范围只覆盖文本+图片/视频**；音频编码器全程冻结，不做音频 RL。

## 10. 本机资源现状

| 项 | 值 |
|---|---|
| GPU | 8 × H100 80GB（注意：文档里写的 H800 是原作者的机器） |
| 磁盘可用 | 3.4 TB |
| conda 环境 | `/root/anaconda3/envs/vita`（Python 3.10.18） |
| 权重 | `/usr/local/kai/lx/vita_weights/`（19 GB） |
| 冒烟数据 | `$VITA_WEIGHTS/smoke_data/`（24 条合成样本） |
| 网络 | 走 HTTP 代理；**SSH 22/443 均被封**，git 只能用 HTTPS + token |
| HF 下载 | 实测 7.2 MB/s 单流；`max_workers=8` 时约 140 GB/h |

**GPU 0 上常有外部进程**（约 900 MB），推理时建议用 `CUDA_VISIBLE_DEVICES=1`
避开；8 卡训练不受影响。

### 下一步做什么

- **想训真实数据** → [DATASETS.md](../02-data/DATASETS.md)；本 fork RL 实际用的数据见其 §3.3
- **想看 RL 的完整结果** → DPO 见 [EXPERIMENT_LOG.md](../03-experiments/EXPERIMENT_LOG.md)，
  GRPO 见 [GRPO_DEEP_DIVE.md](../03-experiments/GRPO_DEEP_DIVE.md)；设计走读见
  [ARCHITECTURE.md §14](../00-background/ARCHITECTURE.md#14-the-rl-stack-dpo-and-grpo)
- **想弄懂原理** → [PRIMER.md](../00-background/PRIMER.md)
- **换机器重建** → [MIGRATION.md](./MIGRATION.md)
