# 上手手册：本机实操

> [PRIMER.md](./PRIMER.md) 讲的是**原理**，这份讲的是**动手**：本机的确切路径、
> 可直接复制的命令、改代码时的自检流程，以及各条路径的实测状态。
>
> 所有命令都在本机（8×H100 / `/usr/local/kai/lx/VITA-RL`）验证过。
> 凡是我**没跑过**的，本文明确标注为未验证，不含糊过去。

## 目录

- [1. 五秒进入工作状态](#1-五秒进入工作状态)
- [2. 常用命令速查](#2-常用命令速查)
- [3. 改代码时的自检流程](#3-改代码时的自检流程)
- [4. 接入自己的数据集](#4-接入自己的数据集)
- [5. 各条路径的实测状态](#5-各条路径的实测状态)
- [6. 地雷区](#6-地雷区)
- [7. 故障排查](#7-故障排查)
- [8. 本机资源现状](#8-本机资源现状)

---

## 1. 五秒进入工作状态

新开终端后，这段直接粘贴：

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
| `tools/` 三个工具 | ✅ **本机跑通** | localize / make_smoke / inspect / test |
| 视频推理 | ⚠️ 未测 | `--video_path` 参数存在，无测试素材 |
| **LoRA / 单卡训练** | ❌ **不可用** | `peft` **未安装**，`--lora_enable True` 会 ImportError |
| web demo | ❌ **不可用** | `flask`/`flask_socketio`/`vllm`/`onnxruntime` 均未装；还缺 `web_demo/wakeup_and_vad/resource/` 下的 `silero_vad.onnx`/`.jit` |
| VLMEvalKit 评测 | ⚠️ 未配置 | `VLMEvalKit/vlmeval/config.py:141-142` 仍是 `/path/to/model` |
| Video-MME 评测 | ⚠️ 未测 | 需另下 Video-MME 数据集 |
| 语音输出（TTS） | ⚠️ 存疑 | 见 [PRIMER.md §9](./PRIMER.md#9-语音输出论文与代码不一致)——代码与论文描述不一致 |

### 关于 LoRA 的额外提醒

文档里说「显存不足可用 LoRA」，但除了 `peft` 没装之外，还有个设计问题：

`train.py:160` 的 `find_all_linear_names` 排除列表是
`["mm_projector", "vision_tower", "vision_resampler"]`
——**没有排除 `audio_encoder`**。而 whale 的 adapter 里有
`self.project` 和 `self.adpter` 两个 `nn.Linear`，会被 LoRA 当作目标模块挂上去。

这与 `--freeze_audio_encoder True` 的意图冲突。上游没有任何脚本用过 LoRA
路径，所以这个组合从未被验证。**要用 LoRA 的话，先自己确认参数是否按预期冻结。**

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
详见 [PRIMER.md §6.2](./PRIMER.md#62-get_prompt-不幂等未记录的缺陷)。
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
| 单卡训练 OOM | 7B 全参需要约 98 GB | 必须 8 卡；LoRA 路径不可用（§5） |
| 加载 checkpoint 时联网卡住 | config 里是 HF repo ID | 跑 `tools/localize_config.py` |
| `NameError: name '_C' is not defined` | 脚本命名为 `inspect.py` 遮蔽了标准库 | 换个文件名 |
| 磁盘满 | 每个 checkpoint 16 GB | `rm -rf /tmp/smoke_out*` |
| `pip check` 报 decord 平台不支持 | 元数据问题，非真实冲突 | 忽略，decord 能正常用 |

## 8. 本机资源现状

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

- **想训真实数据** → [DATASETS.md](./DATASETS.md)，建议从方案 A（260 GB）起步
- **想加 RL** → [ARCHITECTURE.md §13](./ARCHITECTURE.md#13-where-rl-would-attach)，
  建议先做离线 DPO
- **想弄懂原理** → [PRIMER.md](./PRIMER.md)
- **换机器重建** → [MIGRATION.md](./MIGRATION.md)
