# VITA-1.5 架构与代码走读

一份理解这个代码库的指南：模型是什么、代码如何组织、推理和训练的每一步究竟发生了什么。

除标注为本 fork 特有的章节外，本文描述的都是**上游 VITA-1.5**。行号引用对应撰写时的仓库状态，可能随代码变动而漂移。

> 语言：[English](./ARCHITECTURE.md) | **中文**
>
> 配套文档：[REPRODUCE_zh-CN.md](./REPRODUCE_zh-CN.md) 是操作日志（怎么装、怎么跑、什么坏了）。本文是概念地图。

## 目录

- [1. 模型是什么](#1-模型是什么)
- [2. 仓库布局](#2-仓库布局)
- [3. 核心思想：负数索引占位 token](#3-核心思想负数索引占位-token)
- [4. 三个编码器](#4-三个编码器)
- [5. `prepare_inputs_labels_for_multimodal`：模型的心脏](#5-prepare_inputs_labels_for_multimodal模型的心脏)
- [6. 状态符号（拒答机制）](#6-状态符号拒答机制)
- [7. 推理全流程](#7-推理全流程)
- [8. 训练全流程](#8-训练全流程)
- [9. 数据管线](#9-数据管线)
- [10. 实时全双工 demo](#10-实时全双工-demo)
- [11. 模型变体](#11-模型变体)
- [12. 已知缺陷与粗糙之处](#12-已知缺陷与粗糙之处)
- [13. RL 该接在哪里](#13-rl-该接在哪里)

---

## 1. 模型是什么

VITA-1.5 是一个**全模态 LLM**：单个语言模型接收图像、视频、音频和文本，输出文本——并通过一个独立的解码器输出语音。

设计上属于 LLaVA 血统，向音频做了扩展：

```
图像/视频 ──> InternViT-300M ──> MLP projector ──┐
                                                  ├──> Qwen2.5-7B ──> 文本
音频 ────────> whale 编码器 ──> CNN adapter ─────┘         │
                                                           └──> TTS 解码器 ──> 语音
```

最关键的思想是：**每种模态都被转换成活在 LLM embedding 空间里的向量**，然后拼接进 token embedding 序列。LLM 本身是未经修改的 Qwen2.5——它从未学习新的输入类型，只是接收了一些恰好来自图像或波形（而非 token 查表）的 embedding。

VITA-1.5 与普通 VLM 的区别：

| | |
|---|---|
| **端到端语音输出** | 无外挂 TTS。语音解码器直接消费 LLM 的 hidden states，这正是延迟收益的来源（约 4 秒 → 约 1.5 秒）。 |
| **负样本训练** | 模型被显式训练去**拒绝**回答噪声或非语音音频，这才让常开麦克风的交互变得可行。 |
| **渐进式训练** | 视觉和音频分阶段对齐，使得加入语音几乎不损害视觉能力（均分 71.3 → 70.8）。 |

## 2. 仓库布局

```
vita/
├── constants.py                 # 占位 token 索引 —— 先读这个
├── conversation.py              # prompt 模板（9 种 conv 模式）
├── model/
│   ├── vita_arch.py             # ★ VITAMetaModel / VITAMetaForCausalLM：模态融合
│   ├── builder.py               # 推理用的 load_pretrained_model()
│   ├── language_model/
│   │   ├── vita_qwen2.py        # ★ 主模型（VITA-1.5）
│   │   ├── vita_fo_qwen2.py     # 全双工变体，含状态预测头
│   │   ├── vita_mixtral.py      # VITA-1.0 遗留
│   │   └── vita_nemo.py         # Mistral/Nemo 变体
│   ├── multimodal_encoder/
│   │   ├── builder.py           # 按名称子串分发视觉塔
│   │   ├── internvit/           # VITA-1.5 实际使用的视觉塔
│   │   ├── clip/ eva_clip/ siglip/   # 备选
│   │   └── whale/               # ★ 音频编码器（自研）
│   ├── multimodal_projector/
│   │   └── builder.py           # mlp2x_gelu 等
│   └── vita_tts/                # 端到端语音合成
├── train/
│   ├── train.py                 # ★ 唯一的训练入口
│   └── vita_trainer.py          # HF Trainer 子类
├── util/
│   ├── data_utils_video_audio_neg_patch.py   # ★ 当前启用的数据管线
│   ├── data_utils_*.py          # 另外 6 个变体，通过改 import 切换
│   └── mm_utils.py              # tokenizer 辅助函数
└── config/
    ├── dataset_config.py        # 数据集路径（发布时为空）
    └── __init__.py              # DataConfig 注册表

script/train/*.sh                # 18 个启动脚本（阶段 × 骨干 × 单机/多机）
web_demo/                        # Flask + SocketIO 实时 demo，vLLM 加速
videomme/                        # Video-MME 基准
VLMEvalKit/                      # OpenCompass 评测套件的内嵌副本
tools/                           # 本 fork 新增
```

标 ★ 的四个文件是真正逻辑所在。仅 `vita_arch.py` 一个文件就承载了这个模型大部分不直观的地方。

## 3. 核心思想：负数索引占位 token

出自 `vita/constants.py`：

```python
IGNORE_INDEX      = -100
IMAGE_TOKEN_INDEX = -200
AUDIO_TOKEN_INDEX = -500
```

这些**不是真正的 token ID**。tokenizer 只会产生非负 ID，因此负值是能安全存活在 `input_ids` 张量里、且绝不与任何东西冲突的标记。

流程如下：

1. 文本中含有字面字符串 `<image>` / `<audio>`。
2. `tokenizer_image_audio_token`（`vita/util/mm_utils.py:73`）在这些字符串处切分，并用 `-200` / `-500` 替代正常 token ID：

   ```python
   for chunk in re.split(r"(<audio>|<image>)", prompt):
       if chunk == "<audio>":   prompt_chunks.append([audio_token_index])
       elif chunk == "<image>": prompt_chunks.append([image_token_index])
       else:                    prompt_chunks.append(tokenizer(chunk).input_ids)
   ```

3. `prepare_inputs_labels_for_multimodal` 找到这些位置，**用真实的编码器输出向量替换掉每个标记**。

需要牢记的结论：**模型接收的是 `inputs_embeds`，不是 `input_ids`。** 等到 LLM 真正运行时，占位位置已经变成了稠密向量，原始的整数序列不再存在。这就是为什么 `generate()` 接收 `inputs_embeds`，也是任何需要事后重算 log-probability 的工作面临的最大复杂性来源——见[§13](#13-rl-该接在哪里)。

## 4. 三个编码器

### 4.1 视觉：InternViT-300M-448px

`vita/model/multimodal_encoder/internvit/internvit_encoder.py`

图像首先由 `dynamic_preprocess`（`data_utils_video_audio_neg_patch.py:1499`）切片——采用 InternVL 的方案：从候选集合中挑选最匹配原图的长宽比，缩放，然后切成 448×448 的瓦片（最多 `max_dynamic_patch` 个，默认 12），并可选地追加一张整图的缩略图。

每个瓦片随后经过 ViT，再经一次**pixel shuffle**，用空间分辨率换取通道数：

```
448×448 瓦片 ─ViT/14─> 32×32 = 1024 patches × 1024 维
             ─pixel_shuffle(0.5)─> 16×16 = 256 tokens × 4096 维
```

在本 checkpoint 上实测：单个瓦片 → `(1, 256, 4096)`。

也就是说**一个瓦片消耗 256 个 LLM token**，12 瓦片的图像约消耗 3072 个。这解释了为什么训练脚本里的 `model_max_length` 是 6200 而不是某个小值，也解释了实时 demo 为何要把 `max_dynamic_patch` 降到 1——256 token 对比 3328 token，就是"可交互"与"迟滞"的分界。

`scale_pix_shuffle = 0.5`，`select_layer = -1`（最后一层 hidden，并丢弃 CLS token：`image_features[:, 1:]`）。

### 4.2 音频：whale

`vita/model/multimodal_encoder/whale/`

一个自研编码器（约 341M 参数），**不是** Whisper。结构：

```
波形 ─kaldi.fbank─> 80 维 mel 帧 @100fps
     ─GlobalCMVN─> 归一化
     ─whaleEncoder─> 编码器状态      (Transformer + FSMN + DTC blocks)
     ─CNNAdapter─> LLM 维度的向量
```

下采样过程在 `audioEncoderProcessor.process`（`whale/init_model.py:35`）里一眼可见：

```python
attn_mask = torch.ones(mat.shape[0])
attn_mask = attn_mask[2::2][2::2][0::2]   # 三次 stride-2 => 除以 8
```

实测：3.54 秒的 wav → 352 个 fbank 帧 → **44 个 LLM token**，即约 **12.5 token/秒**音频。相比图像便宜得多。

注意其配置**不在本仓库内**——而是运行时从音频编码器目录加载（`train.yaml`、`global_cmvn`、`final.pt`），通过 `get_file_from_repo` 获取，因此既能解析本地路径也能解析 HuggingFace repo ID（`multimodal_encoder/builder.py:46-49`）。

adapter（拼写为 `adpter`，注意这是上游的拼写——`train.py:390` 和 `vita_trainer.py:321` 都靠这个字符串做匹配）是阶段 1 中唯一被训练的音频组件。

### 4.3 Projector

`vita/model/multimodal_projector/builder.py:154`。VITA-1.5 使用 `mlp2x_gelu` = `Linear(mm_hidden, hidden) → GELU → Linear(hidden, hidden)`。另有若干备选（`spp`、`ldp`、`minigpt`、`vanilla`、`identity`），但发布配置均未使用。

## 5. `prepare_inputs_labels_for_multimodal`：模型的心脏

`vita/model/vita_arch.py:308`。其余都是管道工程；这个函数才是模态真正融合的地方。全长约 290 行。逻辑如下：

**第 1 步 —— 缓存解码的提前返回**（第 312 行）。若 `input_ids.shape[1] == 1`，说明正在用 KV cache 逐 token 生成，没有东西需要拼接，于是只扩展 attention mask 后返回。

**第 2 步 —— 编码**（第 334 行）。整个 batch 的所有图像瓦片被拼成一个张量做单次 ViT 前向，再按样本切回。音频在一次批量调用中通过 whale 编码器。

**第 3 步 —— 在标记处切分每个序列**（第 418 行）：

```python
image_audio_token_indices = [-1] + torch.where(
    (cur_input_ids == IMAGE_TOKEN_INDEX) | (cur_input_ids == AUDIO_TOKEN_INDEX)
)[0].tolist() + [cur_input_ids.shape[0]]
```

序列被切成标记之间的文本片段。文本片段照常经 `embed_tokens` 嵌入；在每个标记处则插入对应的编码器输出。

**第 4 步 —— label 同步扩展**（第 452 行）。每一个插入的视觉或音频向量都获得 `IGNORE_INDEX` 标签：

```python
cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, ...))
```

这一点很关键：**模型从不被训练去预测图像或音频 token**，只预测其后的文本。`input_ids` 里的一个标记会在 `inputs_embeds` 里变成数百个位置，label 张量必须同步等量增长，否则 loss 会**静默错位**。

**第 5 步 —— padding 与堆叠**（第 522 行）。序列被 pad 到 batch 内最大长度，截断到 `tokenizer_model_max_length`，并按新长度重建 `position_ids` / `attention_mask`。

**返回**（第 602 行）：

```python
return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
#      ^^^^ input_ids 被刻意置为 None
```

两个值得知道的细节：

- **返回时 `position_ids` 通常是 `None`**（第 599 行）。除非设置了 `shared_v_pid_stride`，否则交由 LLM 自行推导位置。
- **`shared_v_pid_stride`**（第 628 行，`make_shared_position_ids`）是一个可选方案，让多个视觉 token 共享**同一个** position id，从而压缩长视觉片段的位置编码占用。默认关闭。

## 6. 状态符号（拒答机制）

这是 VITA-1.5 实现常开麦克风交互的机制，很容易被忽略，因为它的实现只是三个 Unicode 字符。

在 `preprocess_multimodal`（`data_utils_video_audio_neg_patch.py:128`）中，训练时每条 assistant 回复都会被加上一个字符的前缀：

```python
if i == inserted_id:                          # ☞ ☟ ☜
    sentence["value"] = "☟" + sentence["value"]
elif sentence["from"] == "gpt":
    if "<audio>" in source[i - 1]["value"]:
        sentence["value"] = "☞" + sentence["value"]
    else:
        sentence["value"] = "☜" + sentence["value"]
```

| 符号 | 含义 |
|---|---|
| `☜` | 对**文本**查询的回复 |
| `☞` | 对**有效语音**查询的回复 |
| `☟` | **负样本**条件下的回复——噪声音频、背景人声，以及任何并非向助手发出的输入 |

JSON 中的 `inserted_id` 标记哪一轮 assistant 回复是负样本。

收益体现在推理阶段。在实时服务端（`web_demo/server.py:312`）：

```python
def judge_negative(text):
    is_negative = text.startswith('☟')
    return is_negative
```

**生成的第一个字符**就告诉服务端该说出这段回复还是保持沉默——一个在**一个 token 之后**即可获得的拒答决策，无需独立的分类器。这正是常开麦克风得以实用的原因。

可直接观察到：`asset/q1.wav`（清晰语音）产生前缀 `☞` 的回复；`asset/q2.wav`（噪声）产生 `☟`。

紧挨这段代码上方是被注释掉的第一版尝试——同样的方案，但用 `<1>` / `<2>` / `<3>`。改用生僻 Unicode 字形的收益在 tokenizer 上是可度量的：`☜` / `☞` / `☟` **各占 1 个 token**（id 分别为 145789 / 144766 / 146164），而 `<1>` / `<2>` / `<3>` **各占 3 个 token**（`<`、数字、`>`）。单 token 才使得拒答/接受的决策恰好在一步解码后可用，而且不会被恰好包含 `<1>` 的普通文本意外触发。

## 7. 推理全流程

沿着 `video_audio_demo.py` 端到端追踪：

```
1. load_pretrained_model()                      builder.py:14
     ├── VITAQwen2ForCausalLM.from_pretrained()  加载 7B + projector
     ├── vision_tower.load_model()               InternViT，转为 fp16
     └── audio_encoder                           whale，来自 train.yaml/final.pt

2. 构建 prompt
     ├── conv_templates["qwen2p5_instruct"]      conversation.py:326
     ├── 按模态选择 system prompt                 图像 / 视频 / 纯文本
     └── "<image>\n<audio>\n" 放入 user 轮次

3. tokenize
     └── tokenizer_image_audio_token()           mm_utils.py:73
         "…<image>…" -> [… , -200, …, -500, …]

4. 预处理输入
     ├── dynamic_preprocess()                    图像 -> N 个 448² 瓦片
     └── audio_processor.process()               wav -> fbank 帧

5. model.generate(inputs, images, audios)        vita_qwen2.py:198
     └── prepare_inputs_labels_for_multimodal()  标记 -> embedding
         └── super().generate(inputs_embeds=…)   标准 HF 生成

6. 解码
     └── tokenizer.batch_decode()                首字符是 ☜/☞/☟
```

单张 H800 实测：文本查询 7.4 秒，音频查询 2.3 秒（音频 prompt 的 token 数远少于一段长文本指令）。

## 8. 训练全流程

### 8.1 渐进式三阶段配方

`script/train/` 下的 18 个脚本是 {阶段} × {骨干} × {单机, 多机} 的笛卡尔积。输出目录名直接透露了阶段划分：

| 阶段 | 脚本前缀 | 输出目录 | 学习率 | 解冻的部分 |
|---|---|---|---|---|
| 1a | `pretrain_mlp_*` | `llava-s1-pretrain_mlp_video` | 5e-4 | 仅 `mm_projector` |
| 1b | `pretrain_audio_mlp_*` | `llava-s1-pretrain_audio_mlp` | 5e-4 | 仅 `audio_encoder.adpter` |
| 2 | `finetune_*` | `llava-s2-pretrain_video` | 2e-5 | LLM + projector（qwen 版还含视觉塔） |
| 3 | `finetuneTask_*` | `llava-s3-finetune_task` | 2e-5 | LLM，音频完全冻结 |
| 3-neg | `finetuneTaskNeg_*` | `llava-s3-finetune_task_neg` | 1e-5 | LLM + **音频 adapter** |
| 3-fo | `finetuneTaskNeg_qwen_fo_*` | 同上 | 1e-4 | 仅音频 prompt embedding |

README 指定的续训入口是 `finetuneTaskNeg_qwen_nodes.sh`（4 节点 × 8 卡）。

所有阶段共通：DeepSpeed **ZeRO-3**、bf16、tf32、gradient checkpointing、1 个 epoch、cosine 调度、`warmup_ratio=0.03`、`save_total_limit=1`。

### 8.2 冻结开关

`train.py:377-412` 是阶段划分背后的机制。共 8 个开关：

| 开关 | 解冻 |
|---|---|
| `--tune_mm_mlp_adapter` | `mm_projector` |
| `--tune_audio_mlp_adapter` | `audio_encoder.adpter` |
| `--audio_prompt_finetune` | `audio_encoder.prompt_embeddings` |
| `--audio_state_predictor_tuning` | `predictor_head` |
| `--freeze_audio_encoder` | （默认 True —— whale 主干在此从不训练） |
| `--freeze_audio_encoder_adapter` | 按阶段切换 |
| `--unfreeze_vision_tower` | ViT |
| `--freeze_mm_mlp_adapter` | 第一项的反向 |

**前四个开关每个都以一次全局 `model.requires_grad_(False)` 开始**，再重新启用某个子模块。因此它们基本上互斥——同时启用两个意味着后者会抹掉前者的解冻。`audio_prompt_finetune` 与 `audio_state_predictor_tuning` 是唯一被设计成可共存的一对。

### 8.3 `VITATrainer` 定制了什么

`vita/train/vita_trainer.py`，四处覆写：

1. **`_get_train_sampler`** —— 可选的 `LengthGroupedSampler`，把长度相近的样本组成 batch，并用正负号区分模态（正长度 = 多模态，负 = 纯文本）。所有发布脚本中都是关闭的。
2. **`create_optimizer`** —— 支持独立的 `--mm_projector_lr`。注意第 190 行：`mm_projector` 那个条件被注释掉了，所以尽管参数名如此，该开关现在只影响 `vision_tower`。
3. **`_save_checkpoint` / `_save`** —— 在只训练 adapter 时，仅保存 `mm_projector.bin` 或 `audio_adpter.bin`，而不是 16 GB 的完整模型。这正是阶段 1 的 checkpoint 得以廉价的原因。
4. **`training_step`** —— 目前是纯透传；debug 分支被注释掉了。

### 8.4 显存

7B 全参数训练在单张 80 GB GPU 上放不下。AdamW 为每个参数保存两个 fp32 动量（约 56 GB），外加 fp32 主权重（约 28 GB）；ZeRO-3 将这些切分到各 rank。单卡运行会完成前向和反向，然后在分配优化器状态时 OOM。**8 卡是全参数训练的实际下限。** 参见 [REPRODUCE_zh-CN.md](./REPRODUCE_zh-CN.md#显存说明)。

## 9. 数据管线

### 9.1 样本格式

```json
{
  "set": "sharegpt4",
  "id": "000000000164",
  "conversations": [
    {"from": "human", "value": "<image>\n<audio>\n"},
    {"from": "gpt",   "value": "This is a well-organized kitchen…"}
  ],
  "image": "coco/images/train2017/000000000164.jpg",
  "audio": ["output_wavs/f61cf238b7872b4903e1fc15dcb5a50c.wav"],
  "inserted_id": 3
}
```

- `set` 用作 `FolderDict` 的 key，以解析图像目录。
- 音频解析为 `os.path.join(AudioFolder, "audio", file)`——注意那个硬编码的 `audio` 段，因此 `AudioFolder` 是 `audio/` 的**父目录**。
- `inserted_id`（可选）标记哪一轮 assistant 回复是负样本。
- `"gpt"` 是从 LLaVA 继承的标签，含义是"ground truth"，不是模型名。

### 9.2 七个变体，在源码中选择

`train.py:18-22` 导入七个 `data_utils_*` 模块中的一个，其余都被注释掉了。**切换数据管线意味着改源码，而不是传参数。**

| 变体 | 特征 |
|---|---|
| `neg_patch` | **当前默认** —— 动态切片 + 负样本 |
| `neg_patch_fo` | 增加 `state_labels`（`-101`/`-102`），供全双工预测器使用 |
| `patch` | 同 `neg_patch`，少了 `preprocess_mixtral_zh` |
| `patch_sf` | slow-fast 视频抽帧 |
| `neg_frameCat` | 沿通道拼接 5 帧，而非切片 |
| `video_patch_audio`、`video_audio` | 更早的迭代版本 |

它们高度重复（各约 1400 行，合计约 9500 行）。这是研究迭代留下的产物而非有意设计——改任何东西之前，先确认 `train.py` 到底 import 了哪一个。

### 9.3 Label 掩码

`preprocess_qwen2p5_instruct`（`data_utils_video_audio_neg_patch.py:526`）按角色分隔符切分渲染后的对话，把每一段指令用 `IGNORE_INDEX` 掩掉，只保留 assistant 回复参与监督。

它结尾有一处一致性检查，值得特别注意：

```python
if cur_len != total_len:
    target[:] = IGNORE_INDEX
    print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")
```

一旦长度不匹配，该样本就被**静默地从 loss 中剔除**——整行 label 变成 `IGNORE_INDEX`。它会打印一条警告，但在 `logging_steps` 的噪声中很容易漏掉，从而察觉不到某次运行里大部分样本其实毫无贡献。**如果 loss 看起来异常平坦，先 grep 日志里的 `tokenization mismatch`。**

### 9.4 Collator

`DataCollatorForSupervisedDataset`（第 1390 行）：

- pad `input_ids` 和 `labels`，构建 attention mask。
- 当 `pad_token_id == eos_token_id` 时，临时把 eos 改写为 `-300`，避免 padding 掩掉真实的 eos，之后再还原。
- 把每样本的瓦片/片段列表摊平成单一 batch 维度。
- 将音频打包为 `batch["audios"] = {audios, lengths, lengths_for_llm}`。

`lengths` 与 `lengths_for_llm` 含义不同：前者是给编码器的 fbank 帧数，后者是下采样后给 LLM 的 token 数。

## 10. 实时全双工 demo

`web_demo/server.py`（约 1060 行），Flask + SocketIO，vLLM 加速。

```
麦克风 ──> WakeupAndVAD (silero) ──> 音频分块
                                        │
                            ┌───────────┴─── 模型 worker (vLLM) ─── 首字符？
                            │                                        ├─ ☟ -> 丢弃，保持沉默
                            │                                        └─ 其他 -> 流式输出文本
                            └─── tts_worker ──> TiCodec 声码器 ──> PCM ──> 浏览器
```

容易踩的前置条件：

- **必须给 vLLM 打补丁。** VITA 尚未合入 vLLM 上游，因此需把 `web_demo/vllm_tools/vllm_file/*` 拷入 vLLM 的 `model_executor/models/`，并把 `qwen2p5_model_weight_file/*` 覆盖到 checkpoint 上。
- **silero VAD 未内置** —— 需自行下载 `silero_vad.onnx` / `.jit` 到 `web_demo/wakeup_and_vad/resource/`。
- **在 demo 配置中设 `max_dynamic_patch: 1`。** 保持默认的 12 会让单图消耗 3328 token，交互性彻底崩坏。

注意 demo 重新定义了自己的 token 索引（`IMAGE_TOKEN_INDEX = 51000`，`server.py:61`），并未从 `vita/constants.py` 导入——vLLM 路径使用真实词表槽位，而非负数标记。

## 11. 模型变体

四个 `VITA*ForCausalLM` 类，由 `--model_type` 分发：

| `model_type` | 类 | 骨干 | 状态 |
|---|---|---|---|
| `qwen2p5_instruct` | `VITAQwen2ForCausalLM` | Qwen2.5-7B-Instruct | **VITA-1.5，主路径** |
| `qwen2p5_fo_instruct` | `VITAFOQwen2ForCausalLM` | Qwen2.5-7B | 全双工实验 |
| `mixtral-8x7b` | `VITAMixtralForCausalLM` | Mixtral-8×7B | VITA-1.0 遗留 |
| `nemo` | `VITAMistralForCausalLM` | Mistral-Nemo | 变体 |

### 猴子补丁

`vita_qwen2.py:125`：

```python
Qwen2ForCausalLM.forward = custom_forward
```

该类的 `forward` 在 **import 时被全局替换**，作用于进程内**所有** `Qwen2ForCausalLM`，而不仅是 VITA 的子类。打过补丁的版本增加了全双工变体所需的 `output_hidden_states` 处理。两个后果：它把代码与特定 `transformers` 版本紧密耦合（见[§12](#12-已知缺陷与粗糙之处)）；同一进程内任何实例化原生 Qwen2 的库也会拿到被改过的行为。

### 全双工变体

`vita_fo_qwen2.py` 增加了一个 `predictor_head`——在最后一层 hidden state 上做线性分类，预测用户状态（说话中 / 未说话 / …）。其 loss 被加到 LM loss 上：

```python
state_logits = self.predictor_head(outputs[2][-1]).view(-1, self.predict_usr_state+1)
state_loss = loss_fct(state_logits, s_labels)
loss = loss + state_loss
outputs['loss'] = loss
```

这是添加任何辅助 loss（包括 RL）时可以照抄的模板。

## 12. 已知缺陷与粗糙之处

以下这些若事先不知道就会耗费时间。标为"已修"的是本 fork 的改动。

| 问题 | 状态 |
|---|---|
| `cache_position` 在固定的 `transformers==4.41.1` 上导致生成失败——上游在同一个提交里加入这段代码和这个版本固定，而两者对 Qwen2 路径从未一致 | **本 fork 已修**（[REPRODUCE_zh-CN.md](./REPRODUCE_zh-CN.md#必须的代码修复)） |
| `prepare_inputs_labels_for_multimodal` 在 `audios is None` 时置 `audio_features = None`，但六行后无条件解引用它——于是 `None` 分支不可达，所有调用方只能传一个假的 `torch.zeros(400, 80)`，纯文本和纯图像批次也要白跑 341M 参数的音频编码器 | **本 fork 已修**（`tools/test_audio_optional.py`） |
| `DataConfig` 缺少 `Pretrain_video0` / `Pretrain_audio`，而多个脚本会传这两个值 → `KeyError` | **本 fork 已修** |
| `vita_nemo.py:78,178` 有完全相同的 `cache_position` 缺陷 | **未修** —— 缺少 Nemo 权重无法测试 |
| `requirements.txt` 无法顺利安装（未固定的 `xformers` 要求 `torch>=2.10`；未固定的 `pillow` 需要较新 gcc）；`six`/`timm`/`einops`/`PyYAML`/`opencv`/`librosa` 被 import 但未列出 | 已绕过（[REPRODUCE_zh-CN.md](./REPRODUCE_zh-CN.md#与上游-requirementstxt-的偏离)） |
| `script/train/` 中遍布硬编码的集群路径（`/mnt/cfs/lhj/…`），以及钉死在作者内网的 `MASTER_ADDR`/`INDEX` | 必须在本地修改 |
| `constants.py` 中的 `GLOBAL_WEIGHTS_PATH` 仍是字面量 `/path/to/model_weights` | 仅在 LoRA 分支被用到 |
| `mm_projector_lr` 已不再影响 `mm_projector`（`vita_trainer.py:190`） | 上游问题，未修 |
| tokenization 长度不匹配会静默作废该样本的 label | 上游行为，见[§9.3](#93-label-掩码) |
| `command.sh` 是作者的命令备忘，引用了已删除的文件——不是入口 | —— |
| `Conversation.get_prompt()` **不幂等**：内部执行 `self.system = self.system[0]`，把三元素列表就地换成字符串，于是同一对象第二次调用时索引到的是字符串首字符，整段 system prompt 塌成一个字母。当前无害（数据管线每样本 copy 一次且只调一次），但多轮 RL rollout 复用 conversation 对象时会触发 | 上游缺陷，未修（[PRIMER.md §6.2](./PRIMER.md#62-get_prompt-不幂等未记录的缺陷)） |
| LoRA 路径根本跑不起来：`find_all_linear_names` 没有排除 `audio_encoder`，而 whale 里有两个 `nn.Linear` 的叶子名是数字 `"0"`（`encoder.enc.0.core.out.0`、`encoder.enc.1.embed.0`）。peft 按后缀匹配目标模块，`"0"` 于是命中 `layers.0`——整个 `Qwen2DecoderLayer`——被 peft 拒绝。与显存无关，`--lora_enable True` 必然失败 | **本 fork 已修**（`script/train/smoke_test_lora.sh`，单卡峰值 23.3 GB） |
| `train.py` 中存在 4-bit 路径，但没有任何发布脚本使用 | 未验证 |
| README 的音频示例引用 `asset/vita_newlog.png`，实际文件是 `.jpg` | —— |
| **未提供任何训练数据。** 论文中约 2000 万条 QA 来自约 20 个第三方数据集，另有约 570 万条未发布的合成样本和 11 万小时**内部** ASR 数据 | 见[§13](#13-rl-该接在哪里) |

## 13. RL 该接在哪里

上游**只有监督微调**——没有 reward model、没有偏好优化、没有 rollout 循环（`grep -rE 'reward|ppo|dpo|grpo|rlhf'` 零命中）。增加 RL 是本 fork 的目标。

**有利条件：**

- `VITAQwen2ForCausalLM` 是标准的 `Qwen2ForCausalLM` 子类，返回 `CausalLMOutputWithPast`，logits 可直接用于计算 log-prob。
- `generate()` 已能处理多模态输入。
- `vita_fo_qwen2.py` 演示了如何把自定义 loss 项加到基础 loss 上。
- `VITATrainer.training_step` 是显而易见的注入点。
- 冻结开关让"训练 LLM、冻结编码器"变成改一个参数的事。

**阻碍：**

1. **`generate()` 被 `@torch.no_grad()` 装饰**（`vita_qwen2.py:197`）。在线 RL 需要对采样序列回传梯度——标准做法是用单独的 `forward` 重算 log-prob，而这引出下一个问题。

2. **prompt 的 token 序列不会留存**（[§3](#3-核心思想负数索引占位-token)、[§5](#5-prepare_inputs_labels_for_multimodal模型的心脏)）。`generate()` 消费的是 `inputs_embeds`；采样输出只返回新生成的文本 token。重算 log-prob 就意味着**重跑视觉和音频编码器**。用 PPO 的话，每步约 3 次编码器前向（policy、ref、critic）。缓解办法：在 rollout 时缓存 `inputs_embeds` 并复用——`forward` 本身已接受 `inputs_embeds`（`vita_qwen2.py:160`）。

3. **猴子补丁**（[§11](#11-模型变体)）使得 TRL 的 `AutoModelForCausalLMWithValueHead` 这类封装存在风险。直接自己写 loss，大概比在 `transformers==4.41.1` 上叠加 TRL 更安全。

4. **状态符号需要一个策略决定**（[§6](#6-状态符号拒答机制)）。开头的 `☜`/`☞`/`☟` 算不算 action 的一部分？如果 reward model 从未见过它，训练与推理的分布就会偏移。最简单的做法：rollout 之后、打分之前把它剥掉。

5. **显存。** PPO 需要 policy + ref + reward + critic，每一份都带着 InternViT 和 whale。在 8 卡上不用 LoRA 共享是不现实的。DPO（policy + ref）和 GRPO（无 critic）要可行得多。已部分缓解：自 `audios` 变为可选（[§12](#12-已知缺陷与粗糙之处)）后，纯文本+图像的 rollout 不再需要在每个模型副本、每一步都跑一遍 341M 的音频编码器。

**建议顺序：** 先做离线 DPO——不需要 rollout，因此障碍 1 和 2 直接消失，只剩障碍 4 和 5，两者都可控。参考模型可以用冻结的基础权重或禁用的 LoRA adapter，避免在显存里放第二个 7B。等在那里验证过 log-prob 计算，GRPO 可以复用它，只需再加 rollout 循环。

注意 RL **不需要**那份缺失的 SFT 数据集：已发布的 VITA-1.5 checkpoint 本身就是训练好的，而偏好数据无论如何都得自己构造。如果确实想先用真实数据跑一遍 SFT，见 [DATASETS.md](./DATASETS.md)——论文 2213 万条里约三分之一未发布，但公开的部分已经够用，文档给了三档匹配磁盘预算的方案。
