# 前置知识：彻底读懂 VITA-RL

> Language: **中文** — this primer is written in Chinese. The code references,
> line numbers and measured figures are language-neutral.
>
> 目的：把读懂这个代码库需要的背景，一次性讲清楚。假设你会 Python 和基本的
> 深度学习，但不假设你熟悉多模态 LLM、LLaVA 系架构、DeepSpeed 或语音处理。
>
> 本文所有数字都是**在本机实测**得出的，不是从论文或注释里抄的。凡是推算而
> 未实测的，会明确标注。行号对应写作时的代码状态。
>
> 配套文档：[ARCHITECTURE.md](./ARCHITECTURE.md) 是代码走读（更细的调用链），
> [REPRODUCE.md](./REPRODUCE.md) 是环境复现，[DATASETS.md](./DATASETS.md) 是
> 数据调研。本文是**读那三份之前**该先有的底子。

## 目录

- [0. 三十秒速览](#0-三十秒速览)
- [1. 必备背景概念](#1-必备背景概念)
- [2. 核心机制：负数索引占位符](#2-核心机制负数索引占位符)
- [3. Token 预算：最该先算清的账](#3-token-预算最该先算清的账)
- [4. 三个编码器](#4-三个编码器)
- [5. 状态 token：拒答机制](#5-状态-token拒答机制)
- [6. 对话模板](#6-对话模板)
- [7. 训练：三阶段与冻结开关](#7-训练三阶段与冻结开关)
- [8. DeepSpeed ZeRO 与显存](#8-deepspeed-zero-与显存)
- [9. 语音输出：论文与代码不一致](#9-语音输出论文与代码不一致)
- [10. 踩坑清单](#10-踩坑清单)
- [11. 术语表](#11-术语表)
- [12. 建议的阅读顺序](#12-建议的阅读顺序)

---

## 0. 三十秒速览

VITA-1.5 是一个**全模态 LLM**：一个语言模型同时吃图像、视频、音频、文本，
输出文本；再经一个独立解码器输出语音。

```
图像/视频 ──> InternViT-300M ──> MLP 投影 ──┐
                                             ├──> Qwen2.5-7B ──> 文本
音频 ──────> whale 编码器 ──> CNN adapter ──┘         │
                                                      └──> TTS 解码器 ──> 语音
```

**唯一需要记住的核心思想**：所有模态都被转换成**与 LLM 词嵌入同维度的向量**，
然后拼接进 token 嵌入序列。LLM 本身是未经修改的 Qwen2.5——它从不知道什么叫
"图像"，只是收到了一些恰好来自图片的向量。

这套思路来自 LLaVA。VITA 的独特之处有三：端到端语音输出、负样本拒答训练、
渐进式三阶段训练。

**本仓库（VITA-RL）是 fork**，目标是复现后加 RL 阶段。
上游代码里 `grep -rniE 'reward|ppo|dpo|grpo|rlhf'` **命中数为 0** ——
RL 是这个 fork 自己加的。目前**离线 DPO 和 GRPO 都已实现并在合成数据上验证**
（约 1400 行，见 §12 阶段四），缺的是真实偏好数据和真实任务奖励。

## 1. 必备背景概念

如果下面这些你都熟，可以跳到 §2。

### 1.1 LLaVA 式多模态融合

传统做法是给模型加新的输入类型。LLaVA 的做法更省事：

1. 用一个**视觉编码器**（这里是 InternViT）把图片变成一串向量
2. 用一个**投影层**（projector / adapter，这里是两层 MLP）把这些向量映射到
   LLM 的嵌入空间维度
3. 把它们**当作词嵌入插进序列**

LLM 完全不用改。它看到的只是一串向量，不关心其中哪些来自 `nn.Embedding`
查表、哪些来自一张照片。

**为什么这样可行**：Transformer 的输入本来就是连续向量，token id 只是查表的
索引。跳过查表直接给向量，在数学上毫无区别。

### 1.2 「投影层 / adapter / connector」是同一个东西

不同论文叫法不同，本仓库三种都出现：

- `mm_projector` —— 视觉侧，`vita/model/multimodal_projector/builder.py`
- `adpter` —— 音频侧（**注意这是拼写错误，但贯穿全库，不能改**），
  `whale/adapter.py`
- 论文里叫 connector

功能都是「把编码器输出维度对齐到 LLM 隐藏层维度」。

### 1.3 SFT / RLHF / DPO / PPO / GRPO

本项目的目标涉及这些，简单区分：

| 名词 | 含义 | 需要什么 |
|---|---|---|
| **SFT** | 监督微调，给定输入学着输出标准答案 | 输入-输出对 |
| **RLHF** | 用人类偏好做强化学习的统称 | 偏好数据 |
| **PPO** | 经典 RL 算法，需要 policy/ref/reward/critic 四个模型 | 偏好数据 + 大量显存 |
| **DPO** | 直接偏好优化，跳过 reward model，只要 policy + ref | 偏好数据，显存友好 |
| **GRPO** | 组相对策略优化，有 rollout 但无 critic | **只要 prompt** + 奖励函数 + rollout |

上游只有 SFT。**本 fork 实现了 DPO 和 GRPO**，PPO 没做
（要多带一个 critic，显存代价大而收益不明显）。

两者的关键区别：**DPO 的偏好由数据给定，GRPO 的奖励在训练中实时算**。
所以 GRPO 只需要 prompt——回答由模型自己采样，再由奖励函数打分。
走读见 [ARCHITECTURE.md §14](./ARCHITECTURE.md#14-the-rl-stack-dpo-and-grpo)；
[§13](./ARCHITECTURE.md#13-where-rl-would-attach) 是动手前写的障碍分析，
可以对照看哪些障碍是真的。

### 1.4 语音处理最小知识

读 whale 编码器需要三个概念：

- **fbank（filter bank）**：把波形变成「时间 × 频率」的二维特征图。
  本项目配置：25 ms 窗口、10 ms 步长、80 个梅尔频带 → **每秒 100 帧**。
- **CMVN（倒谱均值方差归一化）**：用全局统计量把特征归一化，
  消除录音设备和环境差异。权重里的 `global_cmvn` 文件就是这些统计量。
- **CTC**：一种不需要逐帧对齐标注的语音识别损失。阶段 2.1 用它训编码器。

### 1.5 bf16 / fp16 / fp32

- **fp32**：32 位，精度高、占显存
- **fp16**：16 位，范围窄，容易溢出
- **bf16**：16 位，范围与 fp32 相同但精度低，**训练首选**，需要 Ampere 及以上

本项目训练用 bf16。一个实际后果：**多卡归约顺序不同会导致同样的输入产生
微小差异**——实测重跑冒烟测试，前两步 loss 完全相同，第三步差 5e-4。
这是正常的，不是 bug。

## 2. 核心机制：负数索引占位符

**这是整个代码库最需要先理解的东西。** 不懂这个，`vita_arch.py` 完全读不懂。

`vita/constants.py`：

```python
IMAGE_TOKEN_INDEX = -200
AUDIO_TOKEN_INDEX = -500
IGNORE_INDEX = -100          # 标准的 PyTorch 忽略标记
```

### 2.1 为什么用负数

`<image>` 和 `<audio>` **从不进入 tokenizer**。它们被替换成负数 id：

```python
# vita/util/mm_utils.py:73 tokenizer_image_audio_token
for chunk in re.split(r"(<audio>|<image>)", prompt):
    if chunk == "<audio>":   prompt_chunks.append([AUDIO_TOKEN_INDEX])
    elif chunk == "<image>": prompt_chunks.append([IMAGE_TOKEN_INDEX])
    else:                    prompt_chunks.append(tokenizer(chunk).input_ids)
```

**为什么是负数而不是新增特殊 token**：负数天然不可能与任何真实 token id 冲突
（真实 id 都 ≥ 0），所以不用扩词表、不用改 embedding 层大小。代价是这些 id
**不能直接喂给模型**——必须在前向之前全部替换掉。

### 2.2 替换发生在哪

`vita/model/vita_arch.py:333` 的 `prepare_inputs_labels_for_multimodal`。
这是全库最重要的函数，做四件事：

1. 按负数 id 把序列**切成若干段**
2. 对文本段查 embedding
3. 在切口处**插入**视觉/音频特征
4. 重新 padding 对齐

**最关键的一行**（`vita_arch.py:651`）：

```python
return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
#      ^^^^ input_ids 被刻意置为 None
```

**这意味着：调用它之后，原始的 token 序列不复存在**，下游只有
`inputs_embeds`。

**这一点对 RL 至关重要**：`generate()` 采样后只能拿回新生成的文本 token，
拿不回 prompt 的 token 序列。要重算 log-prob，就得重跑视觉和音频编码器。
缓解办法：把 rollout 时的 `inputs_embeds` 缓存下来复用——
`forward` 本来就接受这个参数（`vita_qwen2.py:160`）。

### 2.3 一个直观的例子

假设输入 `"<image>\n这是什么?"`：

```
tokenize 后:  [-200, 8948, 3837, 30]        ← -200 是占位符
                ↓ prepare_inputs_labels_for_multimodal
切段:         []  |  -200  |  [8948, 3837, 30]
查表/编码:    []  |  256×4096 的图像特征  |  3 个词嵌入
拼接:         [256 个图像向量] + [3 个词向量] = 259 × 3584
```

注意图像特征是 4096 维，而 LLM 是 3584 维——中间还有 `mm_projector` 做映射，
见 §4.1。

## 3. Token 预算：最该先算清的账

这是理解显存、序列长度、训练速度的基础。**以下数字全部实测。**

### 3.1 视觉

```
InternViT 输入 448×448，patch 大小 14
  → (448/14)² = 32×32 = 1024 个 patch
  → 去掉 CLS token 后仍是 1024
  → pixel_shuffle(0.5)：token 数 ÷4，通道数 ×4
  → 256 个 token，每个 1024×4 = 4096 维
```

实测验证（真实模型跑通）：

```
输入 2×3×448×448 → 输出 (2, 256, 4096)
hidden_size = 4096 = config 1024 × 4
```

**记住这个数：一个 448×448 图块 = 256 个 token。**

`pixel_shuffle` 是空间换通道的技巧：把相邻像素的信息折叠进通道维，
token 数减少 4 倍，信息不丢。这是 InternVL 系列的常用手法，
直接决定了一张图占多少上下文。

### 3.2 动态切图（dynamic tiling）

大图不是简单缩放，而是切成多个 448×448 的块，再加一张全图缩略图。
实测不同长宽比的结果：

| 原图尺寸 | 切块数 | token 数 |
|---|---|---|
| 448×448 | 1 | 256 |
| 896×448 | 3 | 768 |
| 1920×1080 | 9 | 2304 |
| 500×2000 | 5 | 1280 |

（`max_dynamic_patch=12`，`use_thumbnail=True`）

**注意 1920×1080 这种常见分辨率就要 2304 token**，占默认
`model_max_length=6200` 的 37%。

### 3.3 音频

```
fbank: 25 ms 窗 / 10 ms 移 → 每秒 100 帧
whale 下采样：attn_mask[2::2][2::2][0::2] → 共 8 倍
  → 每秒约 12.5 个 token
```

| 语音时长 | token 数 |
|---|---|
| 1 秒 | ~12 |
| 5 秒 | ~62 |
| 10 秒 | ~125 |
| 30 秒 | ~375 |

**音频比图像便宜得多**：30 秒语音（375 token）还不如两个图块（512 token）。

### 3.4 上下文预算的实际含义

默认 `model_max_length=6200`：

| 内容 | token | 占比 |
|---|---|---|
| 一张 12 块 + 缩略图的图 | 3328 | 54% |
| 16 帧视频（每帧 1 块） | 4096 | 66% |
| 30 秒语音 | 375 | 6% |

**所以视觉是绝对的大头。** 视频训练时 `MAX_IMAGE_LENGTH=16`
（`constants.py:2`）这个上限就是这么来的——再多就装不下了。

## 4. 三个编码器

### 4.1 视觉：InternViT-300M-448px

`vita/model/multimodal_encoder/internvit/internvit_encoder.py`

- 参数量实测 **289.9M**（推理日志会打印）
- `select_layer = -1`，取最后一层
- **默认冻结**（`load_model()` 里 `requires_grad_(False)`），
  靠 `--unfreeze_vision_tower` 解冻
- 输出 4096 维 → `mm_projector`（两层 MLP + GELU）→ 3584 维

投影器由 `mm_projector_type` 字符串选择，正则匹配
`^mlp(\d+)x_gelu$`。默认 `mlp2x_gelu` = `Linear(4096→3584) + GELU +
Linear(3584→3584)`。代码里还有 `spp`/`ldp`/`vanilla`/`minigpt` 等，
VITA-1.5 都没用。

### 4.2 音频：whale

`vita/model/multimodal_encoder/whale/`，自研，参数量实测 **341.4M**。

**重要：代码里的默认值和实际权重的配置不一样。** 以下读自
实际下载的 `train.yaml`：

| 配置项 | 实际值 |
|---|---|
| `input_dim` | 80（梅尔频带数） |
| `encoder-layer-config` | `subsampling-transformer` |
| `encoder-output-dim` | 1024 |
| `adpter_type` | **`subsampling`**（不是代码默认的 `cnn`） |
| `llm_embed_dim` | 3584 |
| 采样率 | 16000 Hz |

whale 是**可配置的组件流水线**——`layer-config` 字符串决定串哪些模块
（`subsampling` / `transformer` / `mamba` / `fsmn` / `conv1d` ...）。
读代码时别照着默认参数理解，要看权重目录里的 yaml。

### 4.3 TTS 解码器

见 §9——这里有个论文与代码不一致的地方，单独讲。

## 5. 状态 token：拒答机制

这是 VITA 相对普通 VLM 最有特色的设计，也是**最容易被忽略的一环**。

`data_utils_video_audio_neg_patch.py:128-135` 给每条 gpt 回复强制加前缀：

```python
if i == inserted_id:                        # 负样本
    sentence["value"] = "☟" + sentence["value"]
elif sentence["from"] == "gpt":
    if "<audio>" in source[i - 1]["value"]:
        sentence["value"] = "☞" + sentence["value"]   # 回答语音提问
    else:
        sentence["value"] = "☜" + sentence["value"]   # 回答文本提问
```

| 符号 | 含义 |
|---|---|
| `☜` | 回答**文本**提问 |
| `☞` | 回答**语音**提问 |
| `☟` | **拒答**（噪声/非语音输入） |

**为什么需要它**：常开麦克风的场景下，模型会一直收到环境噪音。
必须能判断「这不是在跟我说话」并拒绝回答，否则无法实用。

**实测验证**（本机三条推理）：

| 查询 | 输出前缀 |
|---|---|
| 文本提问 | `☜` |
| `q1.wav` 正常语音 | `☞` |
| `q2.wav` 噪声 | `☟` |

三种情况完全正确区分。

**tokenizer 层面的事实**（实测）：这三个符号在 Qwen2.5 原生词表里
**各占恰好 1 个 token**（id 分别是 145789 / 144766 / 146164），
不是新增的特殊 token。

**对 RL 的影响**：这个前缀算不算 action 的一部分？如果 reward model
看不到它，训练和推理的分布就会不一致。最简单的处理是 rollout 之后、
打分之前把它剥掉。

## 6. 对话模板

`vita/conversation.py` 定义 9 种模板，VITA-1.5 用 `qwen2p5_instruct`。

### 6.1 system prompt 按模态切换

这个设计容易看漏：`system` 字段是一个**三元素列表**，
`get_prompt(modality)` 按模态选一个：

```python
system=[
    "...你必须严格根据用户给的图像内容回答...注意你看到的是图像，不是视频。",   # [0] image
    "...你必须严格根据用户给的视频内容回答...注意你看到的是视频，不是图像。",   # [1] video
    "...（无视觉约束）",                                                    # [2] lang
]
```

调用时 `conv.get_prompt("image")` / `("video")` / `("lang")` 三选一。
传错会触发 `assert`。

### 6.2 get_prompt 不幂等（未记录的缺陷）

**这是我实测发现的、文档里没记的问题。**

`get_prompt()` 内部执行 `self.system = self.system[0]` ——
把 list **就地替换成 str**。于是第二次调用同一个对象时，
`self.system[0]` 取到的是字符串的**首字符**：

```
第 1 次调用：system 是 659 字符的完整提示
第 2 次调用：'<|im_start|>system\nY<|im_end|>\n...'
                                 ↑ 整段提示塌成一个字母 "Y"
```

**当前为什么没炸**：数据管线每个样本 `conv_templates[...].copy()` 一次、
只调一次 `get_prompt()`。全局模板也没被污染（因为是重新绑定而非原地改）。

**什么时候会炸**：任何复用同一个 conv 对象调两次的代码。
**RL 的多轮 rollout 是高危场景**——写 RL 循环时务必每次重新 `.copy()`。

## 7. 训练：三阶段与冻结开关

### 7.1 论文的三阶段

| 阶段 | 数据 | 训练什么 |
|---|---|---|
| 1.1 视觉对齐 | 20% caption | 仅视觉 adapter |
| 1.2 视觉理解 | 100% caption | 视觉编码器 + adapter + LLM |
| 1.3 视觉 SFT | 100% QA + 20% caption | 同上 |
| 2.1 音频对齐 | 11,000 小时 ASR | 语音编码器（CTC）→ 语音 adapter |
| 2.2 音频 SFT | 4% caption + 20% QA | 全部 + 状态分类头 |
| 3.1 Codec | 3,000 小时 | TiCodec |
| 3.2 解码器 | 同上 | NAR + AR 解码器，**LLM 冻结** |

**值得注意**：阶段 2.2 只用 4% caption + 20% QA，阶段 3 完全冻结 LLM。
所以**不必凑齐论文的 2213 万条数据才能做有意义的训练**。

### 7.2 冻结开关

全在 `vita/train/train.py:377-425`，套路统一是
「先全冻，再解冻指定模块」：

```python
if model_args.tune_mm_mlp_adapter:
    model.requires_grad_(False)                              # 全冻
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = True                               # 只放开投影器
```

关键开关：

| 参数 | 作用 |
|---|---|
| `--tune_mm_mlp_adapter` | 只训视觉投影器（阶段 1.1） |
| `--tune_audio_mlp_adapter` | 只训音频 adapter |
| `--freeze_audio_encoder` | 冻结 whale 主体（默认 True） |
| `--unfreeze_vision_tower` | 解冻 InternViT |
| `--audio_prompt_finetune` | 只训音频 prompt embedding |

**对 RL 友好的一点**：「训 LLM、冻编码器」是一个开关的事。

### 7.3 数据管线在源码里选，不在命令行

`vita/train/train.py:17-21`：

```python
from vita.util.data_utils_video_audio_neg_patch import make_supervised_data_module, DataArguments
#from vita.util.data_utils_video_audio_neg_patch_fo import ...
#from vita.util.data_utils_video_audio_patch import ...
```

**换数据管线要改源码，没有命令行参数。** 七个变体共约 9500 行，
互为近似拷贝（`neg_patch_fo` 与 `neg_patch` 只差 58 行）。
这是研究迭代的产物，不是有意设计。**改任何东西之前，先确认
`train.py` 到底 import 了哪一个。**

## 8. DeepSpeed ZeRO 与显存

### 8.1 为什么必须 8 卡

7B 全参数训练在**单张 80 GB 卡上放不下**。算账：

| 项 | 大小 |
|---|---|
| bf16 权重 | ~14 GB |
| fp32 主权重 | ~28 GB |
| AdamW 两个动量（fp32） | ~56 GB |
| **合计** | **~98 GB** > 80 GB |

单卡实际表现：前向反向都能过，在分配 `exp_avg_sq` 时 OOM。

**ZeRO-3 把这三样都切分到各 rank**，8 卡就能跑。实测冒烟测试
每卡峰值约 18 GB。

### 8.2 ZeRO 三个级别

| 级别 | 切分什么 |
|---|---|
| ZeRO-1 | 优化器状态 |
| ZeRO-2 | + 梯度 |
| ZeRO-3 | + 模型参数本身 |

本项目用 ZeRO-3（`script/deepspeed/zero3.json`）。其中
`stage3_gather_16bit_weights_on_model_save: true` 保证保存时把
分片重新聚合成完整权重——所以产出的 checkpoint 可以直接加载推理，
实测验证过。

### 8.3 一个容易踩的 ZeRO 坑

`vita_arch.py` 里有些看起来多余的**零长度切片**：

```python
cur_input_embeds_parts.append(cur_audio_features[0:0])   # 长度为 0！
```

**这不是冗余代码。** 它把编码器挂在 autograd 图上，
让 DDP/ZeRO 不会把编码器参数报成「未使用」。删掉会报
`did not receive grad`。修改这一带代码时务必保留。

## 9. 语音输出：论文与代码不一致

**这是我实测发现的、值得单独讲的一点。**

论文说 VITA-1.5 的卖点是「端到端语音输出——解码器直接消费 LLM 的
hidden state，不需要外挂 TTS」，延迟因此从 4 秒降到 1.5 秒。

但**仓库里的 demo 代码不是这么做的**。`web_demo/server.py:609`：

```python
embeddings = llm_embedding(torch.tensor(tokenizer.encode(tts_input_text)).to(device))
for seg in tts.run(embeddings.reshape(-1, 896).unsqueeze(0), ...):
```

`tts_input_text` 是**已经生成好的文本**。这里把它重新 tokenize、
重新查 embedding，再喂给 TTS。这是传统的
「文本 → embedding → 语音」级联，**不是 hidden state 直连**。

维度也对得上这个判断：

| 项 | 维度 |
|---|---|
| 主 LLM hidden_size | **3584** |
| TTS decoder idim | **896** |
| 比值 | 3584 / 896 = **4.0** |

`reshape(-1, 896)` 把每个 3584 维向量**拆成 4 个 896 维**，
序列长度变 4 倍。这是刻意设计（整除），但确认了输入是
**文本 embedding 而非 LLM 输出的 hidden state**。

TTS 权重结构（实测）：

```
vita_tts_ckpt/
├── codec/     final.pt + model.json   # TiCodec，24 kHz，1024 码本
└── decoder/   final.pt + model.json   # idim 896, odim 1024
```

**结论**：如果你的目标涉及语音输出，别只读论文。要么代码里的是简化版，
要么完整实现没放出来。这不影响视觉和语音**输入**部分——那两块与论文一致，
已实测跑通。

## 10. 踩坑清单

按「不知道会浪费多少时间」排序。

### 10.1 会让你困惑很久的

| 坑 | 说明 |
|---|---|
| **假音频** | 上游 `audios=None` 时会崩（`None` 下标），所以每次纯文本前向都要塞 `torch.zeros(400, 80)`，白跑 341M 编码器。**本 fork 已修**，见 `tools/test_audio_optional.py` |
| **分词长度不匹配静默作废样本** | `data_utils:642` 把整条 label 置 `IGNORE_INDEX`，样本照常前反向但对 loss 零贡献，只打一行 WARNING。**接真实数据后务必统计这行的出现次数** |
| **`get_prompt()` 不幂等** | 见 §6.2 |
| **`adpter` 拼写错误** | 音频 adapter 全库都拼作 `adpter`，改了会到处崩 |
| **数据管线在源码切换** | 见 §7.3 |

### 10.2 环境相关

| 坑 | 说明 |
|---|---|
| `requirements.txt` 装不上 | 未固定的 `xformers` 要求 torch≥2.10，与固定的 `torch==2.3.1` 冲突。见 [REPRODUCE.md](./REPRODUCE.md) 的分阶段安装 |
| numpy 2.x | torch 2.3.1 早于 numpy 2.0 的 ABI 变更。`numba`/`librosa`/`opencv` 会把 numpy 2 拉回来，**装完要重新固定** |
| flash-attn 必须有 | `train.py` 硬编码 `attn_implementation="flash_attention_2"` |
| `command.sh` 不是构建脚本 | 那是原作者的命令历史，引用的文件很多已不存在 |

### 10.3 配置相关

| 坑 | 说明 |
|---|---|
| 硬编码集群路径 | 17 个训练脚本引用 `/mnt/cfs/lhj/...`；`*_nodes.sh` 把 `MASTER_ADDR` 写死成作者的内网 IP |
| `GLOBAL_WEIGHTS_PATH` 是占位符 | `constants.py:14` 仍是字面量 `/path/to/model_weights` |
| checkpoint 里是 HF repo ID | 加载会走网络。用 `tools/localize_config.py` 改成本地路径 |
| `AudioFolder` 的拼接约定 | 路径是 `os.path.join(AudioFolder, "audio", file)`——中间的 `"audio"` 是**硬编码**的，所以 `AudioFolder` 要指向 `audio/` 的**父目录** |

## 11. 术语表

| 术语 | 含义 |
|---|---|
| **VITA-1.5** | 上游模型。本仓库是它的 fork |
| **InternViT** | 视觉编码器，300M 参数，448×448 输入 |
| **whale** | VITA 自研的音频编码器，341M 参数 |
| **TiCodec** | 语音编解码器，把波形离散成 token |
| **mm_projector** | 视觉投影层，4096 → 3584 |
| **adpter** | 音频 adapter（拼写错误但不能改） |
| **pixel_shuffle** | 空间换通道，token 数 ÷4 通道 ×4 |
| **dynamic tiling** | 大图切成多个 448×448 块 + 缩略图 |
| **fbank** | 梅尔滤波器组特征，语音的标准输入表示 |
| **CMVN** | 倒谱均值方差归一化 |
| **CTC** | 不需逐帧对齐的语音识别损失 |
| **ZeRO-3** | DeepSpeed 的显存优化，切分参数/梯度/优化器状态 |
| **状态 token** | `☜`/`☞`/`☟`，标记回复类型 |
| **负样本 / neg** | 训练模型拒答的噪声样本 |
| **`inserted_id`** | 样本里标记哪一轮 gpt 回复是负样本 |
| **fo（full-duplex）** | 全双工变体，`vita_fo_qwen2.py` |
| **sf（slow-fast）** | 视频抽帧策略，关键帧多 token、其余少 token |
| **LoRA** | 低秩适配，只训练一小组矩阵。本项目 161.5M 可训练（占 2.12%） |
| **adapter（peft）** | LoRA 插入的那组矩阵。关掉它就还原基座——这是参考模型的实现方式 |
| **参考模型 / ref** | RL 里作为「不要偏离太远」基准的固定策略 |
| **rollout** | GRPO 中模型自己采样出的回答。每个 prompt 采 G 个为一组 |
| **优势 / advantage** | 某个回答比同组平均好多少。GRPO 用 `(r-mean)/std` |
| **退化组 / degenerate group** | 组内奖励全同，优势为 0、无梯度。规则奖励下很常见 |
| **KL 惩罚** | 限制策略偏离参考的程度。GRPO 用 k3 估计量，恒非负 |

## 12. 建议的阅读顺序

按这个顺序，四个阶段。每段都标了大致耗时和是否需要 GPU，
可以按你手头的条件挑。

### 阶段一：理解机制（约 1 小时，纯阅读）

这一步不碰环境，目的是让后面的代码读得懂。

1. `vita/constants.py`（14 行）——先看负数索引，全库的地基
2. **本文 §2 和 §3**——占位符机制 + token 预算
3. `vita/model/vita_arch.py:333-651` 的
   `prepare_inputs_labels_for_multimodal`——**全库最重要的函数**。
   配合 [ARCHITECTURE.md §5](./ARCHITECTURE.md#5-prepare_inputs_labels_for_multimodal-the-heart-of-the-model)
   的分步走读一起读

读完应该能回答：为什么模型收到的是 `inputs_embeds` 而不是 `input_ids`？

### 阶段二：跑起来（约 2 小时，需要 GPU）

**看懂和跑通是两回事**，这一步能消掉大量误解。

4. 按 [REPRODUCE.md](./REPRODUCE.md) 装环境、下权重
   （分阶段安装，顺序敏感）
5. 跑三条推理，观察 `☜`/`☞`/`☟` 的区别
   —— 命令在 [HANDBOOK.md §2.1](./HANDBOOK.md#2-常用命令速查)
6. `python tools/test_audio_optional.py`（CPU 几秒，不用权重）
7. `python tools/inspect_dataset.py --dataset-use SmokeTest`
   —— 看清一条样本从 JSON 变成张量的全过程

### 阶段三：理解训练（约 2 小时）

8. `vita/train/train.py`（520 行）——重点看 §7 讲的冻结开关
   （`:420` 起的那几段「先全冻再解冻」）
9. `script/train/smoke_test_qwen.sh`——唯一路径无关的训练脚本
10. `vita/util/data_utils_video_audio_neg_patch.py` 的
    `__getitem__`（`:882`）和 `DataCollatorForSupervisedDataset`（`:1390`）
11. 有 8 卡的话跑一次 `smoke_test_qwen.sh`；只有单卡就跑
    `smoke_test_lora.sh`（峰值 23.3 GB）

### 阶段四：本 fork 的 RL 部分（约 2 小时）

**这是上游没有的东西，也是这个 fork 存在的理由。**
先读 [ARCHITECTURE.md §14](./ARCHITECTURE.md#14-the-rl-stack-dpo-and-grpo)
的整体走读，再按下面顺序看代码：

12. `vita/train/dpo_loss.py`（114 行）——最简单的入口，
    纯数学，配 `tools/test_dpo_loss.py` 一起读
13. `vita/train/dpo_trainer.py`（139 行）——看 `compute_loss`
    怎么用 `disable_adapter()` 当参考模型
14. `vita/train/rewards.py`（222 行）——GRPO 的奖励注册表
15. `vita/train/grpo_loss.py`（135 行）——组内归一化 + KL
16. `vita/train/grpo_trainer.py`（228 行）——最复杂的一个，
    rollout → 打分 → 优势 → 损失

**建议配合跑**（都是单卡）：

```bash
python tools/test_dpo_loss.py     # 19 项，秒级，不用权重
python tools/test_grpo_loss.py    # 39 项，同上
bash script/train/dpo_smoke_test.sh /tmp/dpo_out 1    # 看首步 loss = 0.6931
bash script/train/grpo_smoke_test.sh /tmp/grpo_out 1  # 看 reward 上升
```

那两个数字（DPO 的 `0.6931`、GRPO 的首步 KL `0`）是**数学恒等式**，
不是经验值——看到它们就说明参考模型接对了。
详见 [HANDBOOK.md §8](./HANDBOOK.md#8-dpo离线偏好优化) 和
[§9](./HANDBOOK.md#9-grpo组相对策略优化)。

### 各文档什么时候看

| 文档 | 用途 |
|---|---|
| **PRIMER.md**（本文） | 最先看，其余文档的前置知识 |
| **HANDBOOK.md** | 动手时看：命令、地雷、排查表 |
| **ARCHITECTURE.md** | 想弄清某段代码为什么这么写时看 |
| REPRODUCE.md | 装环境时看 |
| DATASETS.md | 要接真实数据时看 |
| MIGRATION.md | 换机器时看 |

### 不建议读

- `command.sh` —— 原作者的命令历史，引用的文件很多已不存在
- `data_utils_*` 的其余六个变体 —— 与启用的那个是近似拷贝
  （`neg_patch_fo` 只差 58 行）
- `vita_mixtral.py` / `vita_nemo.py` —— VITA-1.0 遗留和其他变体，与 1.5 无关
- `vita/model/vita_tts/` —— 除非专门研究语音输出，
  且注意 §9 说的代码与论文不一致

### 读到 `vita_qwen2.py` 时注意

`:125` 有一行全局 monkey patch：

```python
Qwen2ForCausalLM.forward = custom_forward
```

**import 即生效、进程内不可逆。** 这解释了为什么这个项目与
`transformers==4.41.1` 死绑，也是给它套 TRL 之类封装时的主要风险来源。
