# RLAIF-V 上的全参 SFT 与多模态 DPO：实现细节与机制深读

> 本文是对 VITA-RL 项目中"全参 SFT + 多模态 DPO"路线的代码级深读与机制分析，
> 覆盖数据构建、训练实现、显存/架构/动力学、on/off-policy 辨析、数据泄露讨论，
> 以及一个旨在隔离"分布对齐"与"chosen 记忆"两因子的 disjoint 实验设计。
>
> 实验过程与数字见 [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)，
> 项目综述见 [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md)。

---

## 目录

| 章节 | 内容 |
|---|---|
| [§1 共同根基：RLAIF-V 数据](#1-共同根基rlaif-v-数据) | 数据源、偏好来源、管线三件事、数据体检 |
| [§2 全参 SFT 实现细节](#2-全参-sft-实现细节) | 数据构建、训练脚本、入口、loss masking、多模态融合、DeepSpeed、结果 |
| [§3 多模态 DPO 实现细节](#3-多模态-dpo-实现细节) | 数据构建、trainer、compute_loss、disable_adapter、dpo_loss、image_group_size、首步不变量 |
| [§4 ZeRO 显存分析](#4-zero-显存分析) | ZeRO-1/2/3 显存数学、脚本头注释不严谨之处 |
| [§5 mm_projector 的作用](#5-mm_projector-的作用) | 架构、参数量、管线位置 |
| [§6 跟着 LLM 全参 SFT 是否影响已有功能](#6-跟着-llm-全参-sft-是否影响已有功能) | 实测代价 + 四个约束机制 |
| [§7 on-policy 还是 off-policy](#7-on-policy-还是-off-policy) | 三层概念澄清、第 6 轮是 off-policy |
| [§8 先 SFT 再 DPO 的依据](#8-先-sft-再-dpo-的依据) | 探针 52.2%、初始 gap −7.84→+4.75、C vs D 对照 |
| [§9 数据泄露讨论](#9-数据泄露讨论) | 三层：benchmark 无泄露 / SFT-DPO 重叠但被参考项抵消 / 反证 + 方法论保留 |
| [§10 disjoint 实验设计](#10-disjoint-实验设计) | 隔离两因子、脚本、环境阻塞状态 |
| [§11 关键不变量与坑汇总](#11-关键不变量与坑汇总) | |
| [§12 面试问答](#12-面试问答) | 21 问：原理（两种"分不开"、为什么不只 SFT）、实现（disable_adapter 边界、−log0.5 局限、46%）、统计（McNemar、SNR、诚实口径）、追问陷阱 |

---

## 1. 共同根基：RLAIF-V 数据

SFT 和 DPO 两条路线建在同一份数据源上。

### 1.1 数据源与样本

**HuggingFace `openbmb/RLAIF-V-Dataset`**，14 个 parquet 分片约 83000 对，实际用了 4 个分片（`shard000/001/003/004`）。

原始样本字段：`image`（内嵌字节）、`question`、`chosen`（更好回答）、`rejected`（更差回答）。

### 1.2 偏好来源（关键）

查 RLAIF-V 的 `origin_split` 字段：**chosen/rejected 都由 OmniLMM-12B 生成，优劣也由 OmniLMM-12B 判定**。两个回答都不是 VITA 写的——这成为前三轮 DPO 无效的根因（见 §7、§8）。

### 1.3 数据管线必须处理的三件事

| 事项 | 不做会怎样 |
|---|---|
| **图像按内容寻址**（SHA-1 做文件名） | parquet 里图是内嵌字节且跨行重复。15000 对实际只有 **8821 张**不同的图 |
| **补 `<image>` token** | `LazySupervisedDataset` 靠这个字面量决定往哪插视觉特征，**缺了会静默退化成纯文本训练**，loss 曲线照常好看 |
| **丢掉两侧相同的 pair** | DPO logit 恒为 0，loss 永远 `-log(0.5)`，白跑前向。实测 15297 条里有 21 条 |

### 1.4 数据体检（训练前该看的数字）

| 指标 | 3000 对 | 15000 对 | 判读 |
|---|---|---|---|
| chosen 平均长度 | 299 字符 | 307 | |
| rejected 平均长度 | 298 字符 | 307 | 几乎相同 |
| chosen 更长的比例 | 46.5% | 47.1% | **无长度偏置**（50% 为完全无偏） |
| 两回答文本相似度 | 0.29 | 0.294 | **是两个不同的答案**，不是改词 |

这"高质量无捷径"恰恰是对 7B 基座很陡的原因。

---

## 2. 全参 SFT 实现细节

### 2.1 为什么做 SFT——绕开判据难题

DPO 需要"知道哪个更好"，SFT 只需要好答案、不需要判优劣。所以**同一批数据，扔掉 `rejected`、只用 `chosen` 做监督训练**，判据问题自动消失。

更深层：探针测出基座对这批 pair 的可分性只有 53.6%（接近随机）。这句话的另一面是**基座对这个分布建模得不好**。SFT 提升分布内能力后，可分性会自然改善——因为可分性是模型的属性，不只是数据的属性。

### 2.2 数据构建：`tools/make_rlaif_v_sft_data.py`

转换 RLAIF-V parquet → SFT 记录格式，**只保留 `chosen`**（丢 `rejected`）。

- **必填列**：`("image", "question", "chosen")`（注意不需要 `rejected`，对比 DPO 构建器需要 `rejected`）
- `--limit 20000`，`--min-chars 16`（丢太短，"两词答案只教长度不教内容"），`--max-chars 1200`（不让离群点撑长序列）
- **图像内容寻址**：`sha1(raw_bytes).hexdigest()` 做文件名，RGB JPEG quality=95，相同图塌缩成一个文件
- 创建空的 `audio/` 目录（`LazySupervisedDataset` 会拼 `AudioFolder/"audio"` 字面量，缺了会报错）

**输出记录结构**（`make_rlaif_v_sft_data.py:120-130`）：
```python
{
    "set": "rlaif_v_sft",
    "id": "sft_000001",
    "conversations": [
        {"from": "human", "value": "<image>\n" + question},
        {"from": "gpt",   "value": answer},          # 即 chosen
    ],
    "image": "a1b2c3....jpg",
}
```
- 无 `rejected` 字段（与 DPO 构建器的关键区别）
- 无 `audio` 字段 → 走纯图像路径
- `set="rlaif_v_sft"` → 由 `vita/config/dataset_config.py` 的 `FolderDict["rlaif_v_sft"]` 解析图片目录

输出：`rlaif_v_sft_train.json` + `images/<sha1>.jpg`。实际产出：20000 样本、11042 张唯一图、目标长度均值 323 字符。

### 2.3 训练脚本：`script/train/sft_rlaif_v_8gpu.sh`

| 配置项 | 值 | 理由 |
|---|---|---|
| **DeepSpeed** | `zero3.json`（**不是 zero2**） | 全参 7B 单卡需 ~98 GB（bf16 权重 14 + fp32 主权重 28 + AdamW 双动量 56）。zero2 只分片优化器状态、参数仍复制；DPO 能用 zero2 是因为 LoRA 可训练参数极少 |
| **lr** | **1e-6** | 比 LoRA DPO 低一个数量级。基座已校准良好（MME 2353 vs 论文 2362），风险是**用 2 万条 VQA 答案覆盖掉通用能力**，不是欠拟合。与上游 finetune 脚本一致 |
| **冻结视觉塔 + 音频编码器** | — | 2 万条答案远不足以重训感知，解冻是最快搞坏 benchmark 的方式 |
| **workers=0** | — | `/dev/shm` 仅 512MB，8 卡 × 4 worker 走共享内存传 448×448 切片会 OOM（Bus error） |
| optimizer | adamw_torch，weight_decay 0.0 | — |
| scheduler | cosine，warmup_ratio 0.03 | — |
| batch | per_device 1 × grad_accum 4 × 8 GPU = 32 | — |
| epochs / 步数 | 1 / 625 步 | — |
| model_max_length | 6200 | — |
| bf16 + gradient_checkpointing | True | — |

耗时：8 卡、625 步、**1 小时 54 分**、约 30 GB/卡（脚本头注释说 ~18 GB，实测约 30 GB；见 §4）。

### 2.4 训练入口与模型构建：`vita/train/train.py`

`train.py` 是 SFT 和 DPO 共享入口（DPO 通过 `train_dpo.py` 复用）。关键流程：

- `set_random_seed(42)`
- 构造 `VITAQwen2ForCausalLM.from_pretrained(model_name_or_path, torch_dtype=bfloat16, attn_implementation="flash_attention_2")`
- `model.config.use_cache = False`
- `freeze_backbone=False` → **LLM 不冻结，全参训练**
- `gradient_checkpointing=True` + `enable_input_require_grads()`（让冻结的视觉/音频塔不触发 DDP "unused parameters"）
- **LoRA 块跳过**：`lora_enable` 默认 `False`，SFT 脚本不传 `--lora_enable True`，所以 `get_peft_model` 从不调用 → **整个 7B 模型全参训练，无 LoRA**
- `initialize_vision_modules` 挂载 InternViT 塔 + image_processor
- `initialize_audio_modules` 挂载音频编码器 + adapter
- 视觉塔、音频编码器、音频 adapter 移到 bf16/device 但**冻结**（`unfreeze_vision_tower=False`、`freeze_audio_encoder=True`、`freeze_audio_encoder_adapter=True`）
- `freeze_mm_mlp_adapter=False` → **mm_projector 可训练**（与 LLM 一起训练）

**净效果**：整个 LLM（Qwen2.5-7B）+ mm_projector 全参训练；InternViT 视觉塔 + 音频编码器 + 音频 adapter 冻结。这与上游 `finetune_qwen.sh`（`unfreeze_vision_tower True`、`mm_projector_lr 2e-6`、加载 stage-1 projector）形成对比——RLAIF-V SFT 不加载 projector（它已在 VITA-1.5 release checkpoint 里），也不解冻视觉塔。

### 2.5 Loss 计算：response-only masking

标准 next-token CrossEntropy，**shifted logits/labels**，定义在 monkey-patch 的 `custom_forward`（`vita/model/language_model/vita_qwen2.py:98-109`）：

```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:, :].contiguous()
loss_fct = CrossEntropyLoss()              # ignore_index=-100 = IGNORE_INDEX
loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
```

**masking 在对话预处理器里完成**（`preprocess_qwen2p5_instruct`，`data_utils_video_audio_neg_patch.py:526-652`），核心逻辑：

- `targets = input_ids.clone()`
- 按 `\n<|im_start|>user\n` / `\n<|im_start|>assistant\n` 切轮次
- `instruction_len = len(tokenizer_image_token(parts[0]))`（指令部分长度）
- `target[cur_len : cur_len + instruction_len] = IGNORE_INDEX` —— **mask 掉 system + user + 角色标记**
- assistant 回答 token 保留真实 ID → **参与 loss**
- `target[cur_len:] = IGNORE_INDEX`（超出部分 mask）
- **安全网**：若 `cur_len != total_len`（分词不匹配，如图像 token 替换导致），整个 target 设为 IGNORE_INDEX 并打 warning → 样本从 loss 中丢弃（这是 DPO 死对检测的来源）

**所以 loss 只算在 assistant（gpt）回答 token 上**——system、user question、角色标记、图像/音频占位区域全部 mask 成 -100。这不是"全序列"监督。

此外在 `prepare_inputs_labels_for_multimodal` 内部，**图像特征和音频特征在 embedding 序列中的位置也被填成 `IGNORE_INDEX`**——即使拼接进来的视觉特征位置也不参与 loss，只有图像特征之后的文本回答 token 训练。

**无 sequence packing**：每个样本是一条对话（一图、一轮 Q&A）。collator 用 `pad_token_id` 右填充 input_ids、`IGNORE_INDEX` 右填充 labels，截断到 `model_max_length=6200`。

### 2.6 多模态（图像+文本）处理

**步骤 1 — 带图像占位符的分词**：`tokenizer_image_token`（`mm_utils.py:45-70`）按字面量 `<image>` 切分 prompt，在每个图像位置插入 `IMAGE_TOKEN_INDEX = -200`。所以 `input_ids` 里有 `-200` 标记图像特征要插入的位置。

**步骤 2 — 图像加载 + dynamic patching**（`LazySupervisedDataset.__getitem__`）：
- 打开 PIL 图，转 RGB
- `image_aspect_ratio="square"`（SFT 脚本设的）
- `dynamic_preprocess`（`data_utils:1499-1540`）：把图 resize 到最接近的宽高比，切成 `min_num=1, max_num=12` 个 448×448 tile，>1 tile 时加 thumbnail。返回子图列表 + `patch_num`
- 每个子图经 `vision_tower.image_processor` → `pixel_values` 张量 `[3,448,448]`
- `preprocess_multimodal` 把单个 `<image>` token 展开成 `DEFAULT_IMAGE_TOKEN * patch_num[0]`（= tile 数）个 token，**一个 image feature 对应一个 `<image>` token**（`vita_arch` 的断言）

**步骤 3 — 视觉特征提取 + 拼接**（`prepare_inputs_labels_for_multimodal`，`vita_arch.py:333-562`）：
- `encode_images(images)`（`:160-164`）：`vision_tower(images)` → InternViT 特征 → `mm_projector(...)` → 投影特征。视觉塔冻结，projector 训练
- 对 `input_ids` 里每个 `IMAGE_TOKEN_INDEX(-200)`，切出对应文本 embedding 切片，在该位置拼接图像特征（`:491-510`）
- 图像特征位置的 labels 设为 `IGNORE_INDEX`（`:499-505`）
- 产物是 `inputs_embeds`（不再是 `input_ids`）——文本 token embedding 和视觉特征 embedding 的混合序列

**步骤 4 — 前向**（`vita_qwen2.py:154-195`）：`VITAQwen2ForCausalLM.forward` 当 `inputs_embeds is None` 时调用 `prepare_inputs_labels_for_multimodal`，然后把拼接后的 `inputs_embeds` 喂给 Qwen2 LLM。loss 在回答 token 上算。

### 2.7 DeepSpeed 配置：`script/deepspeed/zero3.json`

```json
"zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true   // 关键：让 ZeRO-3 下能 gather 全 16-bit 权重保存
}
```
- Stage 3（参数 + 梯度 + 优化器状态全分片），**无 CPU offload**（8 卡 × ~30 GB 装得下）
- `bf16: auto` → 因 `--bf16 True` 启用
- optimizer: AdamW，lr/weight_decay 用 `"auto"` 流入 HF 的 `--learning_rate 1e-6`、`--weight_decay 0.0`
- checkpoint 保存：因 `lora_enable=False`，走 `safe_save_model_for_hf_trainer` → ZeRO-3 gather 全权重保存（`stage3_gather_16bit_weights_on_model_save: true`）

### 2.8 SFT 结果

**训练**：本项目第一条正常学习曲线——loss 1.2307 → 0.9751（降 21%），对比 DPO 只降 0.3%–3%。差 7 倍不奇怪：SFT 学"复现这个答案"（模型有能力学），DPO 学"把 A 排在 B 前面"（基座判别力 53%，目标本身模糊）。

**可分性大幅提升**（用同一批 15000 对，SFT 前后由基座打分）：

| 指标 | SFT 前 | SFT 后 | 变化 |
|---|---|---|---|
| accuracy | 52.2% | **58.0%** | +5.8pt |
| logp gap 均值 | +1.84 | **+6.04** | 3.3× |
| **信噪比** | 0.055 | **0.218** | **4.0×** |

**信噪比首次跨过 0.2 门槛**——这正是 SFT 之所以能让后续 DPO 发力的机制证据。

**代价**：通用能力小幅下降（MMStar −1.13、Hallu fAcc −3.76），但是分布相关的标准形状——RLAIF-V 训"描述图片时别编造物体"迁移到 POPE，但不迁移到 HallusionBench 的"抵抗视觉错觉"（见 §6）。

---

## 3. 多模态 DPO 实现细节

### 3.1 数据构建：`tools/make_rlaif_v_data.py`

同源数据但**保留两个回答**。必填列 `("image", "question", "chosen", "rejected")`。

- **丢两侧相同 pair**（`chosen == rejected`）——DPO logit 恒为 0
- 图像处理与 SFT 构建器完全相同（SHA-1 寻址、RGB JPEG quality=95）
- 过滤 `--min-chars 8`、`--max-chars 1200`

**输出记录结构**（`make_rlaif_v_data.py:147-158`）：
```python
{
    "set": "rlaif_v",
    "id": "pref_0001",
    "conversations": [
        {"from": "human", "value": "<image>\n" + question},
        {"from": "gpt",   "value": chosen},          # chosen 放在 gpt 轮
    ],
    "rejected": rejected,                              # 额外字段
    "image": "....jpg",
}
```
与 SFT 格式的唯一区别是多了 `rejected` 字段。`set="rlaif_v"` → `FolderDict["rlaif_v"]` 解析。

### 3.2 训练脚本：三个 DPO 脚本

| 脚本 | 用途 | 关键配置 |
|---|---|---|
| `dpo_smoke_test.sh` | 单卡 wiring 检查（合成 24 对） | lr 5e-6、**lora_dropout=0**（保首步 `-log(0.5)` 精确）、beta 0.1 |
| `dpo_rlaif_v.sh` | 单卡 RLAIF-V 真跑 | lr 5e-6、grad_accum 16（有效 batch 16）、lora_dropout 0.05、beta 0.1、RLAIFV |
| `dpo_rlaif_v_8gpu.sh` | 8 卡大规模跑 | lr **2e-5**、grad_accum 8（有效 batch 64）、WORKERS=0、beta 0.1 |

最终有效方案（第 6 轮）用的是 `dpo_rlaif_v_8gpu.sh` 的变体：6 卡 × 累积 10 = 有效 batch 60、lr 2e-5、250 步、2 小时 26 分。

**关键：所有 DPO 脚本都用 `zero2.json`，不是 zero3**。理由（脚本 header）：policy 和 reference 是同一权重仅切换 adapter，zero3 的参数分片让 `disable_adapter()` 变得微妙；LoRA 的显存 profile（~28 GB/卡）不需要 zero3。

**DPO 的 optimizer 是 HF 的 AdamW，不是 DeepSpeed 的**：`zero2.json` 里**没有 optimizer/scheduler block**，所以 HF Trainer 的 `adamw_torch` + `--lr_scheduler_type cosine` 直接生效；只有 SFT 用的 `zero3.json` 才内嵌了 AdamW + WarmupDecayLR。DPO 侧的调度器是干净的 cosine。

### 3.3 DPO Trainer：`VITADPOTrainer`

继承 `VITATrainer`，**只重写 `compute_loss`**（并改 `training_step` 绕过 VITATrainer 的直通，直接到 HF Trainer）。

构造函数：`dpo_beta=0.1, dpo_label_smoothing=0.0`。

**`_fuse(model, inputs, image_group_size=None)`**（`dpo_trainer.py:38-54`）——**每步只跑一次多模态融合**，而非四次。调用 `prepare_inputs_labels_for_multimodal`，返回 `(inputs_embeds, labels, position_ids, attention_mask)`。设计理由：`VITAQwen2ForCausalLM.forward` 只在 `inputs_embeds is None` 时才调融合函数，所以在这里预先融合、把 `inputs_embeds` 同时传给 policy 和 reference，避免给两者各跑一次 InternViT + 音频编码器。它还返回**重新对齐的 labels**——因为拼接图像/音频 embedding 改变了序列长度，collator 的 labels 已不再对齐融合后的序列。

### 3.4 `compute_loss` 核心逻辑（`dpo_trainer.py:56-132`）

```python
num_pairs = int(inputs.pop("num_pairs").flatten()[0])
image_group_size = ...                          # 来自 collator

inputs_embeds, labels, position_ids, attention_mask = self._fuse(
    model, inputs, image_group_size=image_group_size)

def logps():                                    # policy 和 reference 共用闭包
    out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                position_ids=position_ids, labels=None, use_cache=False, return_dict=True)
    return batch_sequence_logps(out.logits, labels)

policy_logps = logps()

# 参考模型：同一权重，关 adapter，无梯度
unwrapped = self.accelerator.unwrap_model(model)
with torch.no_grad(), unwrapped.disable_adapter():
    ref_logps = logps().detach()

# collator 布局是 [chosen..., rejected...]
policy_chosen, policy_rejected = policy_logps[:num_pairs], policy_logps[num_pairs:]
ref_chosen,   ref_rejected     = ref_logps[:num_pairs],   ref_logps[num_pairs:]
```

**死对检测**（`:92-106`）：被分词不匹配路径 void 掉的 pair（`data_utils:642` 把 `target[:] = IGNORE_INDEX`，只打 warning）在两侧都打 0.0 分，永远贡献 `-log(0.5)`，与正常训练不可区分。trainer 统计 `((policy_chosen == 0.0) & (policy_rejected == 0.0))`，在 1/10/100/每 500 步发递进 warning。

然后调 `dpo_loss(...)`，`loss = losses.mean()`，记录 `rewards/chosen`、`rewards/rejected`、`rewards/margin`、`rewards/accuracy`（`(margin>0).float().mean()`）等指标。

### 3.5 参考模型处理：`disable_adapter()`

**参考 = 同一权重、LoRA adapter 关掉**，用 peft 的 `disable_adapter()` context manager，在 `torch.no_grad()` 下。这精确恢复基座输出，**不占额外显存**（不需要第二个冻结的 7B），这也是 `--lora_enable True` 强制要求的原因。

**配套冻结**（`train_dpo.py:68-77`）：`train.py` 在 LoRA 之后（line 388）才 `initialize_vision_modules`（line 395），而该方法会强制启用 `mm_projector` 的梯度（`vita_arch.py:59-61` 注释"In case it is frozen by LoRA"）。若不处理，`mm_projector` 会继续训练、`disable_adapter()` 不会还原它、参考模型从第 1 步起就悄悄漂离基座——而 loss 还看起来合理。**`train_dpo.py` 显式冻结所有非 LoRA 可训练参数**，让参考严格等于基座，这才让首步 `-log(0.5)` 检查有意义。

**LoRA 的目标范围被显式收窄**：`find_all_linear_names`（`train.py:157-190`）把 `mm_projector`、`vision_tower`、`vision_resampler`、`audio_encoder` 从 LoRA 候选里排除，还跳过纯数字叶名（否则会误配 `layers.0` 这类）。所以 LoRA 只挂在 LLM 的线性层上——这也是 `disable_adapter()` 能精确还原基座、首步 loss 恒等于 0.6931 的前提之一。

### 3.6 DPO Loss：`vita/train/dpo_loss.py`

**`batch_sequence_logps(logits, labels, average=False)`**（`:27-65`）——每序列求和 log-prob：

```python
logits = logits[:, :-1, :]                      # shift: token t 预测 t+1
labels = labels[:, 1:]
mask = labels != IGNORE_INDEX                   # -100
safe_labels = labels.masked_fill(~mask, 0)       # gather 不能取 -100
logps = torch.log_softmax(logits.float(), dim=-1)   # fp32 cast
token_logps = torch.gather(logps, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)
token_logps = token_logps * mask
summed = token_logps.sum(dim=-1)
```

两个代码特定的决策（docstring 说明）：
1. **fp32 log-prob**：`vita_qwen2.py` 的 `custom_forward` 把 `logits = logits.float()` 注释掉了，返回 bf16 logits；在 152k 词表上对 bf16 log-prob 求和会损失 ~2.9 nats。DPO 取"差的差"再乘 beta(≈0.1)，那个误差会淹没信号。内部 `.float()` 把误差降到 ~0.07。
2. **求和而非平均**：平均是已知 DPO 变体但改变了目标；原公式求和。它引入的长度偏置是"DPO 的性质，不是要在这里打的补丁"。`average=False` 默认。

**`dpo_loss(...)`**（`:68-114`）——标准 Rafailov et al. 2023：

```python
policy_delta = policy_chosen_logps - policy_rejected_logps
ref_delta = (ref_chosen_logps - ref_rejected_logps).detach()   # ref 固定基准
logits = policy_delta - ref_delta

losses = -F.logsigmoid(beta * logits)          # label_smoothing=0 时

chosen_rewards   = beta * (policy_chosen_logps  - ref_chosen_logps).detach()
rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
```

- **经典 sigmoid DPO**，`beta=0.1` 默认，**无长度归一化**（求和），**无 IPO/cDPO/SimPO 变体**（`label_smoothing` 实现保守 DPO，但脚本里都是 0.0）
- reward 是隐式奖励 `beta * (policy - ref)`，detach
- **关键不变量**（docstring）：未训练的 adapter 让 policy == reference，所以 `logits = 0`，loss 必然 = `-log(0.5) = 0.6931`。这是"参考模型接对了的最强检查信号"。

### 3.7 多模态融合与图像去重：`image_group_size`

这是 DPO 多模态处理的核心，也是与 GRPO 的关键区别。

**`DPODataCollator`**（`dpo_data.py:113-162`）把 batch 布局成 `[chosen_0..chosen_{B-1}, rejected_0..rejected_{B-1}]`（2B 序列），前向后从中间对半切。它把两半 flat list 交给现有的 `DataCollatorForSupervisedDataset` 做填充和媒体堆叠，设 `batch["num_pairs"] = len(instances)`。

**图像去重逻辑**（`dpo_data.py:158-161`）：两半带字节相同的媒体，所以视觉塔只需看前半。算 `n_tiles = images.shape[0]`，若 `n_tiles % 2 == 0`，设 `batch["image_group_size"] = n_tiles // 2`。完整重复的张量仍过一遍（`vita_arch` 断言一个 image feature 对应一个 `<image>` token），但 `prepare_inputs_labels_for_multimodal` 只编码一半、用 `features.repeat(repeats, ...)` 复制特征——bit-identical，视觉塔成本减半。

**`encode_images_deduped(images, group_size)`**（`vita_arch.py:166-189`）：只编码前 `group_size` 个 tile，`repeats = total // group_size`，若 `repeats == 1` 直接调 `encode_images`。docstring 说明：对"对同一媒体评分多个回答"的目标（DPO 的 chosen/rejected 对、GRPO 的 rollout 组），batch 是 N 份相同图的重复；视觉塔确定性，所以编码一份再 repeat 特征是 bit-identical 的，成本 1/N。只断言 shape 整除（比像素更省）。

### 3.8 chosen/rejected 共享一切（除最后 assistant 轮）

`DPODataset` 包装 `LazySupervisedDataset`（复用 ~1500 行的 SFT 数据管线做图像切片、fbank、prompt 组装）。`_encode(index, use_rejected)`（`dpo_data.py:76-94`）的技巧：深拷贝记录，把**最后一个 gpt 轮的 value** 换成 `record["rejected"]`（当 `use_rejected=True`），临时塞回 `self.inner.list_data_dict[index]`，调用 `self.inner[index]`（未改的 SFT 管线），`finally` 里还原。

**所以 chosen 和 rejected 只在最后 assistant 轮不同**；prompt token、图像 tile、音频字节相同。媒体共享一份（`__getitem__` 只从 chosen view 取一份 image/audio）——对同一媒体评两个回答，复制只会浪费编码器时间和显存。

### 3.9 首步不变量：`loss = 0.6931`

这是数学恒等式不是经验值：未训练 LoRA 让 policy == reference，DPO logit 为零，loss 必然 `-log(0.5) = 0.6931`。**判断参考模型接线是否正确的最强信号，而且免费**。

附带发现：冒烟测试里 `lora_dropout` 必须设 0。dropout 在 `model.train()` 下对 policy 生效，而参考模型关了 adapter 因此没有 dropout——两侧不对称，首步就不等于 0.6931，这条检查失效。（smoke 用 `dropout=0`；真实跑用 `0.05`，检查仍近似成立。）

单元测试 `tools/test_dpo_loss.py` 验证：policy==ref → 精确 `-log(0.5)`；偏好 chosen 降 loss；偏好 rejected 升 loss；beta 大更陡；label smoothing 升高 confident loss；梯度只到 policy 项（reference 无）；fp32 cast 比 bf16-throughout 在 200-token×152064-vocab 序列上好 >10×。

### 3.10 DPO 数据准备工具全景

| 工具 | 偏好来源 |
|---|---|
| `make_dpo_smoke_data.py` | 合成 24 对（4 图 × 3 拒绝类型 × 8） |
| `make_rlaif_v_data.py` | **真实 AI 反馈**——OmniLMM-12B 生成并判定的 chosen/rejected |
| `make_selfsample_data.py` | **on-policy**——VITA 自己采样 K=6（temp 0.9、max_tokens 96），按 token-F1 vs RLAIF-V chosen 参考答案排序，最相似→chosen、最不相似→rejected。判据是词袋相似度，非 judge LLM |
| `probe_preference_separability.py` | 训前 8 分钟诊断——冻结基座打分，报 accuracy/CI/显著性/SNR。判读：~50% 换数据；显著但 SNR<0.2 训得起来但不变好；65%+ 可训 |

**on-policy 自采样和 off-policy RLAIF-V 共用同一条加载路径**：`make_selfsample_data.py` 写出的文件名和记录形状与 `make_rlaif_v_data.py` 完全一样（`rlaif_v_train.json`、`set` 标成 `"selfsample"`），通过同一个 `VITA_RLAIF_DATA_DIR` / `RLAIFV` 配置加载；`dataset_config.py` 同时登记了 `FolderDict["selfsample"]` 指向同一 images 目录。所以第 4 轮换数据只是把环境变量指向新目录，训练脚本一行不改。

---

## 4. ZeRO 显存分析

### 4.1 结论

**在 8×80GB H100 上，ZeRO-1、ZeRO-2、ZeRO-3 显存上都能装下 7B 全参 SFT。脚本头说"zero2 不够"的推理，从纯显存数学看不严谨——但选 ZeRO-3 有其真实的工程理由。**

### 4.2 7B 全参训练的显存构成

Qwen2.5-7B 实际约 7.6B 参数。混合精度 + AdamW 的完整训练状态：

| 组成 | 精度 | 单份大小 | 说明 |
|---|---|---|---|
| 模型权重（前向/反向用） | bf16 | 15.2 GB | "parameters" |
| fp32 master 权重 | fp32 | 30.4 GB | 属于 optimizer states |
| AdamW 一阶动量 exp_avg | fp32 | 30.4 GB | optimizer states |
| AdamW 二阶动量 exp_avg_sq | fp32 | 30.4 GB | optimizer states |
| 梯度 | bf16 | 15.2 GB | 反向产生 |

**合计 ≈ 121.8 GB**（脚本头的 "~98 GB" 是 14+28+56，**没算梯度**——它指的是"单卡分配到 exp_avg_sq 时 OOM 的那个瞬间"，约 98GB；加上梯度和激活峰值实际要到 ~113GB+）。

### 4.3 ZeRO 三阶段各自分片什么

| 阶段 | 分片优化器状态 | 分片梯度 | 分片参数 |
|---|---|---|---|
| ZeRO-1 | ✅ | ❌（每卡完整） | ❌（每卡完整） |
| ZeRO-2 | ✅ | ✅ | ❌（每卡完整） |
| ZeRO-3 | ✅ | ✅ | ✅ |

### 4.4 8 卡分片后每卡显存

| | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|
| bf16 参数（每卡完整副本） | 15.2 | 15.2 | 1.9（分片） |
| bf16 梯度 | 15.2（完整） | 1.9（分片） | 1.9（分片） |
| 优化器状态（fp32 master + 2 moments，分片） | 11.4 | 11.4 | 11.4 |
| **静态合计** | **41.8** | **28.5** | **15.2** |
| + 激活（gradient checkpointing，6200 token 单序列） | ~2-4 | ~2-4 | ~2-4 |
| + 通信/临时 gather 缓冲 | 小 | 小 | ~2-3 |
| **每卡峰值** | **~45 GB** | **~32 GB** | **~18-20 GB** |

**80GB H100 上：三者都远低于 80GB，都能跑。** ZeRO-2 的 ~32GB 离 80GB 还有近 50GB 裕度。

### 4.5 脚本头为什么说"zero2 would NOT be enough"

脚本头原文（`sft_rlaif_v_8gpu.sh:22-23`）：

> zero2 would NOT be enough here -- it shards optimizer state but replicates parameters, unlike the DPO runs where LoRA kept the trainable set tiny.

**这句从纯显存看不成立**：ZeRO-2 复制参数确实让每卡多占 15GB，但 28.5GB + 激活 ≈ 32GB，在 80GB 卡上绰绰有余。脚本头把"参数复制"直接等价于"不够"，跳过了实际加总这一步。

判断：作者在对比 DPO（LoRA+zero2 ≈ 28GB/卡）时做了一个**偏保守的过度推理**——把 zero2 当成"显存高所以危险"，没做精确加总。真正选 ZeRO-3 的合理理由应该是：

| 选 ZeRO-3 的真实理由 | 说明 |
|---|---|
| **安全裕度最大** | 18GB vs 32GB，留出 60GB 给意外峰值、长序列激活、临时缓冲 |
| **可能在更保守的硬件假设下写的** | 若按 40GB A100 设想，ZeRO-2 的 32GB+激活 确实逼近极限，ZeRO-3 才稳 |
| **`stage3_gather_16bit_weights_on_model_save`** | ZeRO-3 下能 gather 完整 16-bit 权重直接保存；不过 ZeRO-2 保存全参其实更简单（参数本就完整），所以这条不是 zero3 的独有优势 |

反过来，**ZeRO-2 其实有一个 ZeRO-3 没有的优点：速度更快**。ZeRO-3 每层前向/反向都要 all-gather 参数，通信开销大；ZeRO-2 参数常驻，前向无额外通信。所以如果显存够，ZeRO-2 是更优选择。这意味着脚本选 ZeRO-3 是**用速度换显存裕度**——一个保守但可接受的权衡，只是脚本头的理由表述不精确。

### 4.6 ZeRO 阶段与是否用 LoRA 无关

脚本头说"DPO 能用 zero2 是因为 LoRA 可训练参数极少"——这个因果也不完全对。**ZeRO 分片的是全部训练状态，不是只分片可训练参数。** DPO 用 LoRA 时，DeepSpeed 仍然为整个 7B 模型分配优化器状态——只不过 LoRA 的梯度只回传到 adapter，**冻结参数不产生优化器状态**。所以 DPO 的 zero2 显存低，真正原因是"只有 LoRA 那 ~50M 参数有优化器状态"，而不是"zero2 对 LoRA 特殊对待"。

**小结**：ZeRO-1/2/3 显存上都支持 7B 全参 SFT（在 80GB 卡上）。ZeRO-3 是最保守、裕度最大的选择；ZeRO-2 更省通信但显存略高；脚本头"zero2 不够"的推理在数字上不成立，但选 ZeRO-3 的实际效果（~18-30GB/卡、稳定）是对的。

---

## 5. mm_projector 的作用

### 5.1 架构（`builder.py:160-168`）

SFT 脚本传 `--mm_projector_type mlp2x_gelu`，对应一个 **2 层 MLP**：

```python
modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]   # 3200 → 3584
for _ in range(1, mlp_depth):                                      # mlp_depth=2
    modules.append(nn.GELU())
    modules.append(nn.Linear(config.hidden_size, config.hidden_size))  # 3584 → 3584
return nn.Sequential(*modules)
```

即：`Linear(3200→3584) → GELU → Linear(3584→3584)`。

- `mm_hidden_size` = InternViT-300M 的输出维度 = **3200**
- `hidden_size` = Qwen2.5-7B 的隐层维度 = **3584**

### 5.2 在管线中的位置（`vita_arch.py:160-164`）

```python
def encode_images(self, images):
    image_features = self.get_model().get_vision_tower()(images)   # InternViT 前向（冻结）
    image_features = self.get_model().mm_projector(image_features) # MLP 投影（可训练）
    return image_features
```

数据流：

```
图像像素 [3,448,448]
  → InternViT-300M（冻结）→ 视觉特征 [num_patches, 3200]
  → mm_projector（可训练）→ 投影特征 [num_patches, 3584]
  → 拼接进文本 token embedding 序列（3584 维）
  → Qwen2.5-7B LLM
```

### 5.3 作用

**mm_projector 是视觉模态和语言模态之间的维度对齐桥梁**。InternViT 输出 3200 维的视觉 patch 特征，但 Qwen2 LLM 期望的输入维度是 3584（等于它的词嵌入维度）。两者维度不匹配，无法直接拼接。mm_projector 把视觉特征线性投影到语言模型的嵌入空间，使视觉特征能像"特殊的视觉 token"一样插入文本序列，被 LLM 当作普通 token 处理。

这是所有 LLaVA 系多模态模型的标准组件——视觉塔负责"看"，LLM 负责"推理"，mm_projector 负责把"看"的结果翻译成 LLM 听得懂的语言。

### 5.4 参数量

- `Linear(3200, 3584)`：3200×3584 + 3584 ≈ **11.47M**
- `Linear(3584, 3584)`：3584×3584 + 3584 ≈ **12.85M**
- 合计 ≈ **24.3M 参数**

相对 7B LLM，mm_projector 只有 24M，约占 0.3%。

---

## 6. 跟着 LLM 全参 SFT 是否影响已有功能

### 6.1 先纠正前提：实际上是"有影响"的

实验记录明确测出了代价。但影响是**小幅、可控、分布相关**的，不是灾难性的。

### 6.2 SFT 训练了什么、冻结了什么

| 组件 | 参数量 | SFT 中的状态 |
|---|---|---|
| Qwen2.5-7B LLM | ~7.6B | **训练**（`freeze_backbone=False`，全参） |
| mm_projector（mlp2x_gelu） | ~24M | **训练**（`freeze_mm_mlp_adapter=False`） |
| InternViT-300M 视觉塔 | ~300M | **冻结**（`unfreeze_vision_tower` 默认 False） |
| 音频编码器 | — | **冻结**（`freeze_audio_encoder=True`） |
| 音频 adapter | — | **冻结**（`freeze_audio_encoder_adapter=True`） |

所以"视觉理解"这条链上，**视觉塔（承载大部分视觉能力）是冻结的，只有 mm_projector（24M 的投影层）和 LLM 一起微调**。

### 6.3 实测代价（EXPERIMENT_LOG §8.5）

| Benchmark | baseline | SFT 后 | Δ |
|---|---|---|---|
| MME 逐题% | 88.50 | 88.12 | **−0.38** |
| MMStar | 59.80 | 58.67 | −1.13 |
| MMBench | 77.79 | 78.48 | +0.70 |
| AI2D | 79.24 | 79.11 | −0.13 |
| POPE 幻觉率 | 10.97% | 10.34% | −0.63（改善，p=0.027） |
| Hallu aAcc | 61.93 | 60.57 | −1.37 |
| **Hallu fAcc** | **37.57** | **33.82** | **−3.76（显著退化）** |

**所以"不会影响之前功能"这个前提不成立**——通用 MCQ 小幅下降，HallusionBench 视觉错觉类显著退化（fAcc −3.76）。

### 6.4 为什么影响是"小幅可控"而非"灾难性遗忘"

四个机制约束了影响范围：

**(1) 视觉塔冻结——保住了底层视觉表征**

InternViT-300M 承载了从像素到视觉 patch 特征的全部"看"的能力。它冻结意味着**视觉表征空间完全不变**。mm_projector 只是在不变的视觉特征和 LLM 之间做线性重映射。底层视觉理解（物体识别、纹理、空间关系）都在冻结的视觉塔里，不会被 2 万条 VQA 数据碰坏。这是最关键的一道保险——如果解冻视觉塔，脚本头说"是最快搞坏 benchmark 的方式"。

**(2) mm_projector 参数量极小（24M），且 lr 极低（1e-6）**

- mm_projector 24M × lr 1e-6 × 625 步：绝对更新量极小。
- lr 1e-6 比上游 finetune 脚本（2e-5）低一个数量级。脚本头明确说："风险是用 2 万条 VQA 答案覆盖掉通用能力，不是欠拟合。"
- 对比：DPO 用 LoRA，lr 2e-5（高 20 倍），但只动 LoRA adapter；SFT 全参但 lr 极低。两者策略相反但都保守。

**(3) 只 1 epoch、625 步**

训练量有限，不足以把 7B 权重大幅拉离原始分布。loss 1.23→0.98（降 21%）说明学到了东西，但 625 步的更新远不到"覆盖"的程度。

**(4) 训练数据与视觉任务同分布**

RLAIF-V 是 VQA 数据（描述图片、回答关于图片的问题），和评测的视觉问答任务（POPE/MMBench/AI2D）分布接近。所以 mm_projector 和 LLM 的微调方向是"更好地做视觉问答"，而不是被拉向无关方向。这也解释了退化的形状——**与 RLAIF-V 分布对齐的能力改善，不对齐的退化**：

| 受益（与 RLAIF-V 对齐） | 受损（不对齐） |
|---|---|
| POPE 物体存在（p=0.027 显著） | HallusionBench 视觉错觉（fAcc −3.76） |
| MMBench 逻辑推理 +4.84 | OCR：5 题全错 |
| MME artwork +7、posters +3 | code_reasoning −2 |

RLAIF-V 训的是"描述图片时别编造物体"——迁移到了 POPE 的物体存在判断，但没迁移到 HallusionBench 的"抵抗视觉错觉"（后者需要对反直觉证据的推理，不是描述的克制）。

### 6.5 为什么不把 mm_projector 也冻结

如果 mm_projector 冻结、只训 LLM：LLM 学到的视觉-语言映射调整无法被投影层吸收。mm_projector 是视觉特征进入 LLM 的**必经门户**——如果它冻结，LLM 必须独自补偿 RLAIF-V 分布带来的全部偏移，效率低，且 LLM 的更新会被迫更大反而更危险。联合训练让 24M 的投影层和 LLM 协同适应，投影层用很小的更新分担一部分分布偏移，LLM 的更新可以更温和。

`initialize_vision_modules`（`vita_arch.py:56-61`）甚至**强制** mm_projector 的 `requires_grad=True`（注释"In case it is frozen by LoRA"）——这是 VITA 代码的设计意图：mm_projector 在 SFT 阶段就该跟着训。

### 6.6 一句话总结

mm_projector 跟着 LLM 一起全参 SFT **确实影响了之前的功能**（HallusionBench 退化 −3.76 是实证），但影响可控，因为：(1) 视觉塔冻结保住底层视觉能力；(2) mm_projector 只 24M 参数 + lr 1e-6 + 625 步，更新极小；(3) 训练数据与视觉任务同分布，退化是分布相关的。这是"有代价的真实改进"，不是"零损失的提升"。

---

## 7. on-policy 还是 off-policy

### 7.1 三层概念必须分清

**① 算法层面：DPO 全程是离线算法。** 整个项目（包括成功的第 6 轮）的 DPO 都是预收集好 (chosen, rejected) 静态数据，`compute_loss` 只做前向打分，**训练中不生成新回答**。和它对照的是 GRPO——GRPO 在训练里调 `Qwen2ForCausalLM.generate` 做 `_rollout`，那才是算法意义的 on-policy。

**② 数据起源层面："on-policy 偏好数据"指被排序的回答是不是当前策略自己生成的。** 这是 DPO 文献里 "on-policy DPO / iterative DPO" 的标准用法：
- 第 1–3 轮：chosen/rejected 由 **OmniLMM-12B** 生成 → off-policy 数据（π_data ≠ π_ref）。
- 第 4 轮：VITA 自己采样 K=6 → **on-policy 数据**（π_data = π_ref = VITA base）。但仍是离线（采一次、训一次，不是 online/iterative）。

**③ 关键澄清：成功的第 6 轮根本不是 on-policy 数据。** §9.1 白纸黑字："与 C 轮**完全相同**（15000 对、有效 batch 60、lr 2e-5）…唯一的变量是起点"。也就是说第 6 轮用的是和第 3 轮**同一批 RLAIF-V 数据**——chosen/rejected 仍是 OmniLMM-12B 写的，**off-policy**。第 6 轮和第 3 轮唯一的区别是**起点从原始基座换成 SFT checkpoint**。

### 7.2 各轮对照

| 轮 | 数据起源 | 起点 | 结果 |
|---|---|---|---|
| 1–3 | off-policy（OmniLMM） | 原始基座 | ❌ 全在噪声内 |
| **4** | **on-policy（VITA 自采样）** | 原始基座 | ⚠️ 方差降 26%，SNR 0.141 仍 <0.2，**没走通** |
| 5 | 不需要偏好（只用 chosen） | 原始基座 | ⚠️ 可分性 53.6%→58.0% |
| **6** | **off-policy（同第 3 轮）** | **SFT 后** | ✅ POPE −2.15pt |

**on-policy 那条路（第 4 轮）是个没走通的岔路**——它解决了方差（logp gap 标准差 35.87→26.72，−26%），但判据噪声让 accuracy 只到 54.8%。真正让 DPO 发力的是 **SFT（第 5 轮）+ off-policy DPO（第 6 轮）**，不是 on-policy 数据。

**一句话：这个项目的成功配方是"SFT 做分布对齐 + off-policy DPO"，不是"on-policy DPO"。on-policy 那条路试过、没成。**

---

## 8. 先 SFT 再 DPO 的依据

### 8.1 第一层：探针直接量出"分不开"（§6.1）

冻结基座给 chosen/rejected 打分（15000 对那批）：
- accuracy **52.2%**，CI [47.8%, 56.6%]，**p=0.325（和随机不可区分）**
- logp gap 均值 +1.84，标准差 33.51，**信噪比 0.055**

52.2% ≈ 50%——**基座给 chosen 和 rejected 分配的概率几乎一样**。SNR 0.055 意味着每个 batch 的 DPO 梯度方向几乎被噪声主导。这就是前三轮 DPO 失效的根因。

### 8.2 第二层：DPO 训练第 0 步的初始 logp gap 直接印证（§9.2）

这是 DPO **还没开始训**时、参考策略给两侧打的分：

| 起点 | chosen − rejected 初始 logp gap |
|---|---|
| 原始基座（第 3 轮 C） | **−7.84**（给 rejected 打分更高！） |
| SFT 后（第 6 轮 D） | **+4.75** |

原始基座那行是**负的**——基座不仅分不开，还系统性地更偏好 rejected。在这种起点做 DPO，等于让模型"把现在更讨厌的 chosen 排到前面"，梯度方向和模型自身倾向是反的。SFT 把它从 −7.84 拉到 +4.75，gap 从负变正。

### 8.3 第三层：最干净的受控对照——C 轮 vs D 轮（§9.2）

这是整个项目最强的证据。两轮 DPO **数据相同、超参相同、只换起点**：

| 轮 | 起点 | accuracy | margin | 策略漂移 | loss(末5) |
|---|---|---|---|---|---|
| C | 原始基座 | 0.604 | +0.0473 | 1.81% | 0.6724 |
| **D** | **SFT 后** | **0.653** | **+0.1811** | **7.96%** | **0.6366** |

C→D：margin **3.8×**、漂移 **4.4×**、loss 降幅 **2.7×**。**同一批数据、同一套超参，只因为先把模型对齐到这个分布，DPO 训练信号就放大近 4 倍。**

### 8.4 机制

（§8.1）：DPO 梯度正比于"模型当前错得有多系统"。SNR 0.055 时梯度被噪声主导；SFT 把模型推向"能给 chosen 更高概率"的区间，gap 变大、SNR 升到 0.218（4×），信号才从噪声里冒出来。**可分性是模型的属性，不只是数据的属性**——同一批数据 SFT 前 52.2%、SFT 后 58.0%。

### 8.5 SFT 之后有没有直接测模型效果？做不做 DPO 的对比？

**有，而且项目专门留了一列做这个对比。** §1.2 的结果表有独立的 "SFT" 列（SFT 单独）和 "SFT+DPO" 列，§8.5 专门报 SFT 单独的代价，§9.3 把 DPO 的边际贡献拆出来：

| 指标 | baseline | SFT 单独 | SFT+DPO | DPO 边际（SFT→SFT+DPO） |
|---|---|---|---|---|
| POPE 幻觉率 | 10.97% | 10.34% (p=0.027) | **8.82%** | **−1.52pt** |
| MMBench | 77.79 | 78.48 | 79.26 | +0.78 |
| MME 逐题 | 88.50 | 88.12 | 88.37 | +0.25 |
| AI2D | 79.24 | 79.11 | 79.31 | +0.20 |

所以确实做了"做不做 DPO"的对比，分得很细：
- **SFT 单独已显著**：POPE 10.97→10.34（p=0.027，假阳性少 23 例）。SFT 不是"为 DPO 服务的无独立价值预处理"。
- **DPO 在 SFT 之上再显著加一层**：SFT→SFT+DPO，POPE 改对 61 / 改错 23，**p<0.0001**（§9.3）。
- §9.5：四项 MCQ 在 DPO 阶段**首次全部非负**（三正一平），四轮 DPO 里唯一一次。

---

## 9. 数据泄露讨论

### 9.1 SFT 是在全部 20000 样本上做吗？

**是，20000 样本、1 epoch、625 步**（§8.2：batch 32，20000/32=625 步）。shards = {0,1,3,4}。

### 9.2 数据泄露——三层拆开说

#### (1) 对 benchmark：没有泄露

§2.3 明确：RLAIF-V 的图和问题来自 VQA **train** split（VQAv2/OK-VQA/GQA/TextVQA/COCO），六个评测集是**另一批图和题**，不重叠。SFT 记住的是 RLAIF-V 的 chosen，**不是 POPE 的题**——POPE 的图它从没见过。这是所有 before/after 数字的前提。

#### (2) SFT 和 DPO 之间确实有重叠——这是真的

- SFT：shards **{0,1,3,4}**，20000 条 chosen。
- 第 6 轮 DPO：shards **{0,1}**，15000 对（§9.1：和 C 轮完全相同）。

{0,1} ⊂ {0,1,3,4}，所以 **DPO 里的 chosen 文本，就是 SFT 已经训过的 chosen 的子集**。项目**没有显式做 SFT/DPO 的 disjoint split**（§2.3 只管了 benchmark 污染，没管这两阶段之间）。

#### (3) 但这个重叠不会让 DPO "白捡信号"——DPO + 参考策略的设计正好把它抵消了

关键在 DPO loss：

$$L = -\log\sigma\Big(\beta\big[(\log\pi(c)-\log\pi(r)) - \underbrace{(\log\pi_{ref}(c)-\log\pi_{ref}(r))}_{\text{第6轮 } \pi_{ref}=\text{SFT checkpoint}}\big]\Big)$$

第 6 轮 π_ref = SFT checkpoint。SFT 训过 chosen、没训 rejected，所以 π_ref 确实给 chosen 高分——ref_chosen − ref_rejected 已经是个大正数。**但这正是被减掉的那一项。** DPO 只奖励策略把 (policy_chosen − policy_rejected) 推得**比参考策略已有的还高**，不是奖励"chosen 概率高"本身。

§9.1 的首步检查直接验证：
> "首步 rewards/chosen = rewards/rejected = **0.0** 精确为零——确认参考模型正确指向了 SFT 权重。"

rewards = β(policy − ref)，第一步 policy==ref（LoRA 未训），两侧 reward 都是 0。**如果 SFT 记忆能让 DPO 白捡，首步 reward 就该是正的；它是精确 0，说明记忆被参考项完全抵消，DPO 从零信号开始、得自己学。**

#### (4) 那 DPO 收益是不是"记住 chosen 所以 benchmark 涨"的假象？有专门反证

- **§9.4"这不只是阈值平移"**：POPE 上若纯靠"一律少答 Yes"，翻转率 2.15% 会在"存在物体"半边损失约 32 题；实测只损失 24 题，而在"不存在物体"半边修好 78 题。**选择性改善**——纯记忆/纯调阈值不会选择性。
- **§9.5**：四项 MCQ（MME/MMStar/MMBench/AI2D）的题**根本不在 RLAIF-V 里**，DPO 阶段它们全非负。记住 RLAIF-V 的 chosen 答案不可能帮它答 MMBench 选择题。
- **§9.3 McNemar p=4.2×10⁻⁵**：在 held-out 的 POPE 上，DPO 边际 61 对 / 23 错。

### 9.3 诚实的方法论保留

更干净的实验应当 **SFT 用 shards {3,4}、DPO 用 shards {0,1}**，彻底 disjoint，把"分布对齐"和"chosen 记忆"两因子完全分开。项目**没做这个切分**——所以严格说，"SFT 让 DPO 训练信号变强"里，一小部分可能来自"π_ref 因见过 chosen 而更偏好 chosen"（让 ref gap 更大、训练更稳），而非全来自"模型真理解了这个分布"。但：
- 首步 reward=0 证明记忆不是免费午餐；
- held-out benchmark + 选择性改善证明收益是真泛化；
- C vs D 的 3.8× margin 放大**主要**来自可分性 52.2%→58.0%（模型属性变化），不只是 ref gap 变大。

**结论：重叠存在、是个方法论瑕疵，但不构成"数据泄露导致结果虚高"——因为泄露的对象（chosen）被参考策略减掉了，而评测集本身是 held-out 的。** 如果要复现且想最严谨，把 SFT/DPO 数据切成 disjoint 是值得补的一个实验（见 §10）。

---

## 10. disjoint 实验设计

### 10.1 目的

隔离两个混淆因子：
- **因子 A：分布对齐**——SFT 让模型对 RLAIF-V 分布建模更好，从而可分性提升、DPO 信号变强。
- **因子 B：chosen 记忆**——SFT 训过的 chosen 文本恰是 DPO 里的 chosen，π_ref 因见过而给 chosen 更高概率，ref gap 变大。

原方案（SFT shards {0,1,3,4} → DPO shards {0,1}）两者混在一起。disjoint 方案让 SFT 和 DPO 数据**零重叠**，因子 B 被消除，因子 A 独立可测。

### 10.2 实验配置

| 阶段 | 数据 | shards | 说明 |
|---|---|---|---|
| **disjoint SFT** | 20000 条 chosen | **{3,4}** | 与 DPO 数据完全不重叠 |
| **disjoint DPO** | 15000 对 | **{0,1}** | 与 SFT 数据完全不重叠 |
| 起点 | disjoint SFT checkpoint | — | 对应原方案第 6 轮 |
| 超参 | 与原方案第 6 轮完全相同 | — | lr 2e-5、有效 batch 60、beta 0.1、250 步 |
| 评测 | 六项 benchmark + McNemar | — | 重点看 POPE 是否仍有显著改善 |

**判读标准**：
- 若 disjoint SFT+DPO 在 POPE 上仍有显著改善（p<0.05），则因子 A（分布对齐）是主因，因子 B（记忆）非必要——**原结论得到加强**。
- 若 disjoint 方案 POPE 改善消失或大幅缩水，则因子 B 贡献了原方案的一部分收益——**需修正原结论的归因**。
- 无论哪种，首步 reward=0 的不变量必须成立（验证参考模型正确指向 disjoint SFT 权重）。

### 10.3 命令（复现）

```bash
# 1. disjoint SFT 数据（只从 shard003/004）
python tools/make_rlaif_v_sft_data.py --parquet shard003.parquet shard004.parquet \
    --out-dir $WEIGHTS_ROOT/rlaif_v_sft_disjoint --limit 20000

# 2. disjoint 全参 SFT（与原方案同超参）
VITA_SFT_DATA_DIR=$WEIGHTS_ROOT/rlaif_v_sft_disjoint WORKERS=0 \
    bash script/train/sft_rlaif_v_8gpu.sh /path/sft_disjoint_out

# 3. disjoint DPO 数据（shard000/001，与 SFT disjoint）
python tools/make_rlaif_v_data.py --parquet shard000.parquet shard001.parquet \
    --out-dir $WEIGHTS_ROOT/rlaif_v_dpo_disjoint --limit 15000

# 4. 从 disjoint SFT 起点跑 DPO
MODEL_PATH=/path/sft_disjoint_out/sft-rlaif-v \
VITA_RLAIF_DATA_DIR=$WEIGHTS_ROOT/rlaif_v_dpo_disjoint \
WORKERS=0 GRAD_ACC=10 LR=2e-5 GPUS=2,3,4,5,6,7 \
    bash script/train/dpo_rlaif_v_8gpu.sh /path/dpo_disjoint_out

# 5. 合并 + 评测 + 对比（与原方案同标尺）
python tools/merge_and_eval.py --base /path/sft_disjoint_out/sft-rlaif-v \
    --adapter /path/dpo_disjoint_out/dpo-rlaif-v-8gpu \
    --out $WEIGHTS_ROOT/VITA-1.5-sft-dpo-disjoint
python tools/compare_pope.py --before .../baseline/vita_qwen2 --after .../disjoint/vita_qwen2
```

### 10.4 环境状态（2026-08-20 更新）

> 2026-08-18 时此处记录过三个硬阻塞（GPU 全占用 / RLAIF-V parquet 未下 /
> 无既有 checkpoint）。截至 2026-08-20 已全部解除：权重、RLAIF-V 分片、
> CLEVR 数据均已就绪，环境 `/data/agent/conda/envs/vita-rl` 可用，
> GRPO 四轮训练已在此环境完成（见
> [`GRPO_DEEP_DIVE.md`](GRPO_DEEP_DIVE.md)、
> [`EXPERIMENT_LOG.md` §14](EXPERIMENT_LOG.md)）。
> disjoint 实验本身仍未执行，保留为可选待办；执行顺序如下。

**执行顺序**：下 shard003/004 → 生成 disjoint SFT 数据 → 跑 SFT → 下 shard000/001 → 生成 disjoint DPO 数据 → 跑 DPO → 合并评测对比。预估耗时（8×H100）：数据下载 ~30min + SFT ~2h + DPO ~2.5h + 评测 ~1.5h ≈ 6.5 小时。

---

## 11. 关键不变量与坑汇总

### 11.1 正确性不变量表（DPO/GRPO 都靠这些）

| 信号 | 期望 | 含义 |
|---|---|---|
| DPO 首步 loss | ≈ 0.6931 | policy==reference；非零 = 参考模型接错 |
| DPO 首步 rewards | chosen=rejected=0 | 参考正确指向起点权重 |
| GRPO 首步 `grpo/kl` | ≈ 0 | policy==reference；非零 = 融合后参考接错或 mm_projector 漏冻 |
| GRPO 首步 `grpo/ratio` | ≈ 1 | old_logps 取自同一前向 |

### 11.2 踩过的坑（影响 SFT/DPO 链路的）

- `/dev/shm` 仅 512MB → `WORKERS=0`（SFT 和 DPO 都受影响）
- SFT checkpoint 的 audio encoder 路径：SFT 产物把冻结编码器记成 config 里的**绝对路径**而非子目录，`AUDIO_ENCODER` 从 `MODEL_PATH` 推导会失败——恰好在"SFT→DPO"链上，已加 fallback
- `merge_and_eval.py` 会拷 base 所有子目录：当 base 是 SFT 输出时里面有 `checkpoint-600/`（优化器状态），16GB 模型产出 109GB 目录，已加 `SKIP_PREFIXES` 过滤
- `conv.get_prompt()` 缺 modality 参数 → 无消息断言，自采样脚本所有 `AssertionError` 的真凶，正确调用 `get_prompt("image")`
- prompt 里 `<image>` 要重复 `p_num` 次：一个 image feature 对一个 `<image>` token，只写一个配多 tile 会失败

### 11.3 判据机制的三种来源（理解整个 RLAIF 工作的关键）

| 路线 | 偏好/奖励来源 | 有无 judge LLM |
|---|---|---|
| DPO 1–3 轮 | RLAIF-V 内置 chosen/rejected（OmniLMM-12B 自产自判） | 无，标签是数据集自带的 |
| DPO 第 4 轮 | VITA 自己采样 K=6，按 token-F1 vs RLAIF-V chosen 排序 | 无，词袋相似度 |
| SFT 第 5 轮 | 不需要偏好，只用 chosen | 无 |
| GRPO | 不用偏好对；奖励函数给 rollout 打分（keyword/length/no_repeat/state_token） | 可选 `JudgeReward`（小模型 1–5 分，默认不用） |

这恰好解释了为什么 SFT 能成而前三轮 DPO 不成：SFT 把"判优劣"这个瓶颈整个跳过了。

---

## 12. 面试问答

> 按被问概率和挂人概率排的。每条答案给要点和出处章节，数字都能在
> [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) 找到原始表格。

### 12.1 原理与概念

**Q1：推导一下 DPO。**
从 RLHF 的 KL 约束奖励最大化出发，最优策略有闭式解
π* ∝ π_ref·exp(r/β)，反解出 r = β·log(π/π_ref) + β·log Z，代入
Bradley-Terry 偏好模型，配分函数 Z 在 chosen/rejected 做差时消掉。
要点：**隐式奖励是 β·log(π_θ/π_ref)**，DPO 是在偏好数据上直接拟合它。

**Q2：基座分不开 chosen/rejected，不正是 DPO 要解决的问题吗？**
（本项目最容易被追问的概念陷阱。）要区分两种"分不开"：
- **一贯地排错**（如 accuracy 20%）：模型表示里存在与偏好强相关的特征
  （只是方向反了），梯度会一致地修它——这种 DPO 完全能治。
- **排序结果是抛硬币**（本项目：52.2%，gap 均值 +1.84 / 标准差 33.5，
  SNR 0.055）：模型的似然几何里这批 pair 的好坏与它能感知的任何特征
  都不相关，一半样本梯度往东、一半往西，期望梯度≈0。此时 DPO 学到的
  是**逐对记忆**——C 轮 margin 真的涨了（+0.047）、漂移 6.7×，但评测
  纹丝不动，就是记忆不泛化的形状。
结论：**DPO 能重排模型已有表示里的偏好，不能在表示里无中生有造特征；
造特征是 SFT/预训练的活。** 实验闭环是 C vs D：同数据同超参只换起点，
margin 差 3.8×、评测收益差 6×（§8）。

**Q3：为什么不只 SFT，还要 DPO？**
四层：① 目标函数——SFT 只最大化 log P(chosen)，**结构上表达不了
"不要做什么"**；幻觉是坏模式问题，负样本信息只在 DPO 损失里存在。
② 信息量——数据是 (x, y_w, y_l) 三元组，SFT 把"y_w 比 y_l 好"这个
比较标注整个扔掉了。③ 实测——POPE 总降幅 2.15pt 里 SFT 占 0.63
（p=0.027 勉强）、DPO 增量 1.52（p<1e-4，2.4 倍），且 DPO 把 SFT
造成的 MCQ 退化拉回全非负。④ 边际——SFT loss 末段已走平
（0.9718→0.9696），模仿通道近枯竭，继续加剂量只会累积分布迁移代价。

**Q4：DPO 与 PPO/RLHF 的取舍？与 IPO/KTO/SimPO 的区别？**
DPO：无 RM、无 rollout、离线、实现简单；代价是受限于离线数据的支撑集
——本项目前三轮撞到的正是这个固有弱点（off-policy 数据），不是实现
bug。IPO 用平方损失防偏好过拟合；KTO 只需单边标注（好/坏），不需成对；
SimPO 去掉参考模型、用长度归一化的平均 log-prob。

**Q5：DPO 训练中 chosen 的绝对概率会降吗？**
会。损失只关心差距，两边概率可以同时降（rejected 降更快即可）。
实测 margin 上涨主要靠 rejected 掉得快，这是已知的 squeezing 现象。
追问"KL 约束下漂移怎么到 7.96%"：漂移按 token 级 argmax 变化率量，
与 KL 不是同一尺度。

### 12.2 实现

**Q6：参考模型用 `disable_adapter()` 的前提和失效场景？**
前提：π_ref = 训练起点 = LoRA 挂载的基座。第 6 轮从 SFT checkpoint
出发时 adapter 挂在 SFT 权重上，关断还原的自动就是 SFT 权重——参考
模型跟着起点走，正是想要的。失效：全参 DPO、多 adapter 叠加、挂载点
≠ 起点。此时必须真存一份副本（或 ZeRO-3 分片 / CPU offload）。

**Q7：首步 −log 0.5 验证什么？验证不了什么？**
恒等式（非近似）：初始化时 policy≡ref（LoRA B 零初始化），两个
log-ratio 都是 0，loss = −log σ(0) = 0.6931，rewards 双零。能抓住：
参考模型指错权重、多模态融合后 labels 错位、chosen/rejected 串位。
**验证不了训练有效性**——loss 永远停在 0.6931 附近可能是"没学到"
（A/B 轮形状）。它是链路检查，不是效果检查。

**Q8：图像去重为什么省 46% 而不是 50%？**
省的是 InternViT 前向的一半，但融合/拼接开销不减；46% 是实测值
（13 图块 53.9ms→29.2ms）。机制：collator 只留一份图像 tensor，
`image_group_size` 告诉融合函数每张图供两条序列共享（§3）。
对比 GRPO：G 个 rollout 的复制发生在融合**后**的 embedding 上，
vision tower 本来就只跑一次，不需要这个机制——共享结构不同。

**Q9：多模态 DPO 比文本 DPO 难在哪？**
融合会拼入图像 embedding 改变序列长度，collator 的 labels 必须在融合
后**重新对齐**——最容易错的一处。`_fuse` 每步只融合一次，四个前向
（policy/ref × chosen/rejected）共享同一份 inputs_embeds（§3）。

**Q10：为什么 SFT 用 ZeRO-3、DPO 用 ZeRO-2？**
全参 7B 单卡要 ~98GB（bf16 权重 14 + fp32 主权重 28 + AdamW 双动量
56），必须 ZeRO-3 分片参数；DPO 是 LoRA，可训练参数极少，优化器状态
可忽略，ZeRO-2 即可（§4 有完整的账）。

### 12.3 实验设计与统计

**Q11：为什么第一步是修评测而不是训练？**
没有 baseline 时训练后的数字无法解读；且数字落在论文附近**同时验证了**
prompt 模板、图像预处理、答案抽取三件事都对。一个数字仅仅存在证明
不了什么。

**Q12：为什么用 McNemar 而不是两比例检验？**
before/after 是同一批题的**配对样本**，McNemar 只看翻转的题（改对 61 /
改错 23），功效远高于把两次当独立样本。噪声带按二项分布算且 n 取
**独立题目数**（MMBench 1292 题而非 4876 行——circular eval 把一题展开
多行，用行数会低估噪声带一倍）。

**Q13：SFT 和 DPO 数据重叠，允许吗？**
三层（§9）：① 对 benchmark 无泄露（train split，硬前提）；② DPO 的
参考项结构性地抵消重叠——π_ref 就是 SFT checkpoint，它对 chosen 的
偏好是被减掉的基准，首步 reward **精确为 0** 证明记忆不是免费午餐；
③ 诚实保留：更干净是 disjoint 切分（§10 已设计未跑），但四项 MCQ 的
题不在 RLAIF-V 里、DPO 阶段全非负，记忆解释不了这个。

**Q14：探针为什么看 SNR 而不只看 accuracy？**
DPO 梯度权重 ∝ σ(−β·logits)，gap 分布以 0 为中心且方差巨大时，期望
梯度≈0 但方差巨大——"训得动但不变好"。SNR = |gap 均值|/标准差直接
量这个；项目判读标准：SNR<0.2 → 别指望评测收益（§6.4 出处在
EXPERIMENT_LOG）。

**Q15：可分性提升的统计口径？**
"SFT 后能显著区分"很硬（58.0%，CI [53.2%, 62.8%] 不跨 50%，p=0.001）；
"提升量本身显著"只是边缘（两比例 z=1.74，p=0.082）。更可信的是 SNR
4×：均值涨 3.3× 和标准差降 17% 共同驱动，两个独立方向同时变好不是
抽样噪声的形状。**主动说这个比被挖出来强。**

### 12.4 追问陷阱

**Q16："SFT 拿了大头，DPO 是不是可有可无？"**
分解表就是答案：SFT −0.63（p=0.027）、DPO 再 −1.52（p<1e-4），DPO
增量是 SFT 的 2.4 倍；且四项 MCQ 里 DPO 贡献三正一平，把 SFT 的退化
拉回。两阶段消费的是数据里不同的信息位。

**Q17："recall 掉了 1.6，是不是阈值平移？"**
定量排除：答 Yes 率 34.0%→32.0%（真值 29.3%，方向对）；若纯粹全局
少答 Yes，修好 78 题的翻转率应在"存在物体"半边损失 ~32 题，实测只损
24 题，全集**净增 54 题**——翻转有选择性，是判别力改善。

**Q18："HallusionBench 掉了 4pt 怎么办？"**
诚实答：没修复且机制上预期修不了——RLAIF-V 训"描述别编造物体"，
迁移到 POPE 的存在判断，但"抵抗视觉错觉"要对反直觉证据推理，不同构
（两个幻觉基准的答 Yes 率朝相反方向动，§8.6 出处在 EXPERIMENT_LOG）。
要修得换数据，不是加算力。

**Q19："为什么不用 TRL/veRL 现成的 DPOTrainer？"**
VITA 的融合（负索引占位符、动态 patch、三编码器）与标准
`forward(input_ids)` 接口不兼容，融合必须发生在 trainer 内部；正确性
由 19 项 CPU 测试 + 首步恒等式兜底。工程判断：改造通用框架的多模态
接口成本 > 自写 300 行 loss + trainer。

**Q20："52.2% 和 53.6% 哪个是可分性？"**
两批数据各测过一次（3000 对批 53.6%、15000 对批 52.2%，n=500 抽样），
都在"与随机不可区分"区间，结论一致。引用时说清对应哪批。

**Q21："重来一遍会先做什么？"**
探针先行：8 分钟的检查能省掉前三轮约 12 GPU 时的无效训练。这是项目
最可迁移的方法论产出——"失败先量数据，别先调参"。GRPO 线的对应物
是奖励的组内方差诊断（见 GRPO_DEEP_DIVE）。

---

## 附：如果只记住三件事

1. **全参 SFT 与 LoRA DPO 是互补的两步**：SFT 用 ZeRO-3 全参（LLM + mm_projector 训练，视觉塔/音频冻结）做分布对齐，lr 1e-6 极低；DPO 用 ZeRO-2 + LoRA（只训 adapter，reference = disable_adapter）做偏好优化，beta 0.1。两者共享同一份数据源（RLAIF-V），但 SFT 只取 chosen、DPO 保留 chosen+rejected。

2. **"先 SFT 再 DPO"的依据是可分性**：基座对这批 pair 可分性 52.2%（接近随机），初始 logp gap −7.84（甚至更偏好 rejected）。SFT 把可分性拉到 58.0%、gap 拉到 +4.75、信噪比 4×。C vs D 受控对照（同数据同超参只换起点）显示 DPO margin 放大 3.8×。

3. **数据泄露是个方法论瑕疵但不是致命问题**：SFT shards {0,1,3,4} 与 DPO shards {0,1} 重叠，但 DPO 的参考策略减掉了 SFT 记忆（首步 reward=0），且评测集本身 held-out。disjoint 实验（§10）待资源到位后执行以彻底隔离两因子。
