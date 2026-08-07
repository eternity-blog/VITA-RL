# 代码导航

> 在 GitHub 网页上读这个项目时的跳转表。
>
> 其他文档里的 `file.py:123` 是纯文本，点不动；这份把最常需要看的位置
> 做成了**可点击链接**，直接跳到 GitHub 上的对应行。
>
> 链接指向 `main` 分支的当前状态，所以代码更新后行号可能偏移几行——
> 但函数名不会变，找不到时用 GitHub 的文件内搜索（按 `/`）找函数名即可。
>
> 配套：[PRIMER.md §12](./PRIMER.md#12-建议的阅读顺序) 是**读的顺序**，
> 这份是**跳转的地址**。

## 目录

- [1. 必读的三个位置](#1-必读的三个位置)
- [2. 上游核心代码](#2-上游核心代码)
- [3. 本 fork 的 RL 代码](#3-本-fork-的-rl-代码)
- [4. 本 fork 的修复](#4-本-fork-的修复)
- [5. 工具与脚本](#5-工具与脚本)
- [6. 地雷位置](#6-地雷位置)

---

## 1. 必读的三个位置

只看三个地方的话，看这三个：

| 位置 | 是什么 | 为什么必读 |
|---|---|---|
| [`constants.py`](../../blob/main/vita/constants.py) | 14 行常量 | 负数索引占位符，全库的地基 |
| [`vita_arch.py` 的 `prepare_inputs_labels_for_multimodal`](../../blob/main/vita/model/vita_arch.py#L333) | 约 320 行 | **全库最重要的函数**，模态在这里融合 |
| [`vita_qwen2.py` 的 `custom_forward`](../../blob/main/vita/model/language_model/vita_qwen2.py#L29) | 约 95 行 | 猴子补丁过的前向，loss 在这里算 |

搭配 [ARCHITECTURE.md §3](./ARCHITECTURE.md#3-the-central-idea-negative-index-placeholder-tokens)
和 [§5](./ARCHITECTURE.md#5-prepare_inputs_labels_for_multimodal-the-heart-of-the-model) 读。

## 2. 上游核心代码

### 模型

| 跳转 | 说明 |
|---|---|
| [`vita_arch.py`](../../blob/main/vita/model/vita_arch.py) | `VITAMetaModel` / `VITAMetaForCausalLM` |
| [└ `encode_images`](../../blob/main/vita/model/vita_arch.py#L160) | 视觉编码唯一入口 |
| [└ `prepare_inputs_labels_for_multimodal`](../../blob/main/vita/model/vita_arch.py#L333) | ★ 模态融合 |
| [└ 图像特征计数断言](../../blob/main/vita/model/vita_arch.py#L429) | DPO/GRPO 必须满足的约束 |
| [└ `return None, ...`](../../blob/main/vita/model/vita_arch.py#L651) | **`input_ids` 被置 None**，RL 的主要障碍 |
| [`vita_qwen2.py`](../../blob/main/vita/model/language_model/vita_qwen2.py) | VITA-1.5 主模型 |
| [└ 全局猴子补丁](../../blob/main/vita/model/language_model/vita_qwen2.py#L125) | import 即生效，进程内不可逆 |
| [└ `forward`](../../blob/main/vita/model/language_model/vita_qwen2.py#L154) | 注意 `if inputs_embeds is None` 分支 |
| [└ `generate`](../../blob/main/vita/model/language_model/vita_qwen2.py#L197) | `@torch.no_grad()`，且拒绝 `inputs_embeds` |

### 编码器

| 跳转 | 说明 |
|---|---|
| [`internvit_encoder.py`](../../blob/main/vita/model/multimodal_encoder/internvit/internvit_encoder.py) | 视觉塔，289.9M |
| [└ `pixel_shuffle`](../../blob/main/vita/model/multimodal_encoder/internvit/internvit_encoder.py#L41) | 空间换通道，token ÷4 通道 ×4 |
| [`whale/init_model.py`](../../blob/main/vita/model/multimodal_encoder/whale/init_model.py) | 音频编码器，341.4M |
| [└ 下采样 8 倍](../../blob/main/vita/model/multimodal_encoder/whale/init_model.py#L58) | `attn_mask[2::2][2::2][0::2]` |
| [`multimodal_projector/builder.py`](../../blob/main/vita/model/multimodal_projector/builder.py) | `mlp2x_gelu` 等 |

### 数据与训练

| 跳转 | 说明 |
|---|---|
| [`data_utils_video_audio_neg_patch.py`](../../blob/main/vita/util/data_utils_video_audio_neg_patch.py) | 当前启用的数据管线 |
| [└ 状态 token 注入](../../blob/main/vita/util/data_utils_video_audio_neg_patch.py#L128) | `☜`/`☞`/`☟` 从这里来 |
| [└ `preprocess_qwen2p5_instruct`](../../blob/main/vita/util/data_utils_video_audio_neg_patch.py#L526) | VITA-1.5 实际走的分支 |
| [└ **静默作废样本**](../../blob/main/vita/util/data_utils_video_audio_neg_patch.py#L642) | ⚠ 真实数据训练的头号风险 |
| [└ `__getitem__`](../../blob/main/vita/util/data_utils_video_audio_neg_patch.py#L882) | 一条样本怎么变成张量 |
| [└ `DataCollatorForSupervisedDataset`](../../blob/main/vita/util/data_utils_video_audio_neg_patch.py#L1390) | 批处理与 padding |
| [└ `dynamic_preprocess`](../../blob/main/vita/util/data_utils_video_audio_neg_patch.py#L1499) | 动态切图 |
| [`train.py`](../../blob/main/vita/train/train.py) | 唯一训练入口 |
| [└ 数据管线选择](../../blob/main/vita/train/train.py#L17) | **改这里换管线**，无命令行参数 |
| [└ `find_all_linear_names`](../../blob/main/vita/train/train.py#L157) | LoRA 目标模块（本 fork 修过） |
| [└ `train()`](../../blob/main/vita/train/train.py#L232) | 三个工厂参数使 RL 复用它 |
| [└ LoRA 应用](../../blob/main/vita/train/train.py#L388) | 注意在 `initialize_vision_modules` 之前 |
| [`vita_trainer.py`](../../blob/main/vita/train/vita_trainer.py) | HF Trainer 子类 |
| [└ `mm_projector_lr` 失效处](../../blob/main/vita/train/vita_trainer.py#L190) | 被注释掉了，参数名与行为不符 |
| [`conversation.py`](../../blob/main/vita/conversation.py) | 9 种对话模板 |
| [`mm_utils.py` 的 `tokenizer_image_audio_token`](../../blob/main/vita/util/mm_utils.py#L73) | 负数 id 在这里替换 |

## 3. 本 fork 的 RL 代码

**这是上游没有的部分。** 建议按此顺序读（由简到繁）：

| 顺序 | 跳转 | 行数 | 读点 |
|---|---|---|---|
| 1 | [`dpo_loss.py`](../../blob/main/vita/train/dpo_loss.py) | 114 | 纯数学，最好的入口 |
| 2 | [`dpo_data.py`](../../blob/main/vita/train/dpo_data.py) | 170 | 怎么复用 SFT 管线产出成对样本 |
| 3 | [`dpo_trainer.py`](../../blob/main/vita/train/dpo_trainer.py) | 139 | `disable_adapter()` 当参考模型 |
| 4 | [`train_dpo.py`](../../blob/main/vita/train/train_dpo.py) | 94 | 入口 + **非 adapter 冻结** |
| 5 | [`rewards.py`](../../blob/main/vita/train/rewards.py) | 222 | 可插拔奖励注册表 |
| 6 | [`grpo_loss.py`](../../blob/main/vita/train/grpo_loss.py) | 135 | 组内归一化 + k3 KL |
| 7 | [`grpo_data.py`](../../blob/main/vita/train/grpo_data.py) | 145 | 为什么**不**复用 SFT 管线 |
| 8 | [`grpo_trainer.py`](../../blob/main/vita/train/grpo_trainer.py) | 228 | 最复杂：rollout → 打分 → 优势 |
| 9 | [`train_grpo.py`](../../blob/main/vita/train/train_grpo.py) | 121 | 入口 |

走读见 [ARCHITECTURE.md §14](./ARCHITECTURE.md#14-the-rl-stack-dpo-and-grpo)。

**几个值得专门看的点**：

| 跳转 | 为什么 |
|---|---|
| [`dpo_loss` 的 `ref_delta` detach](../../blob/main/vita/train/dpo_loss.py#L98) | 单元测试抓到的 bug |
| [`grpo_loss` 的退化组处理](../../blob/main/vita/train/grpo_loss.py#L57) | `std < eps` 时置 0，防 NaN |
| [`grpo_trainer` 绕过 `generate`](../../blob/main/vita/train/grpo_trainer.py#L88) | 直接调 `Qwen2ForCausalLM.generate` |
| [`grpo_trainer` 的 log-prob 重算](../../blob/main/vita/train/grpo_trainer.py#L124) | 复用缓存 prompt embeds |
| [`train_grpo` 的冻结逻辑](../../blob/main/vita/train/train_grpo.py#L78) | 不冻结参考模型会漂移 |

## 4. 本 fork 的修复

对照上游看这三处，能理解修的是什么：

| 修复 | 跳转 | 上游的问题 |
|---|---|---|
| `audios` 可选 | [`vita_arch.py#L434`](../../blob/main/vita/model/vita_arch.py#L434) | `None` 分支不可达，必须塞假音频 |
| LoRA 可用 | [`train.py#L157`](../../blob/main/vita/train/train.py#L157) | 数字叶子名让 peft 命中整个 DecoderLayer |
| `cache_position` | [`vita_qwen2.py#L78`](../../blob/main/vita/model/language_model/vita_qwen2.py#L78) | 与固定的 transformers 4.41.1 不兼容 |
| 图像去重 | [`vita_arch.py#L166`](../../blob/main/vita/model/vita_arch.py#L166) | （新增能力，非修复） |

## 5. 工具与脚本

### CPU 测试（不用 GPU、不用权重，秒级）

| 跳转 | 项数 | 覆盖 |
|---|---|---|
| [`test_dpo_loss.py`](../../blob/main/tools/test_dpo_loss.py) | 19 | `-log(0.5)` 恒等式、梯度流向 |
| [`test_grpo_loss.py`](../../blob/main/tools/test_grpo_loss.py) | 39 | 退化组、KL 非负、clip |
| [`test_rewards.py`](../../blob/main/tools/test_rewards.py) | 44 | 每条规则的边界 |
| [`test_image_dedup.py`](../../blob/main/tools/test_image_dedup.py) | 11 | 逐位相等 |
| [`test_audio_optional.py`](../../blob/main/tools/test_audio_optional.py) | 5 | `audios=None` 路径 |

**读代码时配着测试看**——测试是最精确的行为说明。比如 `-log(0.5)`
那个恒等式，看代码要推导，看测试一眼就懂。

### 其他工具

| 跳转 | 用途 |
|---|---|
| [`inspect_dataset.py`](../../blob/main/tools/inspect_dataset.py) | CPU 上检视数据集，接数据后必跑 |
| [`localize_config.py`](../../blob/main/tools/localize_config.py) | 把 HF repo ID 改写成本地路径 |
| [`make_smoke_data.py`](../../blob/main/tools/make_smoke_data.py) | SFT 合成数据 |
| [`make_dpo_smoke_data.py`](../../blob/main/tools/make_dpo_smoke_data.py) | DPO 偏好对 |
| [`make_grpo_smoke_data.py`](../../blob/main/tools/make_grpo_smoke_data.py) | GRPO prompt |

### 训练脚本

| 跳转 | 配置 |
|---|---|
| [`smoke_test_qwen.sh`](../../blob/main/script/train/smoke_test_qwen.sh) | SFT 全参，8 卡 ZeRO-3 |
| [`smoke_test_lora.sh`](../../blob/main/script/train/smoke_test_lora.sh) | SFT LoRA，单卡 |
| [`dpo_smoke_test.sh`](../../blob/main/script/train/dpo_smoke_test.sh) | DPO，单卡 |
| [`grpo_smoke_test.sh`](../../blob/main/script/train/grpo_smoke_test.sh) | GRPO，单卡 G=8 |

### 真实运行日志

[`logs/`](../../tree/main/logs) 存了四次真实运行的指标行，
可以对照 [BENCHMARKS.md](./BENCHMARKS.md) 的表格看。

## 6. 地雷位置

**读代码时看到这些地方要警觉**：

| 跳转 | 地雷 |
|---|---|
| [`data_utils:642`](../../blob/main/vita/util/data_utils_video_audio_neg_patch.py#L642) | 分词不匹配→整条 label 作废，只打一行 print |
| [`data_utils:335`](../../blob/main/vita/util/data_utils_video_audio_neg_patch.py#L335) | 活跃的 `pdb.set_trace()`（共 9 处） |
| [`conversation.py:139`](../../blob/main/vita/conversation.py#L139) | `get_prompt()` 不幂等 |
| [`vita_qwen2.py:125`](../../blob/main/vita/model/language_model/vita_qwen2.py#L125) | 全局猴子补丁 |
| [`vita_qwen2.py:96`](../../blob/main/vita/model/language_model/vita_qwen2.py#L96) | `logits.float()` 被注释掉了 |
| [`vita_arch.py:59`](../../blob/main/vita/model/vita_arch.py#L59) | 强制解冻 `mm_projector`，破坏参考模型 |
| [`vita_trainer.py:190`](../../blob/main/vita/train/vita_trainer.py#L190) | `mm_projector_lr` 已失效 |
| [`constants.py:14`](../../blob/main/vita/constants.py#L14) | `GLOBAL_WEIGHTS_PATH` 仍是占位符 |
| [`command.sh`](../../blob/main/command.sh) | 不是构建脚本，是作者的命令历史 |

详细说明见 [HANDBOOK.md §6](./HANDBOOK.md#6-地雷区)。

## 附：GitHub 上看代码的两个技巧

**按 `/` 搜索文件内容**，比翻行号快——代码更新后行号会漂，函数名不会。

**按 `t` 快速跳文件**，输入文件名片段即可。

想看某行的修改历史，点行号左侧的 `...` → `View git blame`——
本 fork 的修改都有详细的 commit message 解释为什么改。
