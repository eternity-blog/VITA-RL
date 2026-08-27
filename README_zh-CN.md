# VITA-RL：为 VITA-1.5 引入强化学习

> [!IMPORTANT]
> **本仓库不是 VITA 官方仓库。**
>
> 本项目是 [**VITA-MLLM/VITA**](https://github.com/VITA-MLLM/VITA)（VITA-1.5：*Towards GPT-4o Level Real-Time Vision and Speech Interaction*）的 fork 与扩展。
>
> - **上游仓库**：https://github.com/VITA-MLLM/VITA
> - **基线 commit**：[`35d064a`](https://github.com/VITA-MLLM/VITA/commit/35d064a6542a5d812136fcd66fa93d9beb27b03c)（2025-03-28）
> - **上游论文**：[VITA-1.5 (arXiv:2501.01957)](https://arxiv.org/pdf/2501.01957)
>
> 下文描述的所有模型架构、训练配方、基准数据和预训练权重，均为原 **VITA 团队（腾讯优图实验室等）** 的成果。本仓库对此不主张任何权利。请引用[原始论文](#️-引用)，并遵守[原始许可协议](./License.txt)——该协议限制仅可用于**学术、研究与教育目的**。

> 语言：[English](./README.md) | **中文**

<p align="center">
    <img src="./asset/vita_newlog.jpg" width="100%" height="100%">
</p>

<font size=7><div align='center' > [[📖 VITA-1.5 论文](https://arxiv.org/pdf/2501.01957)] [[🏠 上游仓库](https://github.com/VITA-MLLM/VITA)] [[🤖 基础 Demo](https://modelscope.cn/studios/modelscope/VITA1.5_demo)] [[🍎 VITA-1.0](https://vita-home.github.io/)]</div></font>

---

## 🎯 关于本 Fork

本仓库的目标是**端到端复现 VITA-1.5**，然后**为其增加一个强化学习阶段**——这是上游未提供的（原代码库只有监督微调）。

> **范围说明。** 本 fork 的 RL 工作只针对**文本 + 图片/视频**模态。
> VITA-1.5 自带的音频编码器全程作为冻结组件保留（推理仍支持音频查询），
> 但本仓库不做任何音频训练或音频 RL。

### 路线图

| 阶段 | 状态 | 说明 |
|---|---|---|
| 1. 复现推理 | ✅ 已完成 | 文本、音频、噪声音频三种查询均可在已发布的 VITA-1.5 checkpoint 上运行 |
| 2. 验证训练链路 | ✅ 已完成 | 在 8×H800 上用合成数据端到端跑通；checkpoint 可保存并重新加载 |
| 3. 基准复测 | ✅ 已完成 | **MME 2353.5、MMStar 59.8、MMBench 77.8、AI2D 79.2——全部落在论文值 1.2 分以内。** 见 [BENCHMARKS.md §2.6](./BENCHMARKS.md) |
| 4. 真实数据训练 | ✅ 已完成 | 3000 对 RLAIF-V 偏好对，LoRA DPO 一个 epoch，首步 loss 精确命中 `-log(0.5)` |
| 5. 增加 RL | ✅ 已完成 | DPO：SFT→DPO 使 POPE 幻觉率 10.97% → 8.82%（McNemar p<1e-4）——见 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md)。GRPO：多模态扩展在 CLEVR 计数 + 可验证奖励上训练，**400 步将 held-out 准确率 44.6% → 77.4%**（win rate 0.977），通用基准零退化，并以配平 SFT 对照与 OOD 实验界定方法边界——见 [GRPO_DEEP_DIVE.md](./GRPO_DEEP_DIVE.md) |

**想看 3–5 阶段的完整故事，读 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md)**：
RLAIF-V 上三轮 DPO 都没把基准移出噪声带，且原因是量出来的而不是猜的——
基座对这些偏好对的可分性只有 53.6%（95% CI [50.3%, 56.8%]，n=900），
信噪比 0.055–0.11。把有效 batch 从 16 提到 63 后 DPO 真正学起来了
（loss 降破 0.69、reward margin 18 倍），基准仍然不动——POPE 的 5127 个
答案里只翻转了 24 个，12 个改对对 12 个改错。
`tools/probe_preference_separability.py` 八分钟就能预判这一切。
终局方案是 SFT→DPO（POPE 幻觉率 10.97% → 8.82%）。

**GRPO 这条线的完整故事在 [GRPO_DEEP_DIVE.md](./GRPO_DEEP_DIVE.md)**：
同一个教训换了副面孔。RLAIF-V + 代理奖励（keyword 重叠、LLM judge）两轮，
KL 涨了 6 倍而内容奖励纹丝不动——开放式描述里 8 个 rollout 的代理分差
主要是风格运气，组归一化的 advantage 在给噪声排序。换成可验证奖励
（CLEVR 计数，二值精确匹配）后曲线立刻起飞：400 步 held-out 准确率
44.6% → 77.4%。后续对照实验诚实地画出了边界：通用基准零退化、OOD 迁移
到 SuperCLEVR（+17pt）、配平数据预算的 SFT 对照以 1/7.5 的成本分布内
追平且 OOD 反超；阶段二对照（同一 SFT 起点、同批新 prompt、续 SFT vs
接 GRPO）钉死任务天花板 ~77–78%。落出来的实证法则：**SFT loss 还在降
就 SFT；loss 见底且残错 pass@G > pass@1 才轮到 GRPO**。

**第一次接触这个代码库？** 先看 [PRIMER.md](./PRIMER.md)——读懂其余文档所需的
前置知识：负数索引占位符机制、实测的 token 预算、三个编码器，以及最费时间的坑。
它的[建议阅读顺序](./PRIMER.md#12-建议的阅读顺序)给出了一条四阶段的代码通读路径，
标注了每段耗时和是否需要 GPU。

### 什么时候看哪份文档

| 文档 | 什么时候看 | 语言 |
|---|---|---|
| [PRIMER.md](./PRIMER.md) | **最先看。** 其余文档的前置知识 | 中文 |
| [HANDBOOK.md](./HANDBOOK.md) | 动手时：命令、地雷区、故障排查 | 中文 |
| [ARCHITECTURE_zh-CN.md](./ARCHITECTURE_zh-CN.md) | 想弄清某段代码为什么这么写 | 中英双版 |
| [REPRODUCE_zh-CN.md](./REPRODUCE_zh-CN.md) | 装环境时 | 中英双版 |
| [DATASETS.md](./DATASETS.md) | 要接真实数据时 | 中文 |
| [MIGRATION_zh-CN.md](./MIGRATION_zh-CN.md) | 换机器时 | 中英双版 |
| [CODEMAP.md](./CODEMAP.md) | 在 GitHub 上读代码时，直接跳到某个函数 | 中文 |
| [BENCHMARKS.md](./BENCHMARKS.md) | 要实测数字时：耗时、显存、以及判断改动是否等价的可复现 loss | 中文 |
| [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md) | **DPO 实验全程**：设计、每个数字、为什么是这个结果、走过的弯路 | 中文 |
| [SFT_DPO_DEEP_DIVE.md](./SFT_DPO_DEEP_DIVE.md) | SFT + DPO 管线的代码级深读：机制、显存推算、on/off-policy、数据泄露讨论、面试问答（21 问） | 中文 |
| [ENVIRONMENT.md](./ENVIRONMENT.md) | **conda 环境从零重建 + 全部权重/数据集/评测集下载链接**（开发机已回收，这是复现清单） | 中文 |
| [KNOWLEDGE.md](./KNOWLEDGE.md) | **面试复习用知识点总索引**：全项目概念与训练细节一张表——一句话核心 + 精确出处，附必背数字 | 中文 |
| [GRPO_DEEP_DIVE.md](./GRPO_DEEP_DIVE.md) | **GRPO 实验全程**：数学细节、超参、Reward 设计、指标手册、五轮训练与对照记录（代理奖励失败→可验证奖励 +32.8pt→通用回归/OOD/配平 SFT 对照/阶段二天花板）、GRPO 演进（vLLM/DAPO/GSPO）、训练与推理框架选型、面试问答 | 中文 |
| [RESULTS.md](./RESULTS.md) | 同一实验更早期、更窄的记录；已被上面取代 | 中文 |

从已回收开发机上抢救下来的原始实验产物（各轮训练日志、逐 step trainer
state、评测原始输出）在 [artifacts/](./artifacts/README.md)——上面文档里的
每个数字都能在那里找到原始文件。GRPO 时代的 5 个 LoRA adapter 托管在
HF [lee31221/VITA-RL](https://huggingface.co/lee31221/VITA-RL)，用
`tools/merge_and_eval.py` 合并进基座即可精确复原对应轮次的评测模型。

两处值得专门一读的走读：[ARCHITECTURE_zh-CN.md
§5](./ARCHITECTURE_zh-CN.md#5-prepare_inputs_labels_for_multimodal模型的心脏)
拆解了让这个模型成立的那个函数，[§14](./ARCHITECTURE_zh-CN.md#14-rl-栈dpo-与-grpo)
走读本 fork 新增的 RL 栈。

完整日志见 [REPRODUCE_zh-CN.md](./REPRODUCE_zh-CN.md)：可用的依赖组合、必须的代码修复，以及如何运行训练 smoke test。代码库和模型的实际运作方式见 [ARCHITECTURE_zh-CN.md](./ARCHITECTURE_zh-CN.md)——模态融合机制、三个编码器、推理与训练路径，以及 RL 栈（DPO + GRPO）如何接入。训练数据调研见 [DATASETS.md](./DATASETS.md)：论文用了什么、截至 2026 年 8 月哪些还能下载、以及本 fork RL 实际用的数据（§3.3）。

### 如何复现

下文上游的安装与快速开始说明**并非在所有机器上都能照做跑通**（原因见 [REPRODUCE_zh-CN.md](./REPRODUCE_zh-CN.md)）。请从这里开始：

```bash
export VITA_REPO=$(pwd) VITA_WEIGHTS=/path/to/weights   # 需约 25 GB 可用空间
conda create -n vita python=3.10 -y && conda activate vita
# 分步安装依赖：REPRODUCE_zh-CN.md#安装顺序顺序很重要
# 权重下载与配置本地化：REPRODUCE_zh-CN.md#权重
python tools/localize_config.py \
    --model-path "$VITA_WEIGHTS/VITA-1.5" \
    --vision-tower "$VITA_WEIGHTS/InternViT-300M-448px"
```

精确解析出的版本记录在 [`requirements-lock.txt`](./requirements-lock.txt)——请把它当作**已知可用组合的记录**，而非安装路径。

在全新机器上重建？见 [MIGRATION_zh-CN.md](./MIGRATION_zh-CN.md)——git 里只有代码（约 11 MB），权重和 conda 环境需要重新获取。

### 相对上游的改动

- 增加了 `.gitignore`（上游没有），覆盖训练产物、模型权重与密钥。
- 重写本 README，明确归属上游项目并说明本 fork 的目标。
- **修复了固定版本 `transformers==4.41.1` 下的 `cache_position` 问题** —— 上游的 `vita_qwen2.py` 在其自身 `requirements.txt` 所固定的版本上根本无法生成。见 [REPRODUCE_zh-CN.md](./REPRODUCE_zh-CN.md#必须的代码修复)。
- **补上了缺失的 `DataConfig` key**（`Pretrain_video0`、`Pretrain_audio`）—— 多个上游训练脚本会传这两个值，但它们从未被定义。
- **把 `prepare_inputs_labels_for_multimodal` 的 `audios` 改为可选** —— `None` 分支写了但不可达，导致所有纯文本／纯图像前向都必须塞一个假波形，白跑 341M 参数的音频编码器。见 [ARCHITECTURE_zh-CN.md](./ARCHITECTURE_zh-CN.md#12-已知缺陷与粗糙之处)。
- **在 DPO 之上新增 GRPO**：回答由策略自己采样，奖励在训练中由可插拔的奖励函数实时计算，用组内归一化代替 critic。包含 `vita/train/{rewards,grpo_loss,grpo_data,grpo_trainer}.py`、`train_grpo.py`，以及 `tools/test_grpo_loss.py`（39 项）和 `tools/test_rewards.py`（44 项）。已从纯文本扩展到**图像+文本**（视觉特征融合进 prompt embedding 一次、G 个 rollout 共享），支持 PPO 式样本复用（`--grpo_num_iterations`）与可验证奖励（`answer` 精确匹配 + 分级 `format`），并在 CLEVR 计数上完成真实训练：`tools/make_clevr_grpo_data.py`、`script/train/grpo_clevr.sh`、`tools/eval_grpo_heldout.py`——400 步 held-out 准确率 44.6% → 77.4%，通用基准零退化。随后用对照实验界定边界：配平 SFT 对照臂（`tools/make_clevr_sft_data.py`、`script/train/sft_clevr.sh`）、SuperCLEVR OOD 评测（`tools/make_superclevr_eval_data.py`）、阶段二续训对照（`tools/make_clevr_stage2_data.py`），钉死任务天花板 ~77–78%。见 [GRPO_DEEP_DIVE.md](./GRPO_DEEP_DIVE.md) 与 [HANDBOOK.md §9](./HANDBOOK.md#9-grpo组相对策略优化)。
- **新增离线 DPO**，这是本代码库第一个 RL 系目标函数（上游只有 SFT）。包含 `vita/train/dpo_{loss,data,trainer}.py`、`train_dpo.py`、`tools/test_dpo_loss.py`（19 项 CPU 测试）和 `script/train/dpo_smoke_test.sh`。参考模型用同一份权重关掉 LoRA adapter 实现，额外显存为 0。见 [HANDBOOK.md §8](./HANDBOOK.md#8-dpo离线偏好优化)。
- 给 `vita_arch.py` 增加 `encode_images_deduped`：当一个 batch 里多条序列共享同一份媒体时（DPO 的 chosen/rejected 对，以及后续 GRPO 的 rollout 组），视觉编码器只编码一份再复制特征。结果逐位相同（`tools/test_image_dedup.py` 用 `torch.equal` 断言），视觉前向省 44-46%。通过 `image_group_size` 显式开启，SFT 路径不受影响。
- 把 `vita/train/train.py` 的 `train()` 泛化为可接收「额外参数类 / 数据模块工厂 / trainer 工厂」，使 DPO 能复用那约 230 行模型构建逻辑而非复制。不传参数时行为与之前完全一致。
- **让 LoRA 可用。** `find_all_linear_names` 没有排除 `audio_encoder`，而 whale 里有两个 `nn.Linear` 的叶子名是数字 `"0"`；peft 按后缀匹配，于是命中了 `layers.0`——整个 `Qwen2DecoderLayer`——导致 `--lora_enable True` 必然失败。现已排除 `audio_encoder` 并跳过纯数字叶子名。单卡 LoRA 峰值 23.3 GB。
- 新增 `script/train/smoke_test_lora.sh`，仓库中第一个真正跑通 LoRA 路径的脚本。
- 新增 `tools/make_smoke_data.py` 与 `script/train/smoke_test_qwen.sh`，一套用合成数据验证训练链路的 smoke test。
- 新增 `tools/test_audio_optional.py`，上述修复的 CPU 单元测试（编码器打桩，无需权重）。
- 新增 `tools/inspect_dataset.py`，在 CPU 上加载已配置的数据集，报告序列长度、被监督的文本片段、collate 后的张量形状，以及有多少样本的 label 被静默作废——接入数据集后、开 GPU 前先跑它。
- 新增 `tools/localize_config.py`，把 checkpoint 的 `mm_vision_tower` / `mm_audio_encoder` 从 HuggingFace repo ID 改写为本地路径，使加载不需要访问网络。
- 新增 `PRIMER.md`（前置知识，仅中文）、`HANDBOOK.md`（上手手册，仅中文）、`REPRODUCE.md`（操作日志）、`ARCHITECTURE.md`（代码走读）、`DATASETS.md`（训练数据调研，仅中文）、`EXPERIMENT_LOG.md`（DPO 六轮全记录）、`GRPO_DEEP_DIVE.md`（GRPO 五轮训练与对照全记录、深读）、`SFT_DPO_DEEP_DIVE.md`、`PROJECT_SUMMARY.md` 和 `requirements-lock.txt`。其中 REPRODUCE 与 ARCHITECTURE 含中英两版。

后续任何相对上游的偏离都会记录在本节。

> **关于复现的说明。** 上游脚本中含有原作者集群的硬编码绝对路径（`/mnt/cfs/lhj/...`）、硬编码的多机地址，以及一个空的数据集注册表。这些必须先在本地适配，否则任何东西都跑不起来——见[复现注意事项](#-复现注意事项)。

---

<p align="center">
    <img src="./asset/vita_demo.jpg" width="80%" height="80%">
</p>

<font size=7><div align='center' > [[📽 VITA-1.5 Demo 演示 🔥](https://youtu.be/tyi6SVFT5mM?si=fkMQCrwa5fVnmEe7)] </div></font>  
<font size=7><div align='center' > VITA-1.5 同时支持**英文**和**中文**。🌟 </div></font>  
你可以直接在 ModelScope 上体验上游的[基础 Demo](https://modelscope.cn/studios/modelscope/VITA1.5_demo)。实时交互 Demo 需按[说明](#-实时交互-demo)配置。

## 🔥 上游动态

*以下里程碑均来自原 VITA 项目。*

* **`2025.01.17`** 🌟 ModelScope 已支持 VITA-1.5！可在其上试用[基础 Demo](https://modelscope.cn/studios/modelscope/VITA1.5_demo)！
* **`2025.01.06`** 🌟 OpenCompass 的 [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) 已同时支持 VITA-1.5 与 VITA-1.0 模型！
* **`2025.01.06`** 🌟 VITA-1.5 的[技术报告](https://huggingface.co/VITA-MLLM)已发布！
* **`2024.12.20`** 🌟 VITA 团队推出 **VITA-1.5**，一个更强、更实时的版本！
* **`2024.08.12`** 🌟 VITA 团队发布 **VITA-1.0**，首个开源的交互式全模态多模态 LLM！


## 目录 <!-- omit in toc -->

- [VITA-RL：为 VITA-1.5 引入强化学习](#vita-rl为-vita-15-引入强化学习)
  - [🎯 关于本 Fork](#-关于本-fork)
  - [🔥 上游动态](#-上游动态)
  - [👀 VITA-1.5 概览](#-vita-15-概览)
    - [🌟 VITA-1.5 有哪些新变化？](#-vita-15-有哪些新变化)
  - [📈 实验结果](#-实验结果)
  - [🛠 复现注意事项](#-复现注意事项)
  - [⭐ 训练](#-训练)
    - [环境与安装](#环境与安装)
    - [数据准备](#数据准备)
    - [续训](#续训)
  - [📐 推理](#-推理)
    - [快速开始](#快速开始)
    - [Demo](#demo)
      - [📍 基础 Demo](#-基础-demo)
      - [📍 实时交互 Demo](#-实时交互-demo)
  - [📏 在 MLLM 基准上评测](#-在-mllm-基准上评测)
    - [VLMEvalKit](#vlmevalkit)
    - [Video-MME](#video-mme)
  - [✒️ 引用](#️-引用)
  - [📣 声明](#-声明)
  - [📜 相关工作](#-相关工作)
  - [👍 致谢](#-致谢)



## 👀 VITA-1.5 概览

*本节描述上游模型。以下所有结果均由原作者报告。*

2024.08.12，VITA 团队发布了 **VITA-1.0**，**首个开源的交互式全模态 LLM**。2024.12.20，他们发布了 **VITA-1.5**。

### 🌟 VITA-1.5 有哪些新变化？

**VITA-1.5** 带来了一系列改进：

1. **交互延迟大幅降低**。端到端语音交互延迟从约 **4 秒**降至 **1.5 秒**，实现近乎即时的交互，显著改善用户体验。

2. **多模态性能提升**。在 *MME*、*MMBench*、*MathVista* 等多模态基准上的平均性能从 **59.8** 显著提升至 **70.8**。

3. **语音处理能力改进**。语音处理能力提升到新水平，ASR 词错误率（WER，Test Other）从 **18.4** 降至 **7.5**。此外，VITA-1.0 中独立的 TTS 模块被替换为**端到端 TTS 模块**，直接以 LLM 的 embedding 作为输入。

4. **渐进式训练策略**。通过这种方式，加入语音对其他多模态（视觉-语言）性能影响很小。图像理解平均性能仅从 71.3 下降到 70.8。


## 📈 实验结果

*以下所有数字均由上游 VITA 团队在 [VITA-1.5 论文](https://arxiv.org/pdf/2501.01957)中报告。本 fork 自己的实测在别处：复测基线（MME 2353.5、MMStar 59.8、MMBench 77.8、AI2D 79.2——全部落在论文值 1.2 分以内）见 [BENCHMARKS.md §2.6](./BENCHMARKS.md)，DPO 结果见 [EXPERIMENT_LOG.md](./EXPERIMENT_LOG.md)，GRPO 结果见 [GRPO_DEEP_DIVE.md](./GRPO_DEEP_DIVE.md)。*

- **图像与视频理解基准评测。**

<p align="center">
    <img src="./asset/vita_mllm_performance.png" width="100%" height="100%">
</p>

- **VITA-1.5 在 ASR 基准上超越专业语音模型。**

<p align="center">
    <img src="./asset/vita_15_audio_2.jpg" width="96%" height="96%">
</p>

- **加入音频模态对图像与视频理解能力影响很小。**

<p align="center">
    <img src="./asset/vita_15_audio_training.png" width="68%" height="50%">
</p>

## 🛠 复现注意事项

*本节是本 fork 特有的内容，不属于上游 README。*

上游代码是从作者内部集群原样发布的。以下几点必须先适配，否则什么都跑不起来——这些都不是 bug，只是从未参数化的环境相关取值：

1. **硬编码的绝对路径。** `script/train/` 下每个脚本都引用 `/mnt/cfs/lhj/...`、`/mnt/cfs2/lhj/...` 或 `/mnt/shared/data1/lhj/...` 作为模型权重和输出路径。[`vita/constants.py`](./vita/constants.py) 中的 `GLOBAL_WEIGHTS_PATH` 仍是占位符 `/path/to/model_weights`。

2. **硬编码的多机设置。** `*_nodes.sh` 脚本把 `INDEX`（节点序号）和 `MASTER_ADDR` 钉死在作者的集群上，例如 `finetuneTaskNeg_qwen_nodes.sh` 中的 `INDEX=3` 和 `MASTER_ADDR="10.206.0.199"`。每个节点需要不同的 `INDEX`。NCCL 变量（`NCCL_SOCKET_IFNAME=eth0`、`NCCL_IB_GID_INDEX=3`）也假定了特定的互联环境。

3. **空的数据集注册表。** [`vita/config/dataset_config.py`](./vita/config/dataset_config.py) 发布时 `AudioFolder`、`FolderDict` 和 `chat_path` 全是空字符串。此外，[`vita/config/__init__.py`](./vita/config/__init__.py) 中的 `DataConfig` 只定义了 `Pretrain_video` 这个 key，而多个脚本传的是 `--dataset_use Pretrain_video0` 或 `Pretrain_audio`；这些 key 必须补上，否则运行会因 `KeyError` 失败。

4. **数据管线在源码中选择，而非命令行。** `train.py` 通过文件顶部一组被注释掉的 import 行，从七个 `data_utils_*` 变体中选一个。默认值（`..._neg_patch`）与文档所述的续训配方一致。

5. **依赖版本固定且偏旧。** `torch==2.3.1` 和 `transformers==4.41.1`。`vita/model/language_model/vita_qwen2.py` 对 `Qwen2ForCausalLM.forward` 打了猴子补丁，这使其与该 `transformers` 版本紧密耦合——升级极可能导致崩坏。

6. **`command.sh` 不是构建脚本。** 它是原作者的命令备忘，引用了仓库中已不存在的文件。**不要**把它当作入口使用。

## ⭐ 训练

*下面的配方是上游的训练流程，此处照录以便查阅。*

### 环境与安装
```
git clone https://github.com/eternity-blog/VITA-RL
cd VITA-RL
conda create -n vita python=3.10 -y
conda activate vita
pip install --upgrade pip
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

> ⚠️ 上述上游安装步骤在本项目的验证机器上**无法成功**。可用的分步安装方案见 [REPRODUCE_zh-CN.md](./REPRODUCE_zh-CN.md#安装顺序顺序很重要)。

### 数据准备
- 训练数据的 json 示例：
```
[
    ...
    {
        "set": "sharegpt4",
        "id": "000000000164",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\n<audio>\n"
            },
            {
                "from": "gpt",  // 沿用 llava 的设定，"gpt" 仅用于表示这是模型输出的 ground truth
                "value": "This is a well-organized kitchen with a clean, modern aesthetic. The kitchen features a white countertop against a white wall, creating a bright and airy atmosphere. "
            }
        ],
        "image": "coco/images/train2017/000000000164.jpg",
        "audio": [
            "new_value_dict_0717/output_wavs/f61cf238b7872b4903e1fc15dcb5a50c.wav"
        ]
    },
    ...
]
```

- `set` 字段用于在数据加载时定位图像或视频目录。你需要把对应的键值对加到 [./vita/config/dataset_config.py](./vita/config/dataset_config.py) 的 `FolderDict` 中：
```
AudioFolder = ""
FolderDict = {
    #### NaturalCap
    "sharegpt4": "",
}
#### NaturalCap
ShareGPT4V = {"chat_path": ""}
```

- 在 [./vita/config/dataset_config.py](./vita/config/dataset_config.py) 相应字典中设置 `"chat_path"` 的 JSON 路径。
- 在 [./vita/config/dataset_config.py](./vita/config/dataset_config.py) 中设置 `AudioFolder` 的音频目录路径。**注意**：加载器实际拼接的是 `AudioFolder/audio/<文件名>`，因此 `AudioFolder` 应指向 `audio/` 的**父目录**。
- 在 [`./vita/config/__init__.py`](./vita/config/__init__.py) 的 `DataConfig` 中添加数据类：
```
from .dataset_config import *

NaturalCap = [ShareGPT4V]

DataConfig = {
    "Pretrain_video": NaturalCap,
}
```

> ⚠️ **上游未提供训练数据集。** 论文中约 2000 万条 QA 来自约 20 个第三方数据集，另有约 570 万条未发布的合成数据和 11 万小时**内部** ASR 数据。因此**论文级别的训练复现客观上无法完成**。但用你自己的数据做续训完全可行——已发布的 checkpoint 本身就是训练好的。

### 续训
- 下载所需权重（均由上游 VITA 团队发布）：(1) [VITA-1.5 checkpoint](https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main)，(2) [InternViT-300M-448px](https://huggingface.co/OpenGVLab/InternViT-300M-448px)，(3) 阶段 2 音频-语言对齐的[预训练音频编码器](https://huggingface.co/VITA-MLLM/VITA-1.5/tree/main/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning)（参见论文图 3）。**实际上音频编码器就在 VITA-1.5 仓库内部，只需下载两份。**

- 替换 [./script/train/finetuneTaskNeg_qwen_nodes.sh](./script/train/finetuneTaskNeg_qwen_nodes.sh) 中的路径：
```
    ...
    --model_name_or_path VITA1.5_ckpt \
    ...
    --vision_tower InternViT-300M-448px \
    ...
    --audio_encoder audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning \
    ...
```

- 执行以下命令开始训练（把 `OUTPUT_DIR` 设为你自己机器上的路径）：

```
export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTPUT_DIR=/path/to/your/outputs/vita_video_audio
bash script/train/finetuneTaskNeg_qwen_nodes.sh ${OUTPUT_DIR}
```

> 💡 7B 全参数训练在单张 80GB 卡上**放不下**（AdamW 优化器状态需约 84 GB，靠 ZeRO-3 切分）。**至少需要 8 张卡。** 详见 [REPRODUCE_zh-CN.md](./REPRODUCE_zh-CN.md#显存说明)。


## 📐 推理
### 快速开始
- 文本查询
```
CUDA_VISIBLE_DEVICES=2 python video_audio_demo.py \
    --model_path [vita/path] \
    --image_path asset/vita_newlog.jpg \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --question "Describe this images."
```

- 音频查询
```
CUDA_VISIBLE_DEVICES=4 python video_audio_demo.py \
    --model_path [vita/path] \
    --image_path asset/vita_newlog.jpg \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --audio_path asset/q1.wav
```

- 噪声音频查询
```
CUDA_VISIBLE_DEVICES=4 python video_audio_demo.py \
    --model_path [vita/path] \
    --image_path asset/vita_newlog.jpg \
    --model_type qwen2p5_instruct \
    --conv_mode qwen2p5_instruct \
    --audio_path asset/q2.wav
```

> 注：上游 README 在音频示例中写的是 `asset/vita_newlog.png`，但仓库里并无该文件——此处已改为实际存在的 `.jpg`。
>
> 回复开头的 `☜` / `☞` / `☟` 是状态符号，分别表示"回复文本查询"、"回复语音查询"、"负样本（噪声）条件下的回复"。详见 [ARCHITECTURE_zh-CN.md](./ARCHITECTURE_zh-CN.md#6-状态符号拒答机制)。


### Demo

上游使用 [vLLM](https://github.com/vllm-project/vllm) 对模型做了加速。由于 VITA 尚未合入 vLLM，你需要修改 vLLM 的代码使其适配 VITA。

```bash
conda create -n vita_demo python==3.10
conda activate vita_demo
pip install -r web_demo/web_demo_requirements.txt

# 备份一份新的权重文件
cp -rL  VITA_ckpt/ demo_VITA_ckpt/

mv demo_VITA_ckpt/config.json demo_VITA_ckpt/origin_config.json

cd ./web_demo/vllm_tools
cp -rf qwen2p5_model_weight_file/*  ../../demo_VITA_ckpt/
cp -rf vllm_file/*  your_anaconda/envs/vita_demo/lib/python3.10/site-packages/vllm/model_executor/models/
```

#### 📍 基础 Demo

https://github.com/user-attachments/assets/43edd44a-8c8d-43ea-9d2b-beebe909377a

```bash
python -m web_demo.web_ability_demo  demo_VITA_ckpt/
```

#### 📍 实时交互 Demo

运行实时交互 demo 需要做以下准备：

- 确认已执行 [Demo](#demo) 一节中的指令（把文件从 `vllm_tools` 中 `cp` 出来）。

- 准备 VAD（语音活动检测）模块。可下载 [silero_vad.onnx](https://github.com/snakers4/silero-vad/tree/v4.0/files) 和 [silero_vad.jit](https://github.com/snakers4/silero-vad/tree/v4.0/files)，放到 `./web_demo/wakeup_and_vad/resource/` 目录下。

- 为获得更好的实时交互体验，需在 `demo_VITA_ckpt/config.json` 中把 `max_dynamic_patch` 设为 1。运行基础 demo 时可保持默认值 12，以增强模型的视觉能力。

```bash
pip install flask==3.1.0 flask-socketio==5.5.0 cryptography==44.0.0 timm==1.0.12
python -m web_demo.server --model_path demo_VITA_ckpt --ip 0.0.0.0 --port 8081
```


## 📏 在 MLLM 基准上评测
### [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
修改 `VLMEvalKit/vlmeval/config.py` 中 `vita_qwen2` 的模型路径：
```
vita_series = { 
    'vita': partial(VITA, model_path='/path/to/model'),
    'vita_qwen2': partial(VITAQwen2, model_path='/path/to/model'),
}
```

按 [VLMEvalKit 的说明](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md)把 GPT 配置为裁判模型。

如果无法使用 openai api，可以用本地模型作裁判。上游作者发现除 MM-Vet 外，[Qwen1.5-1.8B-Chat](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat) 作裁判的效果与 GPT-4 相当。启动裁判服务：
```
CUDA_VISIBLE_DEVICES=0 lmdeploy serve api_server /path/to/Qwen1.5-1.8B-Chat --server-port 23333
```
然后配置 `VLMEvalKit` 目录下的 `.env` 文件：
```
OPENAI_API_KEY=sk-123456
OPENAI_API_BASE=http://0.0.0.0:23333/v1/chat/completions
LOCAL_LLM=/path/to/Qwen1.5-1.8B-Chat
```
在以下基准上评测：
```
CUDA_VISIBLE_DEVICES=0 python run.py --data MMBench_TEST_EN_V11 MMBench_TEST_CN_V11 MMStar MMMU_DEV_VAL MathVista_MINI HallusionBench AI2D_TEST OCRBench MMVet MME --model vita_qwen2 --verbose
```

### Video-MME

数据准备：下载 [Video-MME 数据集](https://github.com/BradyFU/Video-MME)并抽帧存为图像，以提升 IO 效率。完整的评测命令见[英文版 README](./README.md#video-mme)。

## ✒️ 引用

**本 fork 不产生新的论文成果。** 如果你使用本代码，请引用原始的 VITA 论文——模型与方法的全部功劳归上游作者所有。

```bibtex
@article{fu2025vita,
  title={VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction},
  author={Fu, Chaoyou and Lin, Haojia and Wang, Xiong and Zhang, Yi-Fan and Shen, Yunhang and Liu, Xiaoyu and Li, Yangze and Long, Zuwei and Gao, Heting and Li, Ke and others},
  journal={arXiv preprint arXiv:2501.01957},
  year={2025}
}

@article{fu2024vita,
  title={Vita: Towards open-source interactive omni multimodal llm},
  author={Fu, Chaoyou and Lin, Haojia and Long, Zuwei and Shen, Yunhang and Zhao, Meng and Zhang, Yifan and Dong, Shaoqi and Wang, Xiong and Yin, Di and Ma, Long and others},
  journal={arXiv preprint arXiv:2408.05211},
  year={2024}
}
```


## 📣 声明

**以下声明继承自上游项目，同样适用于本仓库：**

**VITA 在大规模开源语料上训练，其输出具有随机性。VITA 生成的任何内容不代表模型开发者的观点。我们不对因使用、误用和传播 VITA 而引起的任何问题负责，包括但不限于舆论风险和数据安全问题。**

此外：本 fork 是非官方的、仅供研究用途的扩展，未获得原 VITA 作者的认可、关联或支持。代码及上游权重的使用仍受 [`License.txt`](./License.txt) 约束——该协议**仅允许学术、研究与教育用途**，禁止商业或生产用途。


## 📜 相关工作

原作者的上游相关研究：
-  **[VITA-1.0]** [VITA: Towards Open-Source Interactive Omni Multimodal LLM](https://vita-home.github.io/)
-  **[Awesome-MLLM]** [A Survey on Multimodal Large Language Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models)
-  **[MME]** [MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)
-  **[Video-MME]** [Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis](https://github.com/BradyFU/Video-MME) 


## 👍 致谢

首先也最重要的是，本仓库完全衍生自 [**VITA-MLLM/VITA**](https://github.com/VITA-MLLM/VITA) —— 感谢 VITA 团队开源他们的工作。

VITA 本身参考了以下优秀工作构建：[LLaVA-1.5](https://github.com/haotian-liu/LLaVA)、[Bunny](https://github.com/BAAI-DCAI/Bunny)、[ChatUnivi](https://github.com/PKU-YuanGroup/Chat-UniVi)、[InternVL](https://github.com/OpenGVLab/InternVL)、[InternViT](https://huggingface.co/OpenGVLab/InternViT-300M-448px)、[Qwen-2.5](https://github.com/QwenLM/Qwen2.5)、[VLMEvalkit](https://github.com/open-compass/VLMEvalKit) 和 [Mixtral 8*7B](https://mistral.ai/news/mixtral-of-experts/)。
感谢！
