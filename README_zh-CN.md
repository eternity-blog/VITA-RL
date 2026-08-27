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
| 3. 基准复测 | ✅ 已完成 | **MME 2353.5、MMStar 59.8、MMBench 77.8、AI2D 79.2——全部落在论文值 1.2 分以内。** 见 [BENCHMARKS.md §2.6](./docs/04-evaluation/BENCHMARKS.md) |
| 4. 真实数据训练 | ✅ 已完成 | 3000 对 RLAIF-V 偏好对，LoRA DPO 一个 epoch，首步 loss 精确命中 `-log(0.5)` |
| 5. 增加 RL | ✅ 已完成 | DPO：SFT→DPO 使 POPE 幻觉率 10.97% → 8.82%（McNemar p<1e-4）——见 [EXPERIMENT_LOG.md](./docs/03-experiments/EXPERIMENT_LOG.md)。GRPO：多模态扩展在 CLEVR 计数 + 可验证奖励上训练，**400 步将 held-out 准确率 44.6% → 77.4%**（win rate 0.977），通用基准零退化，并以配平 SFT 对照与 OOD 实验界定方法边界——见 [GRPO_DEEP_DIVE.md](./docs/03-experiments/GRPO_DEEP_DIVE.md) |

**想看 3–5 阶段的完整故事，读 [EXPERIMENT_LOG.md](./docs/03-experiments/EXPERIMENT_LOG.md)**：
RLAIF-V 上三轮 DPO 都没把基准移出噪声带，且原因是量出来的而不是猜的——
基座对这些偏好对的可分性只有 53.6%（95% CI [50.3%, 56.8%]，n=900），
信噪比 0.055–0.11。把有效 batch 从 16 提到 63 后 DPO 真正学起来了
（loss 降破 0.69、reward margin 18 倍），基准仍然不动——POPE 的 5127 个
答案里只翻转了 24 个，12 个改对对 12 个改错。
`tools/probe_preference_separability.py` 八分钟就能预判这一切。
终局方案是 SFT→DPO（POPE 幻觉率 10.97% → 8.82%）。

**GRPO 这条线的完整故事在 [GRPO_DEEP_DIVE.md](./docs/03-experiments/GRPO_DEEP_DIVE.md)**：
同一个教训换了副面孔。RLAIF-V + 代理奖励（keyword 重叠、LLM judge）两轮，
KL 涨了 6 倍而内容奖励纹丝不动——开放式描述里 8 个 rollout 的代理分差
主要是风格运气，组归一化的 advantage 在给噪声排序。换成可验证奖励
（CLEVR 计数，二值精确匹配）后曲线立刻起飞：400 步 held-out 准确率
44.6% → 77.4%。后续对照实验诚实地画出了边界：通用基准零退化、OOD 迁移
到 SuperCLEVR（+17pt）、配平数据预算的 SFT 对照以 1/7.5 的成本分布内
追平且 OOD 反超；阶段二对照（同一 SFT 起点、同批新 prompt、续 SFT vs
接 GRPO）钉死任务天花板 ~77–78%。落出来的实证法则：**SFT loss 还在降
就 SFT；loss 见底且残错 pass@G > pass@1 才轮到 GRPO**。

**第一次接触这个代码库？** 先看 [PRIMER.md](./docs/00-background/PRIMER.md)——读懂其余文档所需的
前置知识：负数索引占位符机制、实测的 token 预算、三个编码器，以及最费时间的坑。
它的[建议阅读顺序](./docs/00-background/PRIMER.md#12-建议的阅读顺序)给出了一条四阶段的代码通读路径，
标注了每段耗时和是否需要 GPU。

### 文档地图（按管线组织）

`docs/` 按"过一遍项目"的顺序编号：背景 → 环境 → 数据 → 训练 → 评测 → 复习。

| 阶段 | 目录 | 里面有什么 |
|---|---|---|
| 0 · 背景 | [docs/00-background/](./docs/00-background/) | [PRIMER](./docs/00-background/PRIMER.md)（**最先看**——其余文档的前置知识）· [ARCHITECTURE](./docs/00-background/ARCHITECTURE_zh-CN.md)（[EN](./docs/00-background/ARCHITECTURE.md)）——某段代码为什么这么写 · [CODEMAP](./docs/00-background/CODEMAP.md)——GitHub 上读代码的可点击跳转表 |
| 1 · 环境 | [docs/01-setup/](./docs/01-setup/) | [ENVIRONMENT](./docs/01-setup/ENVIRONMENT.md)（**conda 重建 + 全部权重/数据/评测集下载链接**）· [REPRODUCE](./docs/01-setup/REPRODUCE_zh-CN.md)（[EN](./docs/01-setup/REPRODUCE.md)）——安装顺序与每个 pin 的原因 · [HANDBOOK](./docs/01-setup/HANDBOOK.md)——命令速查、地雷区、故障排查 · [MIGRATION](./docs/01-setup/MIGRATION_zh-CN.md)（[EN](./docs/01-setup/MIGRATION.md)）——换机器 |
| 2 · 数据 | [docs/02-data/](./docs/02-data/) | [DATASETS](./docs/02-data/DATASETS.md)——论文用了什么、哪些能下载、本 fork 实际训练用的数据（§3.3） |
| 3 · 训练与实验 | [docs/03-experiments/](./docs/03-experiments/) | [EXPERIMENT_LOG](./docs/03-experiments/EXPERIMENT_LOG.md)（**DPO 线全程** + GRPO 摘要）· [SFT_DPO_DEEP_DIVE](./docs/03-experiments/SFT_DPO_DEEP_DIVE.md)（机制、显存推算、21 问面试问答）· [GRPO_DEEP_DIVE](./docs/03-experiments/GRPO_DEEP_DIVE.md)（**GRPO 线全程**：数学、奖励设计、指标手册、六轮训练与对照、DAPO/GSPO/vLLM、面试问答） |
| 4 · 评测 | [docs/04-evaluation/](./docs/04-evaluation/) | [BENCHMARKS](./docs/04-evaluation/BENCHMARKS.md)——实测数字、噪声带、耗时与显存 |
| 5 · 复习 | [docs/05-review/](./docs/05-review/) | [KNOWLEDGE](./docs/05-review/KNOWLEDGE.md)（**面试复习总索引**：每个知识点一句话核心 + 精确出处）· [PROJECT_SUMMARY](./docs/05-review/PROJECT_SUMMARY.md)——一页总结 + 阅读路线 |

从已回收开发机上抢救下来的原始实验产物（各轮训练日志、逐 step trainer
state、评测原始输出）在 [artifacts/](./artifacts/README.md)——上面文档里的
每个数字都能在那里找到原始文件。GRPO 时代的 5 个 LoRA adapter 托管在
HF [lee31221/VITA-RL](https://huggingface.co/lee31221/VITA-RL)，用
`tools/merge_and_eval.py` 合并进基座即可精确复原对应轮次的评测模型。
DPO 时代的权重随更早一台开发机丢失，完整记录与复现路径见
[EXPERIMENT_LOG.md §13.3](./docs/03-experiments/EXPERIMENT_LOG.md)。

两处值得专门一读的走读：[ARCHITECTURE_zh-CN.md
§5](./docs/00-background/ARCHITECTURE_zh-CN.md#5-prepare_inputs_labels_for_multimodal模型的心脏)
拆解了让这个模型成立的那个函数，[§14](./docs/00-background/ARCHITECTURE_zh-CN.md#14-rl-栈dpo-与-grpo)
走读本 fork 新增的 RL 栈。

完整日志见 [REPRODUCE_zh-CN.md](./docs/01-setup/REPRODUCE_zh-CN.md)：可用的依赖组合、必须的代码修复，以及如何运行训练 smoke test。代码库和模型的实际运作方式见 [ARCHITECTURE_zh-CN.md](./docs/00-background/ARCHITECTURE_zh-CN.md)——模态融合机制、三个编码器、推理与训练路径，以及 RL 栈（DPO + GRPO）如何接入。训练数据调研见 [DATASETS.md](./docs/02-data/DATASETS.md)：论文用了什么、截至 2026 年 8 月哪些还能下载、以及本 fork RL 实际用的数据（§3.3）。

### 如何复现

上游 README 的安装与快速开始说明**并非在所有机器上都能照做跑通**（原因见 [REPRODUCE_zh-CN.md](./docs/01-setup/REPRODUCE_zh-CN.md)）。请从这里开始：

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

在全新机器上重建？见 [MIGRATION_zh-CN.md](./docs/01-setup/MIGRATION_zh-CN.md)——git 里只有代码（约 11 MB），权重和 conda 环境需要重新获取。

### 相对上游的改动

- 增加了 `.gitignore`（上游没有），覆盖训练产物、模型权重与密钥。
- 重写本 README，明确归属上游项目并说明本 fork 的目标。
- **修复了固定版本 `transformers==4.41.1` 下的 `cache_position` 问题** —— 上游的 `vita_qwen2.py` 在其自身 `requirements.txt` 所固定的版本上根本无法生成。见 [REPRODUCE_zh-CN.md](./docs/01-setup/REPRODUCE_zh-CN.md#必须的代码修复)。
- **补上了缺失的 `DataConfig` key**（`Pretrain_video0`、`Pretrain_audio`）—— 多个上游训练脚本会传这两个值，但它们从未被定义。
- **把 `prepare_inputs_labels_for_multimodal` 的 `audios` 改为可选** —— `None` 分支写了但不可达，导致所有纯文本／纯图像前向都必须塞一个假波形，白跑 341M 参数的音频编码器。见 [ARCHITECTURE_zh-CN.md](./docs/00-background/ARCHITECTURE_zh-CN.md#12-已知缺陷与粗糙之处)。
- **在 DPO 之上新增 GRPO**：回答由策略自己采样，奖励在训练中由可插拔的奖励函数实时计算，用组内归一化代替 critic。包含 `vita/train/{rewards,grpo_loss,grpo_data,grpo_trainer}.py`、`train_grpo.py`，以及 `tools/test_grpo_loss.py`（39 项）和 `tools/test_rewards.py`（44 项）。已从纯文本扩展到**图像+文本**（视觉特征融合进 prompt embedding 一次、G 个 rollout 共享），支持 PPO 式样本复用（`--grpo_num_iterations`）与可验证奖励（`answer` 精确匹配 + 分级 `format`），并在 CLEVR 计数上完成真实训练：`tools/make_clevr_grpo_data.py`、`script/train/grpo_clevr.sh`、`tools/eval_grpo_heldout.py`——400 步 held-out 准确率 44.6% → 77.4%，通用基准零退化。随后用对照实验界定边界：配平 SFT 对照臂（`tools/make_clevr_sft_data.py`、`script/train/sft_clevr.sh`）、SuperCLEVR OOD 评测（`tools/make_superclevr_eval_data.py`）、阶段二续训对照（`tools/make_clevr_stage2_data.py`），钉死任务天花板 ~77–78%。见 [GRPO_DEEP_DIVE.md](./docs/03-experiments/GRPO_DEEP_DIVE.md) 与 [HANDBOOK.md §9](./docs/01-setup/HANDBOOK.md#9-grpo组相对策略优化)。
- **新增离线 DPO**，这是本代码库第一个 RL 系目标函数（上游只有 SFT）。包含 `vita/train/dpo_{loss,data,trainer}.py`、`train_dpo.py`、`tools/test_dpo_loss.py`（19 项 CPU 测试）和 `script/train/dpo_smoke_test.sh`。参考模型用同一份权重关掉 LoRA adapter 实现，额外显存为 0。见 [HANDBOOK.md §8](./docs/01-setup/HANDBOOK.md#8-dpo离线偏好优化)。
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

## 📚 上游（VITA-1.5）资源

本 README 只讲 fork 本身。上游模型介绍、论文成绩、官方训练配方、推理
快速开始、网页 Demo 与基准评测教程，见
[上游 README](https://github.com/VITA-MLLM/VITA#readme) 与
[VITA-1.5 论文](https://arxiv.org/pdf/2501.01957)
（[ModelScope Demo](https://modelscope.cn/studios/modelscope/VITA1.5_demo)）。
本 fork 自己的实测数字在
[BENCHMARKS.md](./docs/04-evaluation/BENCHMARKS.md)、
[EXPERIMENT_LOG.md](./docs/03-experiments/EXPERIMENT_LOG.md) 与
[GRPO_DEEP_DIVE.md](./docs/03-experiments/GRPO_DEEP_DIVE.md)。

## 🛠 复现注意事项

*本节是本 fork 特有的内容，不属于上游 README。*

上游代码是从作者内部集群原样发布的。以下几点必须先适配，否则什么都跑不起来——这些都不是 bug，只是从未参数化的环境相关取值：

1. **硬编码的绝对路径。** `script/train/` 下每个脚本都引用 `/mnt/cfs/lhj/...`、`/mnt/cfs2/lhj/...` 或 `/mnt/shared/data1/lhj/...` 作为模型权重和输出路径。[`vita/constants.py`](./vita/constants.py) 中的 `GLOBAL_WEIGHTS_PATH` 仍是占位符 `/path/to/model_weights`。

2. **硬编码的多机设置。** `*_nodes.sh` 脚本把 `INDEX`（节点序号）和 `MASTER_ADDR` 钉死在作者的集群上，例如 `finetuneTaskNeg_qwen_nodes.sh` 中的 `INDEX=3` 和 `MASTER_ADDR="10.206.0.199"`。每个节点需要不同的 `INDEX`。NCCL 变量（`NCCL_SOCKET_IFNAME=eth0`、`NCCL_IB_GID_INDEX=3`）也假定了特定的互联环境。

3. **空的数据集注册表。** [`vita/config/dataset_config.py`](./vita/config/dataset_config.py) 发布时 `AudioFolder`、`FolderDict` 和 `chat_path` 全是空字符串。此外，[`vita/config/__init__.py`](./vita/config/__init__.py) 中的 `DataConfig` 只定义了 `Pretrain_video` 这个 key，而多个脚本传的是 `--dataset_use Pretrain_video0` 或 `Pretrain_audio`；这些 key 必须补上，否则运行会因 `KeyError` 失败。

4. **数据管线在源码中选择，而非命令行。** `train.py` 通过文件顶部一组被注释掉的 import 行，从七个 `data_utils_*` 变体中选一个。默认值（`..._neg_patch`）与文档所述的续训配方一致。

5. **依赖版本固定且偏旧。** `torch==2.3.1` 和 `transformers==4.41.1`。`vita/model/language_model/vita_qwen2.py` 对 `Qwen2ForCausalLM.forward` 打了猴子补丁，这使其与该 `transformers` 版本紧密耦合——升级极可能导致崩坏。

6. **上游的 `command.sh` 已从本 fork 删除。** 它是原作者的命令备忘（不是构建脚本），引用的文件早已不存在；如需查看可在 git 历史中找到。

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


## 👍 致谢

首先也最重要的是，本仓库完全衍生自 [**VITA-MLLM/VITA**](https://github.com/VITA-MLLM/VITA) —— 感谢 VITA 团队开源他们的工作。

VITA 本身参考了以下优秀工作构建：[LLaVA-1.5](https://github.com/haotian-liu/LLaVA)、[Bunny](https://github.com/BAAI-DCAI/Bunny)、[ChatUnivi](https://github.com/PKU-YuanGroup/Chat-UniVi)、[InternVL](https://github.com/OpenGVLab/InternVL)、[InternViT](https://huggingface.co/OpenGVLab/InternViT-300M-448px)、[Qwen-2.5](https://github.com/QwenLM/Qwen2.5)、[VLMEvalkit](https://github.com/open-compass/VLMEvalKit) 和 [Mixtral 8*7B](https://mistral.ai/news/mixtral-of-experts/)。
感谢！
