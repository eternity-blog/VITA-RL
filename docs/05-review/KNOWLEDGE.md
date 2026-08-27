# KNOWLEDGE：知识点总索引（面试复习用）

> 本文档是**索引不是正文**：每个知识点给一句话核心结论 + 精确出处。
> 复习时按本文过一遍，答不上来的点击出处精读。
> 配套：面试问答清单在 [GRPO_DEEP_DIVE.md §11](../03-experiments/GRPO_DEEP_DIVE.md)
> 和 [SFT_DPO_DEEP_DIVE.md §12](../03-experiments/SFT_DPO_DEEP_DIVE.md)（21 问）；
> 阅读路线在 [PROJECT_SUMMARY.md](./PROJECT_SUMMARY.md) 末尾。
> 数字溯源：原始训练日志与评测结果在 [artifacts/](../../artifacts/README.md)，
> LoRA adapter 在 HF [lee31221/VITA-RL](https://huggingface.co/lee31221/VITA-RL)；
> 环境重建与全部资源下载链接在 [ENVIRONMENT.md](../01-setup/ENVIRONMENT.md)。

## 目录

- [1. RL 后训练算法谱系](#1-rl-后训练算法谱系)
- [2. DPO 知识点](#2-dpo-知识点)
- [3. GRPO 知识点](#3-grpo-知识点)
- [4. Reward 设计](#4-reward-设计)
- [5. 多模态融合（VITA 特有）](#5-多模态融合vita-特有)
- [6. 训练工程](#6-训练工程)
- [7. 推理与框架选型](#7-推理与框架选型)
- [8. 统计与实验方法论](#8-统计与实验方法论)
- [9. 训练细节速查（每轮配置）](#9-训练细节速查每轮配置)
- [10. 必背数字](#10-必背数字)

---

## 1. RL 后训练算法谱系

| 知识点 | 一句话核心 | 出处 |
|---|---|---|
| RLHF/PPO → DPO → GRPO 的演化逻辑 | PPO 要 4 个模型（policy/ref/RM/critic）；DPO 把 RM+RL 合成一步监督式损失；GRPO 去掉 critic 用组内均值当 baseline | [GRPO_DEEP_DIVE §1](../03-experiments/GRPO_DEEP_DIVE.md) |
| DPO 从 RLHF 目标的推导 | KL 约束奖励最大化的闭式解反解出 r=β·log(π/π_ref)+βlogZ，代入 Bradley-Terry，Z 在做差时消掉 | [SFT_DPO_DEEP_DIVE §12 Q1](../03-experiments/SFT_DPO_DEEP_DIVE.md) |
| DPO 变体：IPO / KTO / SimPO | IPO 平方损失防偏好过拟合；KTO 单边标注；SimPO 去参考模型+长度归一化 | [SFT_DPO_DEEP_DIVE §12 Q4](../03-experiments/SFT_DPO_DEEP_DIVE.md) |
| GRPO 后继：DAPO / GSPO | DAPO：Clip-Higher、动态采样、token 级归一、去 KL；GSPO：长度归一化的序列级重要性比 | [GRPO_DEEP_DIVE §12](../03-experiments/GRPO_DEEP_DIVE.md) |
| on-policy vs off-policy 三层澄清 | 数据来源 / 训练目标 / 重要性采样修正是三个独立的层；本项目 DPO 全程 off-policy | [SFT_DPO_DEEP_DIVE §7](../03-experiments/SFT_DPO_DEEP_DIVE.md) |
| 何时 SFT / 何时 RL（项目核心判据） | gold 可直接模仿 → SFT；SFT loss 见底且残错 pass@G > pass@1 → 才轮到 RL | [GRPO_DEEP_DIVE §10 R5](../03-experiments/GRPO_DEEP_DIVE.md)、[EXPERIMENT_LOG §14.2](../03-experiments/EXPERIMENT_LOG.md) |

## 2. DPO 知识点

| 知识点 | 一句话核心 | 出处 |
|---|---|---|
| DPO 损失与隐式奖励 | L=−logσ(β[(π_c−π_r)−(ref_c−ref_r)])；隐式奖励 r=β·log(π/π_ref) | [SFT_DPO_DEEP_DIVE §3](../03-experiments/SFT_DPO_DEEP_DIVE.md)、`vita/train/dpo_loss.py` |
| 首步 loss ≡ −log0.5 恒等式 | policy≡ref 时 logits=0，loss=0.6931 是恒等式；验证链路不验证有效性 | [SFT_DPO_DEEP_DIVE §12 Q7](../03-experiments/SFT_DPO_DEEP_DIVE.md) |
| disable_adapter 参考模型 | LoRA 前向 h=W₀x+BAx，关断即还原基座，零额外显存；前提：ref=训练起点=挂载基座 | [SFT_DPO_DEEP_DIVE §12 Q6](../03-experiments/SFT_DPO_DEEP_DIVE.md) |
| 图像去重（image_group_size） | chosen/rejected 共享一图只编码一次，实测省 44–46%（融合开销不减所以不到 50%） | [SFT_DPO_DEEP_DIVE §12 Q8](../03-experiments/SFT_DPO_DEEP_DIVE.md)、[ARCHITECTURE §14](../00-background/ARCHITECTURE.md) |
| fp32 log-prob | bf16 在 152k 词表上按序列求和差 ~2.9 nats，DPO 做"差的差"再乘 β 会被淹没 | `vita/train/dpo_loss.py` 头注释 |
| 序列 logp 求和不平均 | 原始 DPO 求和；平均是变体会改目标函数；长度偏置是 DPO 性质不是 bug | `vita/train/dpo_loss.py` 头注释 |
| 训练指标怎么算/怎么读 | margin=βz 会被逐对记忆推高；策略漂移 |Δlogp|/|logp| 才与评测同向；dead pairs 防数据静默失效 | [EXPERIMENT_LOG §5.1](../03-experiments/EXPERIMENT_LOG.md)、[SFT_DPO_DEEP_DIVE §11.1](../03-experiments/SFT_DPO_DEEP_DIVE.md) |
| 两种"分不开"（核心概念） | 一贯错=可修（特征在、方向反）；抛硬币=不可修（SNR 0.055，DPO 只会逐对记忆）；DPO 能重排已有表示不能造特征 | [SFT_DPO_DEEP_DIVE §12 Q2](../03-experiments/SFT_DPO_DEEP_DIVE.md) |
| 为什么不只 SFT | SFT 表达不了"不要做什么"；扔掉了比较标注；实测 DPO 增量 2.4× SFT；SFT 通道见底 | [SFT_DPO_DEEP_DIVE §12 Q3](../03-experiments/SFT_DPO_DEEP_DIVE.md) |
| squeezing 现象 | chosen/rejected 概率可同降只要差距拉大；margin 上涨主要靠 rejected 掉得快 | [SFT_DPO_DEEP_DIVE §12 Q5](../03-experiments/SFT_DPO_DEEP_DIVE.md) |
| SFT/DPO 数据重叠三层讨论 | benchmark 无泄露（硬前提）；参考项结构性抵消重叠（首步 reward=0）；disjoint 是待补的更干净设计 | [SFT_DPO_DEEP_DIVE §9–10](../03-experiments/SFT_DPO_DEEP_DIVE.md) |

## 3. GRPO 知识点

| 知识点 | 一句话核心 | 出处 |
|---|---|---|
| 组内归一化优势 | A_i=(r_i−mean)/std，组均值替代 critic；信号来自组内方差 | [GRPO_DEEP_DIVE §4](../03-experiments/GRPO_DEEP_DIVE.md) |
| KL 估计器 k1/k2/k3 | 本项目用 k3：exp(d)−d−1（d=ref−policy），无偏且恒非负；可验证奖励+单步复用下 β=0 无精度损失（R6 消融证实） | [GRPO_DEEP_DIVE §4/§10 R6](../03-experiments/GRPO_DEEP_DIVE.md)、`vita/train/grpo_loss.py:115` |
| 首步三不变量 | kl=0、ratio=1、advantage_std≈1——多卡/融合/参考模型正确性的烟雾判据 | [GRPO_DEEP_DIVE §5](../03-experiments/GRPO_DEEP_DIVE.md) |
| 多模态扩展六要点 | 视觉特征融合进 prompt embedding 一次、G 份 rollout 复用；与 DPO 去重机制的差异 | [GRPO_DEEP_DIVE §6](../03-experiments/GRPO_DEEP_DIVE.md) |
| μ 步样本复用（PPO-style） | 同批 rollout 复用 μ 个优化步；复用步 ratio≠1、clip 生效；μ>1 才需要 clip | [GRPO_DEEP_DIVE §3](../03-experiments/GRPO_DEEP_DIVE.md) |
| 退化组处理 | 组内全对/全错 → std=0 → 零优势跳过，监控 degenerate_frac | [GRPO_DEEP_DIVE §4/§10.5](../03-experiments/GRPO_DEEP_DIVE.md) |
| 指标手册（逐指标含义与诊断） | reward/kl/ratio/clip_frac/degenerate_frac/completion_len 每个的健康形状 | [GRPO_DEEP_DIVE §10.5](../03-experiments/GRPO_DEEP_DIVE.md) |
| 格式是怎么学会的 | prompt 指令给冷启动概率 + format 分级奖励组内排序 + answer 提取兜底防信号锁死 | [GRPO_DEEP_DIVE §8](../03-experiments/GRPO_DEEP_DIVE.md)、`vita/train/rewards.py` |
| 代理奖励为什么失败 | 开放式描述里 8 个 rollout 的代理分差是风格运气，组归一化在给噪声排序（R1/R2 实录） | [GRPO_DEEP_DIVE §10](../03-experiments/GRPO_DEEP_DIVE.md) |

## 4. Reward 设计

| 知识点 | 一句话核心 | 出处 |
|---|---|---|
| 可插拔奖励注册表 | @register_reward 装饰器 + 权重组合（answer:1.0,format:0.3） | `vita/train/rewards.py`、[GRPO_DEEP_DIVE §8](../03-experiments/GRPO_DEEP_DIVE.md) |
| 可验证 vs 代理奖励 | 二值精确匹配的组内对错方差是干净优势信号；代理分的组内排序是噪声——同一实现 +33pt vs 纹丝不动 | [GRPO_DEEP_DIVE §8/§10](../03-experiments/GRPO_DEEP_DIVE.md) |
| 分级 format 奖励 | 1.0 完整结构 / 0.5 仅 answer 标签 / 0.0——早期没人全对时组内仍有方差 | `vita/train/rewards.py:146` |
| LLM Judge 连续分 | 数字 token 概率取期望（非 argmax），得连续分数（R4 未用，R2 用过） | [GRPO_DEEP_DIVE §8](../03-experiments/GRPO_DEEP_DIVE.md) |
| Reward hacking 与监控 | 长度奖励会催长废话、keyword 会催堆砌；用 completion_len + 人工抽查监控 | [GRPO_DEEP_DIVE §8](../03-experiments/GRPO_DEEP_DIVE.md) |

## 5. 多模态融合（VITA 特有）

| 知识点 | 一句话核心 | 出处 |
|---|---|---|
| 负索引占位符机制 | `<image>` → IMAGE_TOKEN_INDEX(−200)，融合时替换成视觉 embedding | [PRIMER](../00-background/PRIMER.md)、[ARCHITECTURE §5](../00-background/ARCHITECTURE.md) |
| 三编码器结构 | InternViT-300M（图）+ 音频编码器（341M，RL 全程冻结）+ Qwen2.5-7B | [PRIMER](../00-background/PRIMER.md) |
| 动态切图与 token 预算 | 动态 patch 最多 13 块，图像 token 数实测表 | [PRIMER](../00-background/PRIMER.md)、[BENCHMARKS](../04-evaluation/BENCHMARKS.md) |
| mm_projector | `Linear(4096→3584) → GELU → Linear(3584→3584)`（4096=1024×4 pixel shuffle），SFT 全参训练、RL 冻结 | [SFT_DPO_DEEP_DIVE §5](../03-experiments/SFT_DPO_DEEP_DIVE.md) |
| 融合后 labels 重对齐 | 拼入图像 embedding 改变序列长度，collator 的 labels 必须融合后重对齐——多模态 DPO 最易错处 | [SFT_DPO_DEEP_DIVE §3](../03-experiments/SFT_DPO_DEEP_DIVE.md) |
| 状态 token（☜☞☟） | 单 token 查询状态标记，生僻字形保证 1 token；GRPO 奖励里已移除 | [ARCHITECTURE §7](../00-background/ARCHITECTURE.md) |

## 6. 训练工程

| 知识点 | 一句话核心 | 出处 |
|---|---|---|
| ZeRO-1/2/3 显存数学 | 全参 7B 单卡 ~98GB（14+28+56），必须 ZeRO-3；LoRA 可训参数少 ZeRO-2 够 | [SFT_DPO_DEEP_DIVE §4](../03-experiments/SFT_DPO_DEEP_DIVE.md) |
| DDP/ZeRO/FSDP/Megatron 选型 | 数据并行 → 分片 → 张量/流水线并行的升级路径与本项目为何停在 ZeRO | [GRPO_DEEP_DIVE §13](../03-experiments/GRPO_DEEP_DIVE.md) |
| LoRA 数学与合并 | h=W₀x+(α/r)BAx；合并=把 BA 加进 W₀；可多次"SFT→合并→再挂新 adapter" | [GRPO_DEEP_DIVE §7](../03-experiments/GRPO_DEEP_DIVE.md) |
| 冻结拓扑 | RL 阶段只训 LLM 的 LoRA；vision tower/audio/projector 冻结；SFT 阶段 projector+LLM 全参 | [GRPO_DEEP_DIVE §7](../03-experiments/GRPO_DEEP_DIVE.md)、[SFT_DPO_DEEP_DIVE §2](../03-experiments/SFT_DPO_DEEP_DIVE.md) |
| bf16/tf32/fp32 精度选择 | 训练 bf16、矩阵乘 tf32、log-prob 求和必须 fp32 | `dpo_loss.py` 头注释、[GRPO_DEEP_DIVE §4](../03-experiments/GRPO_DEEP_DIVE.md) |
| gradient checkpointing | 反向时重算前向，以计算换激活显存；rollout 阶段无效（无反向） | [GRPO_DEEP_DIVE §9](../03-experiments/GRPO_DEEP_DIVE.md) |
| 学习率量级 | 全参 SFT 1e-6（防遗忘）≪ LoRA DPO 2e-5 ≪ 快速实验；理由见各配置表 | [EXPERIMENT_LOG §8.2](../03-experiments/EXPERIMENT_LOG.md) |

## 7. 推理与框架选型

| 知识点 | 一句话核心 | 出处 |
|---|---|---|
| HF generate vs vLLM vs SGLang vs TRT-LLM | 易用性/吞吐/RadixAttention 前缀复用/极致延迟的取舍 | [GRPO_DEEP_DIVE §13](../03-experiments/GRPO_DEEP_DIVE.md) |
| vLLM 两大机制 | PagedAttention（KV cache 分页）+ continuous batching（连续批调度） | [GRPO_DEEP_DIVE §12](../03-experiments/GRPO_DEEP_DIVE.md) |
| RL 训推一体三难题 | 权重同步、训推 logprob 失配（重要性修正）、rollout 成本占比 | [GRPO_DEEP_DIVE §13](../03-experiments/GRPO_DEEP_DIVE.md) |
| 为什么不用 TRL/veRL | VITA 融合与标准 forward(input_ids) 接口不兼容，融合必须在 trainer 内部 | [SFT_DPO_DEEP_DIVE §12 Q19](../03-experiments/SFT_DPO_DEEP_DIVE.md) |
| 本项目 rollout 实际路径 | HF generate 批量 on fused embeddings（G=8 一批），未用 vLLM 的原因与收益预估 | [GRPO_DEEP_DIVE §13](../03-experiments/GRPO_DEEP_DIVE.md) |

## 8. 统计与实验方法论

| 知识点 | 一句话核心 | 出处 |
|---|---|---|
| 四条方法论原则 | delta 带噪声带报 / 一轮一变量 / 训练前量数据 / 中间 checkpoint 全套评测 | [EXPERIMENT_LOG §2.2](../03-experiments/EXPERIMENT_LOG.md) |
| 可分性探针与 SNR | 冻结基座给 pair 打分：accuracy、gap 均值/标准差、SNR；SNR<0.2 → 训得动不变好 | [EXPERIMENT_LOG §6](../03-experiments/EXPERIMENT_LOG.md)、`tools/probe_preference_separability.py` |
| 噪声带计算 | 1.96·√(p(1−p)/n)，n 取独立题目数（MMBench 1292 非 4876 行） | [EXPERIMENT_LOG §2.2](../03-experiments/EXPERIMENT_LOG.md) |
| McNemar 检验 | 配对样本只看翻转题（改对/改错），功效远高于两比例检验 | [EXPERIMENT_LOG §9.3](../03-experiments/EXPERIMENT_LOG.md) |
| bootstrap CI 与 win rate | held-out 评测的区间估计（GRPO 用） | `tools/eval_grpo_heldout.py` |
| 阈值平移排除法 | 分半边看翻转的选择性：净增 54 题证明是判别力改善不是校准平移 | [EXPERIMENT_LOG §9.4](../03-experiments/EXPERIMENT_LOG.md) |
| 评测集选择与污染控制 | 六个规则判分基准；训练数据来自 VQA train split 零重叠 | [EXPERIMENT_LOG §2.3](../03-experiments/EXPERIMENT_LOG.md) |
| 配平对照设计 | 同 prompt 预算/容量/起点的 SFT 对照臂；stage-2 disjoint 采样 | [GRPO_DEEP_DIVE §10 R4 后续/R5](../03-experiments/GRPO_DEEP_DIVE.md) |

## 9. 训练细节速查（每轮配置）

| 实验 | 配置要点 | 出处 |
|---|---|---|
| DPO A/B/C | 3000/3000/15000 对，lr 5e-6/2e-5/2e-5，batch 16/16/63，LoRA r64 ZeRO-2 | [EXPERIMENT_LOG §5.1](../03-experiments/EXPERIMENT_LOG.md) |
| SFT（第 5 轮） | 20000 chosen，8 卡 ZeRO-3 全参（冻视觉/音频），lr 1e-6，625 步 1h54m | [EXPERIMENT_LOG §8.2](../03-experiments/EXPERIMENT_LOG.md) |
| DPO D（第 6 轮） | 与 C 完全同配置、起点换 SFT checkpoint，250 步 2h26m | [EXPERIMENT_LOG §9.1](../03-experiments/EXPERIMENT_LOG.md) |
| GRPO R1/R2 | RLAIF-V 代理奖励，8 卡，R2 加 Judge/lr 5e-6/β 0.01，91 步止损 | [GRPO_DEEP_DIVE §10](../03-experiments/GRPO_DEEP_DIVE.md) |
| GRPO R4 | CLEVR G=8、β=0.04→答案奖励 1.0+format 0.3、lr 5e-6、400 步 4 卡 3.2h | [GRPO_DEEP_DIVE §10](../03-experiments/GRPO_DEEP_DIVE.md) |
| SFT 对照臂 | 同 6400 prompt、LoRA 同容量、26 分钟 | [GRPO_DEEP_DIVE §10 R4 后续](../03-experiments/GRPO_DEEP_DIVE.md)、`script/train/sft_clevr.sh` |
| R5 阶段二 | 同一阶段一 SFT checkpoint 起点、同批 disjoint 新 prompt、续 SFT vs 接 GRPO | [GRPO_DEEP_DIVE §10 R5](../03-experiments/GRPO_DEEP_DIVE.md) |
| R6 | β=0 消融：与 R4 同配置去 KL——held-out 77.0%/OOD 56.5%，与 R4 全在噪声内，KL 项无可测影响 | [GRPO_DEEP_DIVE §10 R6](../03-experiments/GRPO_DEEP_DIVE.md)、wandb `grpo-clevr-r6-beta0` |

## 10. 必背数字

**DPO 线**

| 数字 | 含义 |
|---|---|
| 52.2% / 0.055 → 58.0% / 0.218 | SFT 前后可分性 / SNR（诊断与修复） |
| −7.84 → +4.75 | DPO 初始 logp gap（根因最直观证据） |
| 0.047 → 0.181 | C→D margin（训练信号 3.8×） |
| 10.97 → 10.34 → 8.82 | POPE 幻觉率：baseline → SFT → SFT+DPO（贡献分解） |
| p=4.2×10⁻⁵ / −78 假阳性 / 净增 54 题 | McNemar 与选择性翻转 |
| ±2.27 / ±2.48 | MMBench / MMStar 噪声带（为什么 +1.47 不算显著） |

**GRPO 线**

| 数字 | 含义 |
|---|---|
| 44.6% → 77.4%（+32.8pt，win rate 0.977） | R4 主结果，400 步 |
| 216 步纹丝不动 vs 400 步 +33pt | 代理 vs 可验证奖励（同一实现） |
| MME 2353.5→2354.3 / POPE 89.14→89.07 | 通用回归零退化 |
| 37.5% → 54.5%（GRPO）vs 63.0%（SFT） | SuperCLEVR OOD：SFT 反超 |
| 75.4% / 26 分钟 / 1:7.5 | SFT 对照臂：分布内追平的成本比 |
| ~77–78% | 任务天花板，三次复现：R4 77.4 / R5 续 SFT 78.0 / R6 77.0 |
| 77.0% / 56.5%（β=0）vs 77.4% / 54.5%（β=0.04） | R6 消融：KL 项无可测影响 |
