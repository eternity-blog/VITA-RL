# VITA-1.5 多模态强化学习项目

> 面向简历/面试的项目综述。DPO 与 GRPO 两条线都有真实跑通的数字。
> 完整实验过程见 [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)（DPO 六轮）与
> [`GRPO_DEEP_DIVE.md`](GRPO_DEEP_DIVE.md)（GRPO 全记录），架构见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

## 一句话

在 VITA-1.5（7B 多模态大模型：InternViT-300M 视觉塔 + 音频编码器 + Qwen2.5-7B LLM）上复现并扩展两条强化学习路线：**多模态 DPO**（POPE 幻觉率 10.97% → 8.82%，p<0.0001）与**多模态 GRPO**（从纯文本扩展到图像+文本；在可验证奖励任务 CLEVR 计数上 400 步将 held-out 准确率 44.6% → 77.4%，win rate 0.977，通用基准零退化），并以配平对照实验界定方法边界（SFT 对照、SuperCLEVR OOD、阶段二续训对照）。

## 技术栈

PyTorch 2.3.1+cu121 · Transformers 4.41.1 · DeepSpeed ZeRO-2 · PEFT (LoRA r=64) · flash-attn 2.5.9.post1 · 8×H100 80GB · bf16

## 项目目标

给一个已有开源多模态基座**加一个 RL 阶段**，并严格测出它有没有用——不是"训完拿到一个数"，而是"和噪声可区分的、有代价的真实改进"。

为此整个项目分两步：
1. **先建立可信标尺**：跑通 VLMEvalKit 评测，且数字必须落在论文值附近（"一个数存在"证明不了什么，"落在论文值附近"才同时证明了 prompt 模板、图像预处理、答案抽取三件事都对）。
2. **再做干预并对比**：同一套标尺量 before/after。

---

## A. 复现多模态 DPO（已验证）

### 方法
直接偏好优化（DPO），参考策略 = 关闭 LoRA 适配器的同一权重。首步 loss 必须为 `ln2 ≈ 0.6931`（policy==reference），是接线性正确性的不变量。多模态融合通过 `prepare_inputs_labels_for_multimodal`：负索引占位符（IMAGE_TOKEN_INDEX=-200）被 InternViT 切片特征拼接替换，chosen/rejected 共享同一张图时用 `image_group_size` 去重，视觉塔每对只跑一次。

### 过程（六轮排除法）
前三轮在原始基座上直接 DPO 全部无效（六项指标 delta 全在 ±0.3 内），逐个排除假设：

| 轮 | 干预 | 假设 | 结果 |
|---|---|---|---|
| 1 | DPO 3000 对 / lr 5e-6 | 直接训就有效 | ❌ 全在噪声内 |
| 2 | lr × 4 | 学习率不够 | ❌ 训练更动、评测持平 → 排除调参 |
| 3 | 15000 对 / batch 63 | 规模不够 | ❌ 模型真漂移（6.7×）但净效应零 → 排除规模 |
| 4 | on-policy 自采样 | 数据分布外 | ⚠️ 方差降 26%，判据噪声让 accuracy 上不去 |
| 5 | 全参 SFT（只用 chosen） | 基座建模不足 | ⚠️ 可分性 53.6% → 58.0%，通用能力小降 |
| 6 | **SFT 起点 + DPO** | 组合上述 | ✅ **POPE 幻觉率 −2.15pt** |

根因（§6 两层诊断）：DPO 失效不是超参/规模问题，而是**候选可分性不足 + 判据噪声**。SFT 先把 chosen/rejected 的 logp 拉开到阈值以上，DPO 才能发力。

### 结果（真实数字）

| 指标 | baseline | SFT+DPO | Δ | 噪声带 |
|---|---|---|---|---|
| **POPE 幻觉率** | 10.97% | **8.82%** | **−2.15pt** | — (p=4.2×10⁻⁵) |
| **MMBench** | 77.79 | **79.26** | **+1.47** | ±2.27 |
| MMStar | 59.80 | 58.67 | −1.13 | ±2.48 |
| AI2D | 79.24 | 79.31 | +0.06 | ±1.43 |
| Hallu fAcc | 37.57 | 33.53 | −4.05 | — |

**一句话结论：在幻觉专项上显著变好（假阳性少 78 例），通用 MCQ 基本持平，视觉错觉类退化。**有代价的真实改进，与噪声可区分——不是全面提升，方向正确。

---

## B. 扩展 GRPO 到图像+文本（核心工程贡献）

> 上游 VITA-RL 的 GRPO 是纯文本：`prompt_embeds = embed_tokens(input_ids)`。我把它扩展到图像+文本路径。已完成运行时验证：文本/图像烟雾全过（首步 kl=0、ratio=1、advantage_std≈1），并在真实数据上完成 4 轮训练（终局：CLEVR 计数 held-out 44.6% → 77.4%，见现状表与 GRPO_DEEP_DIVE.md）。

### 设计：为什么这样改

GRPO 无 critic，用组内基准（group baseline）替代价值网络：每个 prompt 采样 G 个补全，组内标准化得 advantage = (r − 均值)/std。这让它在 7B 模型上可行（不需要第二个 7B 当 value head）。

扩展到多模态的关键洞察：**融合必须在 G 折叠展开之前**。原始 prompt 的 embedding 先融合好（视觉塔每张不同图只跑一次），然后再做 `(B,P,D) → (B*G,P,D)` 的纯张量重复给每个 rollout 复用。

这与 DPO 的融合不同：
- **DPO** 需要 `image_group_size` 去重——chosen 和 rejected 共享同一张图，两条序列都要同一份视觉特征，所以显式告诉融合函数"每 G 条序列共用一组特征"。
- **GRPO 不需要** `image_group_size`——每个 prompt 自带自己的图，G 个 rollout 是同一 prompt 的不同采样，复用同一份已融合的 prompt embedding 即可，视觉塔天然每张图只跑一次。

### 改动（两文件）

**`vita/train/grpo_trainer.py`** — 新增 `_fuse(model, inputs)`，镜像 DPO 的 `_fuse` 但去掉 `image_group_size` 参数：
```python
def _fuse(self, model, inputs):
    unwrapped = self.accelerator.unwrap_model(model)
    unwrapped.config.tokenizer_padding_side = "left"   # 生成右接，融合必须左填充
    _, _, attention_mask, _, inputs_embeds, _ = (
        unwrapped.prepare_inputs_labels_for_multimodal(
            inputs["input_ids"], None, inputs["attention_mask"],
            None, None,            # labels=None — GRPO 不监督任何 token
            inputs.get("images"), inputs.get("audios") or None,
            inputs.get("sf_masks"),
            # image_group_size 省略：见上设计
        )
    )
    return inputs_embeds, attention_mask
```

`compute_loss` 按 `any(has_image)` 分支：图像路径走 `_fuse`，文本路径走原始 `embed_tokens`（保持 byte-identity，见下）。其余 rollout/logps/loss 逻辑不变。

**`vita/train/grpo_data.py`** — `__getitem__` 按 `has_image` 分支：
- **图像路径**：`_load_image_tiles`（dynamic_preprocess 切 448×448 块）+ `_build_image_prompt`（`preprocess_multimodal` 把 `<image>` 展开成 n_tiles 份，`get_prompt("image")` 选图像系统提示）。
- **文本路径**：**绝不能调 `preprocess_multimodal`**。该函数在 `is_multimodal=True`（默认）时对每句跑 `.replace("\n\n","\n")`，会改变分词、破坏首步 `grpo/kl==0` 的 policy==reference 不变量。文本路径直接 `conv.append_message` + `get_prompt("lang")`，与原始纯文本 GRPO 数据集 byte 一致。

### 正确性不变量（烟雾测试通过判据）

| 信号 | 首步期望 | 含义 |
|---|---|---|
| `grpo/kl` | ≈ 0 | policy==reference；非零 = 融合后的参考策略接错或 mm_projector 逃出了冻结 |
| `grpo/ratio` | ≈ 1 | old_logps 取自同一前向（非两次 bf16 漂移），构造上为 1 |
| `reward/mean` | 上升趋势 | 图像特征真的到达策略；平坦 = 视觉路径静默 no-op |

参考策略 = `disable_adapter()` 关 LoRA；要求所有可训练参数都在 adapter 内，所以 `train_grpo.py` 冻结 mm_projector。`lora_dropout=0`：参考 pass 无 adapter 无 dropout，policy 端若有 dropout 会让首步 KL 无端非零。

### rollout 的一个非平凡绕过
`VITAQwen2ForCausalLM.generate` 对 `inputs_embeds` 抛 `NotImplementedError`，但 `Qwen2ForCausalLM.generate` 接受。直接调后者，让 prompt embedding 算一次复用——这也是视觉塔每 prompt 只跑一次（而非每 rollout）能成立的另一处依赖。

`_sequence_logps` 重建 `[已融合 prompt embedding | 采样 token embedding]` 做前向重算 logp（而非取自 generation），验证过与 generation 自身 scores 吻合到 ~4e-3（bf16 噪声）。重算 pass 承载梯度必须发生；old_logps 取自同一 pass 使首步 ratio 恰为 1。

---

## 评测方法论：噪声带

**每个 delta 必须和噪声带一起报。** 不知噪声带，`+0.3` 会被当成提升。噪声带按二项分布：`1.96·√(p(1−p)/n)`，n 取**独立题目数**（MMBench 是 1292 题而非 4876 行，因 circular eval 把一题展开成多行）。

| 指标 | 1.96σ 噪声带 |
|---|---|
| MMStar | ±2.48 |
| MMBench | ±2.27 |
| AI2D | ±1.43 |

前三轮 DPO 所有 delta 在 ±0.3 内 → 在这些噪声带内 → 无效干预的标准形状。SFT+DPO 是唯一一列有大幅正负分化的。

---

## 关键工程决策与踩坑（摘选）

1. **环境重建**：代理墙下 HF 大文件全死（0 字节、xet 卡在 1GiB 边界），ModelScope 是唯一返回真实字节的路径；pypi 总带宽受限（~3MB/min，并行不提速）。flash-attn 不能用 GitHub wheel（代理返回 0 字节），从 pypi sdist 本地编译，且必须用正确的 GPU 架构（H100=compute 9.0，非 A100 的 8.0）。
2. **/dev/shm 仅 512MB**：8 rank × N dataloader worker 走共享内存传 448×448 切片会 OOM（Bus error），多 GPU 训练必须 `dataloader_num_workers=0`。
3. **数据集注册**：新数据集必须在 `vita/config/__init__.py` 的 `DataConfig` 字典里登记，不能只定义 `dataset_config.py` 的模块级变量——`GRPOPromptDataset.__init__` 读的是 `DataConfig[name]`。
4. **文本路径 byte-identity**：上面 B 节强调的——为保 `grpo/kl==0` 不变量，文本记录绝不能过 `preprocess_multimodal` 的换行归一化。

---

## 现状与待办

| 模块 | 状态 |
|---|---|
| conda 环境 + 全部依赖（含 flash-attn） | ✅ 就绪 |
| VITA-1.5 主权重（19.6GB）+ InternViT-300M（608MB） | ✅ 已下载（ModelScope） |
| 多模态 DPO 复现 | ✅ 已验证（§A 真实数字） |
| GRPO 文本/图像烟雾（验 kl≈0、fusion） | ✅ 通过（首步不变量全部成立） |
| GRPO 真实训练 R1（RLAIF-V，规则奖励，125 步） | ✅ 跑通；reward 涨但基准不动（剂量+信号双不足） |
| GRPO 真实训练 R2（RLAIF-V，+LLM Judge，lr 5e-6/β 0.01） | ⏹ 91 步止损：KL 涨 6 倍而内容奖励无趋势 → 诊断为代理奖励组内排序噪声 |
| μ 步样本复用（PPO-style，`_ChunkRepeatSampler`） | ✅ 实现并烟雾验证（复用步 ratio≠1、clip 生效） |
| GRPO 真实训练 R4（CLEVR 计数，可验证奖励，400 步/4 卡/3.2h） | ✅ **held-out 44.6% → 77.4%（+32.8pt），win rate 0.977 [0.953, 0.994]** |
| R4 后续验证：通用回归 + SuperCLEVR OOD + 配平 SFT 对照 | ✅ MME/POPE/MMBench **零退化**；OOD 37.5%→54.5%；SFT 对照 26min 达 75.4%（OOD 63.0% 反超 GRPO） |
| R5 阶段二对照（同一 SFT 起点：续 SFT vs 接 GRPO，同批新 prompt） | ✅ 任务天花板 ~77–78%，两通道均近枯竭（GRPO 仅改变 8/500 输出；退化组开局 75%） |
| R6 β=0 消融（与 R4 同配置，唯一变量去 KL 项） | ✅ held-out **77.0%**（R4：77.4）、OOD **56.5%**（R4：54.5），全在噪声内——**KL 项无可测影响**，DAPO 立场获消融验证 |

> 方法论结论：同一套 GRPO 实现，代理奖励下 216 步纹丝不动，可验证奖励下
> 400 步 +33pt——GRPO 的成败首先取决于 reward 能否把组内 rollout 按真实
> 能力排序，其次才是超参剂量。对照实验进一步界定了适用边界：gold 可直接
> 模仿的任务上 SFT 同终点且 1/7.5 成本；"何时 SFT / 何时 GRPO"的判据是
> **SFT loss 是否见底 + 残错上 pass@G 是否 > pass@1**。完整记录与指标
> 手册见 [`GRPO_DEEP_DIVE.md`](GRPO_DEEP_DIVE.md)。

## 面试准备阅读路线

按"先叙事、再数字、后细节"的顺序（总复习清单见
[KNOWLEDGE.md](KNOWLEDGE.md)——全部知识点一句话核心 + 出处）：

1. **叙事骨架**（30 分钟）：本文 + [README_zh-CN.md](README_zh-CN.md)
   的 DPO/GRPO 两段故事——两条线各自的"失败→诊断→修复"弧线要能脱稿讲。
2. **DPO 线**（半天）：[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
   §1 摘要表 + §2.2 四条方法论原则 + §6 探针 + §8–9 贡献分解；然后
   [SFT_DPO_DEEP_DIVE.md](SFT_DPO_DEEP_DIVE.md) **§12 面试问答（21 问）**，
   重点 Q2（两种"分不开"）、Q3（为什么不只 SFT）、Q13（数据重叠）。
3. **GRPO 线**（半天）：[GRPO_DEEP_DIVE.md](GRPO_DEEP_DIVE.md)
   §10 五轮全记录 + §11 面试问答 + §12 演进（vLLM/DAPO/GSPO）+
   §13 框架选型；R5 的"何时 SFT / 何时 GRPO"判据是两条线的合流点。
4. **实现细节抽查**（按需）：[CODEMAP.md](CODEMAP.md) 定位文件 →
   [ARCHITECTURE_zh-CN.md](ARCHITECTURE_zh-CN.md) §14 RL 栈走读。

必背的三组数：DPO 线 **52.2%/0.055 → 58.0%/0.218**（诊断）、
**−7.84 → +4.75**（根因）、**10.97 → 10.34 → 8.82**（贡献分解）；
GRPO 线 **44.6 → 77.4**（R4）、**37.5 → 54.5 vs 63.0**（OOD 对比）、
**~77–78%**（天花板）。
