# VITA-1.5 多模态强化学习项目

> 面向简历/面试的项目综述。DPO 部分有真实跑通的数字；GRPO 多模态扩展是本项目的核心工程贡献，代码完成、待权重下载后跑运行时验证（文中已标 `待回填`）。
> 完整实验过程见 [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)，架构见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

## 一句话

在 VITA-1.5（7B 多模态大模型：InternViT-300M 视觉塔 + 音频编码器 + Qwen2.5-7B LLM）上复现并扩展两条强化学习路线：**多模态 DPO**（已跑通，POPE 幻觉率 10.97% → 8.82%，p<0.0001）与**多模态 GRPO**（从纯文本扩展到图像+文本，核心工程贡献）。

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

> 上游 VITA-RL 的 GRPO 是纯文本：`prompt_embeds = embed_tokens(input_ids)`。我把它扩展到图像+文本路径。代码完成、py_compile 通过、设计对照源码验证；运行时烟雾测试待权重下载后回填。

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
| VITA-1.5 主权重（19.6GB）+ InternViT-300M（608MB） | 🔄 下载中（~22% / ~32%） |
| 多模态 DPO 复现 | ✅ 已验证（§A 真实数字） |
| GRPO 多模态扩展代码 | ✅ 完成，py_compile 通过 |
| GRPO 文本烟雾（`grpo_smoke_test.sh`，验 kl≈0） | ⏳ 待权重 |
| GRPO 图像烟雾（`grpo_mm_smoke_test.sh`，验 fusion） | ⏳ 待权重 |
| GRPO 真实训练（`grpo_rlaif_v_8gpu.sh`，RLAIF-V）+ 评测 | ⏳ 待权重 + GPU |
| 优化方案讨论 | ⏳ 全部跑完后 |

> 下一里程碑：权重下完后 → `localize_config.py` 本地化 → 文本 smoke → 图像 smoke（核心交付物运行时验证）→ 真实训练 + 评测。烟雾数字出来后回填本文件 B 节与现状表。
