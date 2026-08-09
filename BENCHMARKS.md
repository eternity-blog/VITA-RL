# 实测基准记录

> 这份文档记录**只能在原机器上取得、离机后无法复现**的东西：实测性能、
> 精确的环境快照、真实运行日志的关键行、以及硬件配置。
>
> 其他文档讲「怎么做」，这份讲「在这台机器上，做出来是什么数」。
> 换机器后请把它当作**对照基线**，而不是承诺——不同硬件上的数字会不同，
> 但数量级和相对关系应当成立。
>
> 采集时间：2026-08-07。全部为单次或少数次运行的实测值，非统计平均。

## 目录

- [1. 硬件与系统](#1-硬件与系统)
- [2. 性能基准](#2-性能基准)
- [3. 训练指标基线](#3-训练指标基线)
- [4. 数值精度实测](#4-数值精度实测)
- [5. 磁盘与网络](#5-磁盘与网络)
- [6. 真实报错原文](#6-真实报错原文)
- [7. 换机器后怎么用这份文档](#7-换机器后怎么用这份文档)

---

## 1. 硬件与系统

| 项 | 值 |
|---|---|
| GPU | 8 × NVIDIA H100 80GB HBM3（81559 MiB） |
| GPU 互联 | **NV18 全互联**（任意两卡间 18 条 NVLink） |
| NUMA | 全部 GPU 在 node 0，CPU affinity `0,2,4,6,8,10` |
| 驱动 | 535.129.03（支持 CUDA ≤ 12.2） |
| 内核 | 5.14.0-3.1.8.kwai.x86_64 |
| 系统 gcc | **4.8.5**（太老，无法编译 C99/C++17，所以全程用预编译 wheel） |
| 系统 nvcc | 11.4.120（**未使用**——wheel 自带 CUDA 12.1 运行时） |
| conda | 4.9.2（非交互 shell 里 `conda activate` 需先 source，故直接改 PATH） |
| base Python | 3.8.5（**不是** vita 环境用的 3.10.18） |

NV18 全互联意味着 8 卡 ZeRO-3 的通信不是瓶颈。在 PCIe 拓扑的机器上，
同样的训练会明显更慢——这是换机器后最可能变化的一项。

## 2. 性能基准

### 2.1 推理（单卡）

| 场景 | 耗时 |
|---|---|
| 文本查询（首次，含编译） | 8.7 s |
| 文本查询（后续） | 1.6 s |
| 语音查询 `q1.wav` | 2.9 s |
| 噪声音频 `q2.wav` | 2.2 s |
| 视频（5 秒片段） | 1.5 s |
| checkpoint 重载后推理 | 1.6 s |

模型加载本身约 4-5 s（4 个分片）。

### 2.2 训练

| 配置 | 步数 | 耗时 | 峰值显存 | 产出 |
|---|---|---|---|---|
| SFT 全参，8 卡 ZeRO-3 | 3 | 15.4 s | ~18 GB/卡 | 16.6 GB checkpoint |
| SFT LoRA，单卡 ZeRO-2 | 24 | 10.4 s | **23.3 GB** | 308 MB adapter |
| DPO，单卡 | 24 | 20.8 s | — | 308 MB adapter |
| GRPO，单卡 G=8 | 12 | 39.4 s | — | 308 MB adapter |

**全参训练单卡放不下**：7B 需要约 98 GB（bf16 权重 14 + fp32 主权重 28 +
AdamW 两个动量 56）。实测在分配 `exp_avg_sq` 时 OOM。8 卡是下限。

**LoRA 的 23.3 GB 是关键数字**——它意味着一张 24 GB 卡（3090/4090/A10）
就能训。可训练参数 161.5M，占 7.6B 的 2.12%。

### 2.3 组件级

| 组件 | 数字 |
|---|---|
| InternViT 参数量 | 289.9M |
| whale 参数量 | 341.4M |
| 一个 448×448 图块 | **256 token @ 4096 维** |
| 音频 | **约 12.5 token/秒** |
| InternViT 前向（5 块） | 12.6 ms |
| InternViT 前向（10 块） | 22.4 ms |
| InternViT 前向（13 块） | 29.2 ms |
| InternViT 前向（26 块） | 53.9 ms |

图块数与耗时基本线性，所以图像去重省下的正好是那一半。

### 2.4 rollout（GRPO 用，纯文本，max_new_tokens=64）

| G | 耗时 | 峰值显存 | 多样性 |
|---|---|---|---|
| 4 | 1.3 s | 17.4 GB | 4/4 不同 |
| 8 | 1.1 s | 17.6 GB | 8/8 不同 |

G 从 4 到 8 显存几乎不变，说明瓶颈不在批大小——可以放心加大 G。

### 2.5 MME 评测吞吐

| 项 | 值 |
|---|---|
| 题目数 | 2374 |
| 单卡 H100 耗时 | **7 分 10 秒** |
| 吞吐 | 5.5 题/秒 |
| 峰值显存 | 18.4 GB |

单卡 18 GB 意味着 8 卡可以并行跑 8 个 benchmark，或者用
`--nproc` 切分同一个。**这也是 before/after 对比可行的前提**：
一轮完整评测是分钟级，不是小时级。

## 2.6 VLMEvalKit baseline（VITA-1.5 原始权重）

**采集于 2026-08-09，1×H100，`vita_qwen2` wrapper，无 LoRA。**

这是 DPO 前的对照基线——本仓库此前一直缺的东西。

### MME

| 指标 | 实测 | 官方论文 | 差 |
|---|---|---|---|
| **perception** | **1728.86** | 1721.3 | +7.6 |
| **reasoning** | **624.64** | 640.7 | −16.1 |
| **total** | **2353.51** | 2362.0 | −8.5 (−0.36%) |

**总分与论文差 0.36%，落在采样噪声内——这是评测链路正确的最强证据。**
如果差 5% 以上，先怀疑 prompt 模板或图像预处理接错了，而不是急着解释。

细分项（满分：感知类每项 200，认知类每项 200）：

| 子项 | 分数 | | 子项 | 分数 |
|---|---|---|---|---|
| existence | 195.0 | | code_reasoning | 155.0 |
| OCR | 192.5 | | numerical_calculation | 147.5 |
| landmark | 187.25 | | position | 138.33 |
| color | 185.0 | | commonsense_reasoning | 137.14 |
| text_translation | 185.0 | | artwork | 122.5 |
| celebrity | 181.47 | | | |
| scene | 178.0 | | | |
| posters | 173.81 | | | |
| count | 175.0 | | | |

**读这张表**：`existence` 195 接近满分而 `artwork` 122.5 垫底，
是 MME 的普遍形状，不是这个 checkpoint 的毛病。做 DPO 前后对比时，
**看 total 和 hallucination 相关子项（existence/count/position）**——
偏好数据（RLAIF-V 就是为降幻觉设计的）最可能在这几项上动，
而 OCR/celebrity 这类靠预训练知识的项预期不变。

### MMStar

| 指标 | 实测 | 官方论文 | 差 |
|---|---|---|---|
| **Overall** | **59.8** | 60.2 | −0.4 |

细分：

| 子项 | 分数 |
|---|---|
| instance reasoning | 73.2 |
| logical reasoning | 66.4 |
| coarse perception | 66.0 |
| math | 63.6 |
| fine-grained perception | 46.8 |
| science & technology | 42.8 |

**两个 benchmark 都落在论文 ±0.5% 内**，可以认为评测链路没有系统性偏差。
MMStar 是「去掉了能靠文本先验蒙对的题」的 benchmark，所以它比 MME
更能反映真实的视觉依赖——DPO 若真降了幻觉，这里的
`fine-grained perception` 应当有反应。

### MMBench_DEV_EN_V11

| 指标 | 实测 | 官方论文 | 差 |
|---|---|---|---|
| **Overall** | **77.79** | 76.6 (MMBench) | +1.2 |

论文报的是 MMBench 而非明确的 V11 dev split，所以这 1.2 点不必强行解释。

**这个 benchmark 用 circular evaluation**：4876 行其实是 1292 道题
各自轮换选项顺序的产物，**一道题要所有轮换都答对才算对**。
算噪声带时样本量是 1292 不是 4876，用错会把带宽低估一倍——
`tools/compare_eval.py` 里记了这件事。

能力维度（六大类）：

| 维度 | 分数 |
|---|---|
| AR（属性推理） | 84.76 |
| FP-S（单实例细粒度感知） | 84.08 |
| RR（关系推理） | 79.31 |
| CP（粗粒度感知） | 78.73 |
| FP-C（跨实例细粒度感知） | 70.95 |
| LR（逻辑推理） | 58.87 |

最低的三项—— `image_quality` 45.2、`future_prediction` 52.0、
`object_localization` 64.3 ——分别对应画质判断、时序预测和空间定位，
是 7B 级别模型的普遍短板。

### AI2D_TEST

| 指标 | 实测 | 官方论文 | 差 |
|---|---|---|---|
| **Overall** | **79.24** | 79.3 | −0.06 |

**四个 benchmark 全部对齐**，最大偏差 1.2 点（MMBench，且论文未注明 split）。
这套评测可以拿来做 before/after 了。

### 四项汇总

| Benchmark | 实测 | 论文 | 差 |
|---|---|---|---|
| MME (total) | 2353.51 | 2362.0 | −0.36% |
| MMStar | 59.8 | 60.2 | −0.4 |
| MMBench_DEV_EN_V11 | 77.79 | 76.6 | +1.2 |
| AI2D_TEST | 79.24 | 79.3 | −0.06 |

单卡 H100 跑完这四项约 40 分钟。

## 2.7 DPO 后的四项（RLAIF-V，3000 对，1 epoch）

同一套评测跑合并了 DPO adapter 的模型。**详细分析见 RESULTS.md。**

| Benchmark | baseline | LR 5e-6 | LR 2e-5 | 1.96σ |
|---|---|---|---|---|
| MME (total) | 2353.51 | 2352.38 | 2352.11 | — |
| MMStar | 59.80 | 59.80 | 59.93 | ±2.48 |
| MMBench_DEV_EN_V11 | 77.79 | 77.71 | 77.79 | ±2.27 |
| AI2D_TEST | 79.24 | 79.27 | 见 RESULTS | ±1.43 |

**全部在噪声带内**。这组数字的用法是**当作「无效干预」的参照**：
将来任何一次 DPO/GRPO 如果也长这样（绝大多数子项逐位相同、
变化项正负参半），先去跑
`tools/probe_preference_separability.py`，而不是继续调超参。

`1.96σ` 那一列是用二项分布算的：`1.96 * sqrt(p(1-p)/n)`，
其中 MMBench 的 n 取 **1292**（circular eval 的题目数）而非 4876 行。

## 3. 训练指标基线

**这些数字可复现，是判断改动是否等价的标尺。**

### 3.1 SFT 冒烟（8 卡，合成数据 24 条）

| 步 | loss | grad_norm |
|---|---|---|
| 1 | **3.1885** | 45.93 |
| 2 | **3.7144** | 55.69 |
| 3 | 2.7124 / 2.7129 | 38.35 / 38.48 |

第三步的两个值来自两次运行——**差 5e-4 是 bf16 多卡归约顺序的正常抖动**。
前两步逐位相同。

**用法**：改了 `vita_arch.py` 或 `vita_qwen2.py` 之后重跑，
若第一步 loss 不再是 3.1885，说明真的改变了行为。

### 3.2 DPO（单卡 LoRA，24 步）

```
首步: {'loss': 0.6931, 'rewards/chosen': 0.0, 'rewards/rejected': 0.0,
       'rewards/margin': 0.0, 'logps/chosen': -46.18857192993164}
```

**0.6931 = `-log(0.5)`，是数学恒等式**，不是经验值。初始时 LoRA 的 B 矩阵
为 0，策略恒等于参考，DPO logit 为 0。**命不中就说明参考模型接错了。**

`logps/chosen` 首步 **-46.18857192993164**——图像去重优化前后逐位相同，
这是那次优化数值等价的直接证据。

趋势（前半 vs 后半均值）：

| | 前半 | 后半 |
|---|---|---|
| `rewards/margin` | +0.00200 | +0.01054 |
| `rewards/accuracy` | 0.58 | 0.67 |

### 3.3 GRPO（单卡 LoRA，12 步，G=8）

```
首步: {'reward/mean': 0.8238, 'grpo/kl': 0.0, 'grpo/ratio': 1.0,
       'grpo/advantage_std': 1.0, 'groups/degenerate_frac': 0.0}
```

**首步 KL = 0** 同样是恒等式，作用与 DPO 的 0.6931 相同。

趋势：

| | 前半 | 后半 |
|---|---|---|
| `reward/mean` | 0.7891 | **0.8622** |
| `reward/length` | 0.1816 | **0.4009** |
| `reward/keyword` | 0.8854 | 0.9583 |
| `completion/len` | 75.1 | **67.3** |

**最后两行是关键**：length 奖励翻倍的同时回答真的变短——两个独立数字
朝一致方向移动，比单看 loss 下降更难碰巧发生。

### 3.4 健康指标的期望值

| 指标 | 健康值 | 异常意味着 |
|---|---|---|
| `tokenization mismatch` 计数 | **0** | 非 0 = 样本 label 被静默作废 |
| `groups/degenerate_frac`（GRPO） | 接近 0 | 接近 1 = 奖励无区分度 |
| `grpo/advantage_std` | 1.0 | 偏离 = 归一化出问题 |
| `non_lora_trainables.bin` 参数量 | **0** | 非 0 = 参考模型会漂移 |
| DPO 首步 loss | 0.6931 | 偏离 = 参考模型接错 |
| GRPO 首步 KL | 0.0 | 同上 |

## 4. 数值精度实测

### 4.1 log-prob 的 bf16 vs fp32

在 152064 词表、200 token 序列上：

| 计算方式 | 误差 |
|---|---|
| bf16 全程 | **6.29 nats** |
| bf16 输入 + fp32 内部 | **0.0061 nats** |

**差三个数量级。** DPO/GRPO 的 β 典型取 0.1，bf16 那点噪声会淹没偏好信号。
这是 `batch_sequence_logps` 内部上转 fp32 的原因。

### 4.2 图像去重的等价性

真实 InternViT 上，编码 10 块 vs 编码 5 块再复制：

```
两者逐位相同 ? True
最大绝对差   : 0.0
```

不是近似，是精确相等——视觉编码器是确定性的。

### 4.3 组内标准差为 0

```
组内全同   std=0.0000  朴素: NaN   加 eps: 0.000
单元素组   std=NaN     朴素: NaN   加 eps: nan
```

单元素组用**无偏**标准差本身就是 NaN，所以 `group_advantages` 用的是
**总体**标准差。

## 5. 磁盘与网络

| 项 | 值 |
|---|---|
| 可用磁盘 | 3.4 TB |
| VITA-1.5 权重 | 19 GB（4 个 safetensors 分片） |
| InternViT | 647 MB |
| conda 环境 `vita` | **6.8 GB** |
| HF 下载（单流，走代理） | **7.2 MB/s ≈ 26 GB/h** |
| HF 下载（`max_workers=8`） | 约 140 GB/h（19 GB 权重约 8 分钟） |
| pypi | 企业镜像 `pypi.corp.kuaishou.com` |
| SSH 22 / 443 | **均被封锁**，git 只能 HTTPS + token |

下载速度是规划数据集方案的关键约束，`DATASETS.md` 的三档方案就是按此推算的。

## 6. 真实报错原文

离机后无法复现，但换机器时很可能再遇到。

### 6.1 LoRA 崩溃（上游 bug，本 fork 已修）

```
ValueError: Target module Qwen2DecoderLayer(
  (self_attn): Qwen2Attention(...)
) is not supported. Currently, only the following modules are supported:
`torch.nn.Linear`, `torch.nn.Embedding`, `torch.nn.Conv2d`,
`transformers.pytorch_utils.Conv1D`.
```

根因：`find_all_linear_names` 把 whale 里叶子名为数字 `"0"` 的两个 Linear
（`encoder.enc.0.core.out.0`、`encoder.enc.1.embed.0`）收进目标集，
peft 按后缀匹配便命中了 `layers.0`。

### 6.2 `audios=None` 崩溃（上游 bug，本 fork 已修）

```
TypeError: 'NoneType' object is not subscriptable
```

发生在 `vita_arch.py` 的 `audio_features["inputs_embeds"]`。

### 6.3 单卡全参训练 OOM

```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.84 GiB
    at torch/optim/adamw.py:128, state["exp_avg_sq"] = torch.zeros_like(...)
```

前向反向都过了，卡在分配优化器状态。

### 6.4 掉进 pdb（上游遗留）

```
bdb.BdbQuit
  File ".../data_utils_video_audio_neg_patch.py", line 337, in preprocess_mixtral_two
```

我写数据检视脚本时真踩到的——忘记设 `default_conversation` 就会走进
mixtral 分支，那里有 4 处活跃的 `pdb.set_trace()`。多卡下表现为**无输出的挂起**。

### 6.5 omegaconf / antlr4 冲突（已解决，2026-08-09）

```
Exception: Could not deserialize ATN with version 3 (expected 4).
```

`omegaconf 2.3.1` 生成的语法文件是 antlr4 4.9 系列的产物，而企业镜像最低
只有 4.11/4.13。装上去 `import vlmeval` 直接炸在 `ATNDeserializer`。

**解法**（已验证）：从上游 PyPI 装 4.9.3，纯 Python 包，不需要编译器——
这台机器 gcc 4.8.5 编译不了任何东西，所以「无需编译」是关键条件。

```bash
pip install --index-url https://pypi.org/simple/ "antlr4-python3-runtime==4.9.3"
```

装完 `import vlmeval` 通过。**降级不影响其他包**：装完复查
`torch 2.3.1+cu121`、`transformers 4.41.1` 均未被动过。

### 6.5.1 评测链路的另外三个坑（同日一并解决）

按踩到的顺序：

**LMUData 路径被硬编码成上游作者的机器。** `vlmeval/smp/file.py:71`
写死 `/mnt/cfs/lhj`，本机没有这个目录，于是它无视已下载的数据集、
每次都想重新下载。设环境变量 `LMUData=/root/LMUData` 即可，
`LMUDataRoot()` 会优先读它。

**opencompass 的数据集服务器证书已过期。**

```
NSS error -8181 (SEC_ERROR_EXPIRED_CERTIFICATE)
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: certificate has expired
```

`openssl s_client` 查得 `*.openxlab.space` 的叶证书 `NotAfter: Apr 16 2026`，
而当天是 8 月 9 日——**是服务端过期，不是本机 CA 库的问题**（换 certifi
的 bundle 一样失败）。

绕过方式是 `curl -k` 下载后**用 VLMEvalKit 源码里存的 MD5 校验**，
而不是信任那张过期证书：MD5 来自 git 仓库，是独立于传输通道的可信来源，
篡改会被检出。四个数据集全部校验通过：

| 数据集 | 大小 | MD5（源码 `image_mcq.py` / `image_yorn.py`） |
|---|---|---|
| MME | 46 MB | `b36b43c3f09801f5d368627fb92187c3` |
| MMStar | 57 MB | `e1ecd2140806c1b1bbf54b43372efb9e` |
| AI2D_TEST | 160 MB | `0f593e0d1c7df9a3d69bf1f947e71975` |
| MMBench_DEV_EN_V11 | 36 MB | `30c05be8f2f347a50be25aa067248184` |

VLMEvalKit 自己也会在 `prepare_tsv` 里比对同一份 MD5（`image_base.py:83`），
所以校验通过的文件放进 `$LMUData` 后它会直接复用，不再尝试联网。

**pyarrow 在这台机器上必须走预编译 wheel。** `pip install pyarrow` 会
尝试源码编译并失败在 gcc 4.8.5 上。`pip install --only-binary=:all:
"pyarrow==16.1.0"` 直接装 wheel，秒过。这和 REPRODUCE.md 里
flash-attn 用 wheel 是同一个原因。

### 6.6 moviepy 模块路径变更（已用垫片绕过）

```
ModuleNotFoundError: No module named 'moviepy.editor'
```

moviepy 2.x 移除了该模块，镜像上又没有 1.0.3。垫片方案见 `HANDBOOK.md §5`。

## 7. 换机器后怎么用这份文档

**第一步——先对硬件差异**：

| 如果新机器 | 预期影响 |
|---|---|
| GPU 少于 8 张 | 全参 SFT 跑不了，改用 LoRA（23.3 GB） |
| 不是 NVLink 全互联 | 8 卡训练会明显变慢，单卡不受影响 |
| 显存 < 24 GB | LoRA 也要减 batch 或序列长度 |
| Python 不是 3.10 | flash-attn 的预编译 wheel 要换 |
| gcc ≥ 7 | 可以从源码编译 flash-attn，不必用 wheel |

**第二步——按 `MIGRATION.md` 重建**，然后用 §3 的数字对照：

```bash
# 环境自检
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
# 期望 2.3.1+cu121

# 不用 GPU 的验证（最快确认代码没坏）
python tools/test_dpo_loss.py      # 19 项
python tools/test_grpo_loss.py     # 39 项
python tools/test_rewards.py       # 44 项
python tools/test_image_dedup.py   # 11 项
python tools/test_audio_optional.py

# 需要 GPU 的对照
bash script/train/smoke_test_qwen.sh /tmp/x 8   # 首步 loss 应为 3.1885
bash script/train/dpo_smoke_test.sh /tmp/y 1    # 首步 loss 应为 0.6931
bash script/train/grpo_smoke_test.sh /tmp/z 1   # 首步 KL 应为 0
```

**哪些数字换机器后应当不变**：

- DPO 首步 `0.6931` 和 GRPO 首步 KL `0` —— **数学恒等式，必须命中**
- 图像去重的逐位相等 —— 确定性的
- 五套 CPU 测试的全部断言 —— 与硬件无关

**哪些会变**：

- 所有耗时和吞吐
- 峰值显存（受 GPU 型号、CUDA 版本影响）
- SFT 的 loss 数值（bf16 归约顺序与卡数相关；**同机同卡数应当复现**）

如果 CPU 测试全过、但 DPO 首步不是 0.6931，问题几乎必定在参考模型或
`mm_projector` 的冻结上——见
[ARCHITECTURE.md §14.5](./ARCHITECTURE.md#145-two-traps-that-produce-plausible-looking-wrong-runs)。
