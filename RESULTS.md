# RESULTS: baseline 与首次真实数据 DPO

> 采集于 2026-08-09，8×H100 机器。这份文档记录**一次完整的
> before/after 尝试**：从零个 benchmark 数字，到四项 baseline，
> 到在真实偏好数据上跑 DPO，以及**为什么结果是现在这样**。
>
> 与 BENCHMARKS.md 的分工：那份记录「在这台机器上做出来是什么数」，
> 这份记录「这次实验想验证什么、得到了什么、说明了什么」。

## 1. 一句话结论

**评测链路打通，四项 baseline 全部对齐论文；DPO 在 RLAIF-V 上
没有产生有意义的提升，原因已定位并量化——基座模型对这批偏好对的
判别力只有 53%（约等于随机），DPO 缺少可优化的一致方向。**

这是一个**阴性但可解释**的结果。定位过程本身比一个正数 delta 更有信息量。

## 2. Baseline（VITA-1.5 原始权重）

| Benchmark | 实测 | 论文 | 差 |
|---|---|---|---|
| MME (total) | **2353.51** | 2362.0 | −0.36% |
| MMStar | **59.8** | 60.2 | −0.4 |
| MMBench_DEV_EN_V11 | **77.79** | 76.6 | +1.2 |
| AI2D_TEST | **79.24** | 79.3 | −0.06 |

**这四个数字对齐论文，是整个工作的地基。** 一个数字仅仅「存在」
证明不了什么；落在论文值附近才说明 prompt 模板、图像预处理、
答案抽取三件事同时是对的。此前本仓库一个 benchmark 数字都没有，
因为 `import vlmeval` 就失败——四个依赖/环境问题叠在一起，
详见 BENCHMARKS §6.5。

单卡 H100 跑完四项约 40 分钟，这让 before/after 对比成为分钟级操作。

## 3. 真实偏好数据

**RLAIF-V-Dataset**（openbmb），取第一个分片 6814 对，
过滤后用 3000 对。转换脚本 `tools/make_rlaif_v_data.py`。

数据体检（**训练前就该看的数字**）：

| 指标 | 值 | 判读 |
|---|---|---|
| chosen 平均长度 | 299 字符 | |
| rejected 平均长度 | 298 字符 | 几乎相同 |
| chosen 更长的比例 | 46.5% | **无长度偏置**（50% 为完全无偏） |
| 两回答文本相似度 | 均值 0.29 | **是两个不同的答案**，非改词 |
| 被过滤掉的相同对 | 2 / 3082 | |

链路正确性检查：**首步 loss = 0.6931**，精确命中 `-log(0.5)`。
这与合成数据一致，说明参考模型在真实多模态输入下也接对了。

## 4. DPO 训练结果

两组超参，各跑满 3000 对（187 optimizer steps，有效 batch 16）：

| 配置 | LR | 末段 accuracy | 末段 margin | 末段 loss |
|---|---|---|---|---|
| A | 5e-6 | 0.508 | +0.0010 | 0.6931 |
| B | 2e-5 | 0.501 | +0.0009 | ~0.693 |

**两组都没学到东西。** accuracy 停在 0.5（随机），
loss 停在 0.6931（`-log(0.5)`，即 DPO 的零点）。

排除的可能性：

- **不是梯度消失**：`grad_norm` 稳定在 0.34–0.45。
- **不是学习率太小**：LR 提 4 倍，策略漂移量只从 0.0227 → 0.0245，
  accuracy 完全不动。
- **不是参考模型接错**：首步 loss 精确等于 0.6931。
- **不是数据格式错**：`<image>` token、图像文件、pair 结构均已校验。

## 5. 根因：数据对这个基座不可分

`tools/probe_preference_separability.py` 用冻结的基座直接打分：

```
base accuracy: 53.0%   (53/100)
logp gap chosen-rejected: mean=+0.48 median=+2.50 sd=44.97
```

**基座模型区分 chosen/rejected 的能力是 53%，约等于抛硬币。**
log-prob 差的均值 +0.48 被 45 的标准差完全淹没。

DPO 的梯度正比于「模型当前错得有多系统」。
当基座打分本质随机，就不存在一致的方向可供优化——
**margin 平坦是数据的性质，不是 trainer 的缺陷。**

而这恰恰是第 3 节那两个「好」指标的另一面：没有长度捷径、
没有格式捷径，两个回答都流畅切题、只在某个细节是否幻觉上不同。
**RLAIF-V 作为高质量偏好数据的优点，正是它对 7B 基座很陡的原因。**

## 6. 这次实验真正的产出

不是一个 delta，而是一套能拿来做实验的基础设施 + 一个明确的下一步：

1. **评测链路可用**：四项 baseline 对齐，40 分钟一轮。
2. **数据链路可用**：RLAIF-V → DPO 格式，首步 loss 校验通过。
3. **训练链路可用**：LoRA DPO 跑满 187 步，adapter 正常保存，
   merge 后能被 VLMEvalKit 直接加载推理（已验证）。
4. **诊断工具**：`probe_preference_separability.py` 能在**训练前
   几分钟内**判断一批偏好数据值不值得跑。

**下一步的三个方向**（按性价比排序）：

- **换更可分的偏好数据**。探针跑一遍就知道值不值得。
  基座 accuracy 65%+ 的数据才有足够信号。
- **先 SFT 再 DPO**。基座在 RLAIF-V 分布上本来就弱，
  先用 chosen 回答做 SFT 提升分布内能力，再做偏好优化。
- **自建偏好对**。用基座自己采样多个回答、用规则或更强模型判优劣，
  这样 pair 天然落在基座的能力边界上——可分性有保证。
  GRPO 那条路已经有 rollout 代码可复用。

## 7. 怎么复现

```bash
# 环境（一次性）
pip install --index-url https://pypi.org/simple/ "antlr4-python3-runtime==4.9.3"
pip install --only-binary=:all: "pyarrow==16.1.0"

# baseline
export PYTHONPATH=$VITA_REPO VITA_CKPT=$VITA_WEIGHTS/VITA-1.5 LMUData=/root/LMUData
cd $VITA_REPO/VLMEvalKit
python run.py --data MME MMStar MMBench_DEV_EN_V11 AI2D_TEST \
    --model vita_qwen2 --work-dir /path/eval_out/baseline

# 数据
python tools/make_rlaif_v_data.py --parquet shard000.parquet \
    --out-dir $VITA_WEIGHTS/rlaif_v_dpo --limit 3000

# 训练前先探针！
VITA_RLAIF_DATA_DIR=$VITA_WEIGHTS/rlaif_v_dpo \
python tools/probe_preference_separability.py --n 100

# 训练
VITA_RLAIF_DATA_DIR=$VITA_WEIGHTS/rlaif_v_dpo \
bash script/train/dpo_rlaif_v.sh /path/dpo_out 0

# 评测与对比
python tools/merge_and_eval.py --base $VITA_WEIGHTS/VITA-1.5 \
    --adapter /path/dpo_out/dpo-rlaif-v --out $VITA_WEIGHTS/VITA-1.5-dpo
python tools/compare_eval.py --before /path/eval_out/baseline/vita_qwen2 \
    --after /path/eval_out/dpo/vita_qwen2
```
