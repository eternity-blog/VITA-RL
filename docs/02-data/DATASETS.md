# 训练数据集调研

> Language: **中文** — this survey is written in Chinese; the repo ID / size /
> licence tables in §3 and §4 are language-neutral and usable as-is.
>
> 目的：为 VITA-RL 找到可实际下载的训练数据。上游 VITA-1.5 未提供训练数据，
> 这份文档记录论文用了什么、其中哪些今天还能拿到、以及在本机 3.4 TB 磁盘
> 预算下的可行方案。
>
> 核实日期：2026-08-06。所有 `gated` 状态、仓库体积、许可证均通过
> HuggingFace API 实测，非引用数据集卡片。带宽为本机实测值。
>
> **2026-08-20 更新**：本 fork 的 RL 工作定位为**文本 + 图片/视频**多模态
> （音频编码器只作为 VITA-1.5 自带的冻结组件存在，不做音频训练），
> 因此 §4 语音调研与 §6 方案中的语音条目仅作历史参考。RL 阶段实际使用
> 的数据见 **§3.3**（RLAIF-V + CLEVR-70k，均已跑通）。
>
> 配套文档：[REPRODUCE.md](../01-setup/REPRODUCE.md) 环境复现，
> [ARCHITECTURE.md](../00-background/ARCHITECTURE.md) 代码走读（§9 数据管线）。

## 目录

- [1. 论文用了什么](#1-论文用了什么)
- [2. 硬约束](#2-硬约束)
- [3. 图像与视频数据：核实结果](#3-图像与视频数据核实结果)
- [4. 语音数据：核实结果](#4-语音数据核实结果)
- [5. 陷阱清单](#5-陷阱清单)
- [6. 推荐方案](#6-推荐方案)
- [7. 数据格式转换](#7-数据格式转换)
- [8. 未核实项](#8-未核实项)

---

## 1. 论文用了什么

VITA-1.5 论文 Table 1 列出 **22,133.16K（约 2213 万）** 条多模态指令数据，
另有 Table 1 之外的语音数据。按可获取性分类：

| 类别 | 数量 | 占比 | 可获取性 |
|---|---|---|---|
| 公开数据集 | ~14,847K | 67% | 大部分可下载，详见 §3 |
| 标注为 "Synthetic Data" 的五行 | ~7,286K | 33% | **未发布** |
| ICDAR2019-LSVT-QA | 630K | 3% | **未发布**（作者自建标注） |
| 内部 ASR 语料 | 110,000 小时 | — | **内部，不会发布** |
| TTS 生成语音 | 3,000 小时 | — | 未发布，但可自行用 TTS 复现 |

**结论：论文数据量的约三分之一无法复现。** 这不是障碍——那五行合成数据
的源图来自 Wukong / LAION / CC12M，方法论文里写了，但成品标注没放出来。

三阶段训练对数据的用法（论文 §Training）：

| 阶段 | 用什么 | 可训练部分 |
|---|---|---|
| 1.1 视觉对齐 | 20% caption | 仅视觉 adapter |
| 1.2 视觉理解 | 100% caption | 视觉编码器 + adapter + LLM |
| 1.3 视觉 SFT | 100% QA + 保留 20% caption | 同上 |
| 2.1 音频对齐 | 11,000 小时 ASR | 语音编码器（CTC）→ 语音 adapter |
| 2.2 音频 SFT | 4% caption + 20% QA，约半数文本问题换成 TTS 语音 | 全部 + 状态分类头 |
| 3.1 Codec | 3,000 小时文本-语音对 | TiCodec |
| 3.2 解码器 | 同上 | NAR + AR 解码器，**LLM 冻结** |

注意阶段 2.2 只用 4% caption + 20% QA，阶段 3 完全冻结 LLM。
**这意味着不必凑齐 22M 条数据才能开始做有意义的训练。**

## 2. 硬约束

本机实测，这两个数字决定了方案上限：

| 约束 | 实测值 | 测法 |
|---|---|---|
| 可用磁盘 | **3.4 TB** | `df -h /usr/local/kai`（已扣除 19 GB 权重） |
| HF 下载带宽 | **7.2 MB/s 单流 ≈ 26 GB/小时** | 实测下载 317 MB parquet 用时 43.8 s |

单流 26 GB/h 是走企业代理的实测值。`snapshot_download(max_workers=8)`
可以并发提速——下载 19 GB 的 VITA-1.5 权重实际用了约 8 分钟，折合
~140 GB/h——但受对端限速影响，不能线性外推。

**保守规划按 100 GB/小时算。** 于是：

| 方案规模 | 磁盘 | 下载耗时 | 可行性 |
|---|---|---|---|
| 300 GB | 9% | ~3 小时 | ✅ 舒适 |
| 1 TB | 29% | ~10 小时 | ✅ 可行 |
| 3 TB | 88% | ~30 小时 | ⚠️ 濒临爆盘 |
| 10 TB+ | — | ~100 小时 | ❌ **装不下** |

照搬论文的数据规模（AnyWord-3M 880 GB + ShareGPT-4o 4.9 TB +
Emilia 7.2 TB + ...）需要 15 TB 以上，本机物理上做不到。方案必须按预算裁剪。

## 3. 图像与视频数据：核实结果

全部经 HF API 实测。`GB` 列为仓库实际占用（`usedStorage`），
**不含**需另抓的源图。

| 论文行 | 仓库 ID | gated | 仓库 GB | 许可证 | 自带图？ |
|---|---|---|---|---|---|
| ShareGPT4V 99.5K | `Lin-Chen/ShareGPT4V` | 否 | 6.0 | cc-by-nc-4.0 | ❌ **仅 3 个 JSON** |
| ALLaVA-Caption 697K | `FreedomIntelligence/ALLaVA-4V` | 否 | 123.7 | cc-by-nc-4.0 | ⚠️ 部分 |
| ShareGTP4o-Image 55.5K | `OpenGVLab/ShareGPT-4o` | **auto** | **4898.8** | mit | ✅ |
| LLaVA-150K 218K | `liuhaotian/LLaVA-Instruct-150K` | 否 | 4.5 | cc-by-4.0 | ❌ 需 COCO |
| LVIS-Instruct 939K | `X2FD/LVIS-Instruct4V` | 否 | 1.6 | **无声明** | ❌ 需 COCO |
| ScienceQA 12.7K | `derek-thomas/ScienceQA` | 否 | 7.9 | cc-by-sa-4.0 | ✅ |
| LLaVA-OV ×4 行 | `lmms-lab/LLaVA-OneVision-Data` | 否 | **347.0** | **apache-2.0** | ✅ |
| UReader 100K | `Mizukiluke/ureader-instruction-1.0` | 否 | 88.4 | 无声明 | ✅ |
| SynDOG-EN | `naver-clova-ix/synthdog-en` | 否 | 52.1 | 无声明 | ✅ |
| SynDOG-CN | `naver-clova-ix/synthdog-zh` | 否 | 70.5 | 无声明 | ✅ |
| Anyword-3M 1709K | `stzhao/AnyWord-3M` | 否 | 880.6 | apache-2.0 | ✅ |
| ShareGemini 205K | `Share14/ShareGemini` | 否 | 0.4 | cc-by-nc-4.0 | ❌ 仅字幕 |
| ICDAR2019-LSVT-QA 630K | — | — | — | — | **未发布** |

### 3.1 最重要的一条：LLaVA-OneVision

`lmms-lab/LLaVA-OneVision-Data` —— **347 GB、Apache-2.0、未 gated、图像内嵌**。

一个仓库就覆盖论文 Table 1 里四行（General 1754K + Math 1140K +
Doc/Chart/Screen 4431K + General OCR 404K = 7730K，占公开部分的一半以上），
而且**不需要另抓任何源图**。许可证是全表最干净的。

一个已知落差：该仓库共 3.94M 条，而论文那四行加起来 7.73M。
论文计的应该是 QA **轮次**而非样本条数。此外仓库有 **89 个 config**，
名称与论文的四行划分不是一一对应（实测：`ocr` 命中 1 个、`chart` 3 个、
`screen` 1 个、`math` 18 个、`geo` 9 个），需要自行归桶。

### 3.2 「仅标注」与「自带图」的区别

这是 20 GB 与数 TB 的分水岭，逐个说明：

**仅标注，需另抓源图：**

- **ShareGPT4V** —— 实测仓库里只有 3 个 JSON 文件
  （`sharegpt4v_instruct_gpt4-vision_cap100k.json` 等），零图像。
  依赖 COCO train2017 + LAION/CC/SBU + SAM。
- **LLaVA-150K** —— 依赖 COCO train2017。
- **LVIS-Instruct4V** —— 依赖 COCO train2017 + val2017。三者中最干净：
  不需要 SAM，不需要 OCR-VQA。
- **ShareGemini** —— 仅 0.4 GB 字幕，视频需另找。而其中 WebVid 部分
  已因 Shutterstock 撤下而失效。

源图实测可达性：COCO `train2017.zip` 返回 **HTTP 200**（19 GB）。
一份 COCO 即可同时喂饱上面三个标注集。

**自带图，开箱即用：** LLaVA-OneVision（347 GB）、UReader（88 GB）、
SynthDoG（123 GB）、AnyWord-3M（881 GB）、ScienceQA（8 GB）。

### 3.3 本 fork RL 训练实际用的数据（2026-08-20 已落地）

上面各节是"复现论文预训练"的调研；本 fork 的 RL 阶段实际只用了两个
数据集，都已下载并跑通：

| 数据集 | 来源 | 用途 | 体积 | 转换工具 | 环境变量 |
|---|---|---|---|---|---|
| **RLAIF-V** | [`openbmb/RLAIF-V-Dataset`](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset)（HF/ModelScope，parquet） | DPO 偏好对 / SFT（chosen）/ GRPO prompt | 全量 ~14 GB | `make_rlaif_v_data.py` / `make_rlaif_v_sft_data.py` / `make_rlaif_v_grpo_data.py` | `VITA_RLAIF_GRPO_DATA_DIR` 等 |
| **CLEVR-70k 计数** | [`leonardPKU/clevr_cogen_a_train`](https://huggingface.co/datasets/leonardPKU/clevr_cogen_a_train)（HF，R1-V 同款） | GRPO 可验证奖励训练 | ~13 GB parquet → 70,000 图 | `make_clevr_grpo_data.py`（69,500 训练 + 500 held-out，`reward_meta={"answer": N}`） | `VITA_CLEVR_GRPO_DATA_DIR` |
| CLEVR SFT 对照 | 同上（同一 parquet，配平采样） | GRPO 的 SFT 对照臂（同 6,400 prompt，gold 直接监督） | 复用上行图片 | `make_clevr_sft_data.py`（阶段一）/ `make_clevr_stage2_data.py`（阶段二双格式，重建排除保证无重叠） | `VITA_CLEVR_SFT_DATA_DIR` |
| **SuperCLEVR test200** | [`jigsaw-r1/super_clevr`](https://huggingface.co/datasets/jigsaw-r1/super_clevr)（HF，R1-V 同款 OOD 集） | OOD 泛化评测（只评不训） | 26 MB parquet → 200 图 | `make_superclevr_eval_data.py`（GRPO-eval 格式，`eval_grpo_heldout.py` 直接消费） | —（评测传 `--data/--image-root`） |

权重与评测集的下载链接总表见 [ENVIRONMENT.md §5](../01-setup/ENVIRONMENT.md#5-资源下载总表)。

选 CLEVR 的理由与结果（held-out 44.6% → 77.4%）、SFT 对照与 OOD 结论
（通用零退化、OOD 上 SFT 63.0% 反超 GRPO 54.5%、任务天花板 ~77–78%）见
[GRPO_DEEP_DIVE.md](../03-experiments/GRPO_DEEP_DIVE.md)；RLAIF-V 上的 DPO 六轮见
[EXPERIMENT_LOG.md](../03-experiments/EXPERIMENT_LOG.md)。

## 4. 语音数据：核实结果

论文用了 110,000 小时内部 ASR 语料，这部分不可能拿到。公开替代品实测：

| 语料 | 仓库 ID | gated | 仓库 GB | 许可证 | 时长 |
|---|---|---|---|---|---|
| LoquaciousSet | `speechbrain/LoquaciousSet` | 否 | 3767.3 | cc-by-3.0/4.0 | 25,000 h 英 |
| Emilia | `amphion/Emilia-Dataset` | **auto** | 7171.1 | cc-by-4.0 ⚠️ | ~50,000 h 中 |
| GigaSpeech | `speechcolab/gigaspeech` | **auto** | 3041.9 | apache-2.0 ⚠️ | 10,000 h 英 |
| WenetSpeech | `wenet-e2e/wenetspeech` | **auto** | 1016.6 | 无声明 | 10,005 h 中 |
| AISHELL-1 | `AISHELL/AISHELL-1` | 否 | **3.7** | apache-2.0 | 178 h 中 |
| ASCEND | `CAiRE/ASCEND` | 否 | **18.9** | cc-by-sa-4.0 | 10 h 中英混说 |
| Common Voice 22 | `fsicoli/common_voice_22_0` | 否 | 936.1 | cc0-1.0 | 多语 |
| AudioQA-1M | `shenyunhang/AudioQA-1M` | 否 | 1706.3 | apache-2.0 | 语音问答 |

⚠️ **两处许可证名不副实**（API 上的值有误导性）：

- **GigaSpeech 标 `apache-2.0`，但那只覆盖元数据和工具链。**
  实际门禁条款写明「仅限非商业研究与教育用途」，SpeechColab 不持有音频版权。
  研究用途没问题，但它不是实质意义上的 Apache。
- **Emilia 标 `cc-by-4.0` 不完整。** Emilia 本体是 CC BY-**NC** 4.0，
  只有 Emilia-YODAS 部分是 CC BY 4.0。本项目研究用途无碍，
  但若将来发布权重需注意。

### 4.1 三个必须知道的语音坑

1. **Common Voice 已撤出 HuggingFace（2025-10）。**
   `mozilla-foundation/common_voice_17_0` 的 `gated` 是 `false`，看着能用，
   但实测 **`usedStorage: 0`、仅 2 个文件——是空壳**。
   16.1 和 11.0 直接 404。Mozilla 迁到了 Mozilla Data Collective
   （需账号 + `MDC_API_KEY`），并声明不打算再由第三方托管。
   `fsicoli/*` 镜像是实际可用路径，但与 Mozilla 立场相悖。

2. **中文 Common Voice 体量极小**：zh-CN 仅 239 小时已校验。
   对中英双语混合没有实质贡献。

3. **`gated: auto` ≠ 需要审批。** 实测 Emilia、GigaSpeech、WenetSpeech
   三者都是 `auto`——网页点一次同意即可，**不是**数日到数周的表单审批。
   这直接影响排期：可以当天开始下载。

## 5. 陷阱清单

已实测或有明确依据，按踩坑代价排序：

| 陷阱 | 后果 |
|---|---|
| `mozilla-foundation/common_voice_*` 是空壳 | 白等一次下载 |
| `WenetSpeech-Yue` / `-Chuan` 只有元数据 + URL | 号称 21.8k/10k 小时，实际需微信联系作者拿音频 |
| `pkufool/libriheavy` 只有 manifest | 无音频，需另抓数 TB 的 Libri-Light |
| LoquaciousSet 已预混 LibriHeavy/YODAS/People's Speech/CV18/VoxPopuli | **再单独下这些会重复计算，浪费约 5 TB** |
| YODAS 中文分片仅 299 h，且约半数语种标注有误 | 中文走 Emilia，别走 YODAS |
| OCR-VQA 源图链接大面积失效 | LLaVA-1.5 mix665k 路径会缺图 |
| SAM 需 Meta 表单 EULA | ShareGPT4V 完整版受阻 |
| WebVid 已被 Shutterstock 撤下 | ShareGemini 的 webvid 半边（101K）失效 |
| `lmms-lab/ScienceQA` 已改名且无 train split | 用 `derek-thomas/ScienceQA` |
| `FreedomIntelligence/ShareGPT-4o-Image` 不是论文那行 | 那是 2025 年的图像**生成**数据集，认错会下错 262 GB |
| GigaSpeech 2 只有 th/id/vi | 无中无英，与本项目无关 |
| 六个数据集**无许可证声明** | LVIS-Instruct4V、SynthDoG、UReader、LSVT 等——可下载但无授权，属灰色地带 |

## 6. 推荐方案

按磁盘预算给三档。全部满足：未 gated 或仅需点击、许可证允许研究用途、
自带图像或源图可达。

### 方案 A：最小可用（~260 GB，约 3 小时）

先跑通真实数据训练，验证管线，不追求规模。

| 数据集 | GB | 说明 |
|---|---|---|
| `lmms-lab/LLaVA-OneVision-Data`（选若干 config） | ~200 | 图像内嵌，Apache-2.0 |
| `derek-thomas/ScienceQA` | 8 | 图像内嵌 |
| `AISHELL/AISHELL-1` | 3.7 | 中文 ASR，178 h |
| `CAiRE/ASCEND` | 18.9 | 中英混说，直击双语场景 |
| COCO train2017 | 19 | 喂 LLaVA-150K + LVIS |
| `liuhaotian/LLaVA-Instruct-150K` + `X2FD/LVIS-Instruct4V` | 6 | 标注 |
| `Lin-Chen/ShareGPT4V`（cap100k 部分） | 6 | 标注，COCO 子集可覆盖 |
| **合计** | **~260 GB** | 磁盘占用 8% |

**建议从这一档开始。** 它足以驱动阶段 1.3（视觉 SFT）和 2.2（音频 SFT）
的完整流程——回想论文里 2.2 本就只用 20% QA。

### 方案 B：中等规模（~1.6 TB，约 16 小时）

在 A 基础上补 OCR 与更多中文语音：

| 增补 | GB |
|---|---|
| LLaVA-OneVision 全量 | 347（替代 A 的 200） |
| `Mizukiluke/ureader-instruction-1.0` | 88 |
| `naver-clova-ix/synthdog-en` + `-zh` | 123 |
| `wenet-e2e/wenetspeech`（点击同意） | 1017 |
| **合计（含 A）** | **~1.6 TB**（磁盘 48%） |

### 方案 C：接近论文（~2.6 TB，约 26 小时，不推荐）

再加 `stzhao/AnyWord-3M`（881 GB）与 `FreedomIntelligence/ALLaVA-4V`
（124 GB）。**不建议**：磁盘占用 78%，只剩约 760 GB 余量，而训练产生的
checkpoint 每个 16 GB，容易在训练中途爆盘。若一定要做，AnyWord-3M 走
ModelScope 镜像（`iic/AnyWord-3M`，214 GB）可省 666 GB，把总量压到 ~2.0 TB。

### 明确排除

| 数据集 | 原因 |
|---|---|
| `OpenGVLab/ShareGPT-4o` | 实测 **4.9 TB**，单个就超预算 |
| `amphion/Emilia-Dataset` | 7.2 TB，超预算 2 倍 |
| `speechbrain/LoquaciousSet` | 3.8 TB，且本项目重心是中英双语而非纯英 |
| `speechcolab/gigaspeech` | 3.0 TB |

这四个是调研阶段的首选推荐，但实测体积合计 18.9 TB——在 3.4 TB 预算下
**任意两个都装不下**。若将来有大容量存储，Emilia ZH 仍是最贴近论文内部
语料性质的中文选择（播客/访谈/脱口秀，非朗读语音）。

## 7. 数据格式转换

下载完不能直接用——需转成 VITA 的样本格式
（见 [ARCHITECTURE.md §9.1](../00-background/ARCHITECTURE.md#91-sample-format)）：

```json
{
  "set": "sharegpt4",
  "id": "000000000164",
  "conversations": [
    {"from": "human", "value": "<image>\n<audio>\n"},
    {"from": "gpt",   "value": "This is a well-organized kitchen…"}
  ],
  "image": "coco/images/train2017/000000000164.jpg",
  "audio": ["output_wavs/f61cf238b7872b4903e1fc15dcb5a50c.wav"],
  "inserted_id": 3
}
```

三个易错点：

1. **`set` 必须在 `FolderDict` 里注册**（`vita/config/dataset_config.py`），
   否则找不到图像目录。
2. **音频路径拼接是 `os.path.join(AudioFolder, "audio", file)`** ——
   中间那个 `"audio"` 是硬编码的，所以 `AudioFolder` 要指向
   `audio/` 的**父目录**。这点极易搞错，参考
   `tools/make_smoke_data.py` 的做法。
3. **分词长度不匹配会静默作废整条样本**
   （`data_utils_video_audio_neg_patch.py:642`：`target[:] = IGNORE_INDEX`）。
   样本照常进 batch、照常前反向，但对 loss 零贡献，只打印一行 WARNING。
   **接真实数据后务必统计这行的出现次数**——冒烟测试实测为 0，
   真实数据上若大量出现，说明模板有偏差，而 loss 曲线看起来会很正常。

语音方面，论文阶段 2.2 把约半数文本问题用 TTS 转成语音。要复现这一步，
要么用 `shenyunhang/AudioQA-1M`（1.7 TB，Apache-2.0，恰是 VITA 作者账号
`shenyunhang` 发布），要么自己跑 TTS 生成——后者更省磁盘且可控。

## 8. 未核实项

以下未能实测，据实标注而非猜测：

- **AISHELL-2 当前条款** —— `aishelltech.com` 返回 HTTP 429，
  非商业/签署表单的说法来自 Kaldi recipe 和 2018 年论文，非实时核实。
- **CV 26.0 的分语种时长与 MDC 许可证** —— 需 MDC 账号才能查。
- **SynthDoG 的确切条数** —— datasets-server 报 65,983（en）/ 63,728（zh），
  但仓库有 84/87 个 parquet 分片，两者对不上；Donut 论文称每语种 0.5M。
  无论取哪个都超过论文用的 100K。
- **LLaVA-OV 89 个 config 到论文四行的映射** —— 需自行归桶，
  预计实际可用 3.1–3.9M 而非论文的 7.7M。
- **`zhifeixie/Voices-in-the-Wild-2M` 的时长** —— 实测 197.5 GB、
  Apache-2.0、未 gated，但数据集卡片只给样例数不给小时数。
- **各 Google Drive 托管分片的当日配额** —— 目录能列出，
  配额限制只在实际下载时才暴露。
