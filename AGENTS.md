# AGENTS.md — 给 AI 代理的操作说明

本仓库是 VITA-1.5（多模态 MLLM）的 fork，在上游 SFT-only 代码库上加了
DPO 与 GRPO 两条 RL 训练线。人类阅读入口是 [README.md](./README.md)；
本文件是给 AI 代理的最短路径：环境怎么搭、改动怎么验证、哪些约束不能碰。

## 1. 环境重建（权威出处：docs/01-setup/ENVIRONMENT.md §2）

**不要** `pip install -r requirements*.txt`——安装顺序有依赖。
按 [ENVIRONMENT.md §2](./docs/01-setup/ENVIRONMENT.md#2-conda-环境复现)
的 7 步顺序执行（conda python=3.10 → torch 栈 → 核心依赖 → 补漏 →
重 pin numpy → deepspeed/peft → flash-attn 预编译 wheel → 评测依赖）。
自动化版本是 [script/dev/setup_env_dev.sh](./script/dev/setup_env_dev.sh)
（内含旧机器绝对路径，需替换）。装完与
[requirements-lock-grpo.txt](./requirements-lock-grpo.txt) diff 校验。

硬约束（violating any of these breaks the repo）：

- `transformers==4.41.1`：`vita/model/language_model/vita_qwen2.py`
  monkey-patch 了 `Qwen2ForCausalLM.forward`，升级即坏。
- `torch==2.3.1` + flash-attn 用官方预编译 wheel
  （cp310 / cu122 / torch2.3 / cxx11abiFALSE），不要源码编译。
- `numpy==1.26.4`：numba/librosa/opencv 会拉回 2.x，装完必须重 pin。
- VLMEvalKit 的依赖必须 `pip install --no-deps`，否则它升级 torch/transformers。
- 权重下载后必须跑 `python tools/localize_config.py` 把 checkpoint 里的
  HF repo ID 改为本地路径，否则加载要联网。

## 2. 资源获取

全部权重 / 数据集 / 评测集的 HF 链接与转换脚本见
[ENVIRONMENT.md §5](./docs/01-setup/ENVIRONMENT.md#5-资源下载总表)。
本仓库自产的 5 个 GRPO LoRA adapter 在
[lee31221/VITA-RL](https://huggingface.co/lee31221/VITA-RL)，
`tools/merge_and_eval.py` 合并进基座即精确复原评测模型。

## 3. 验证改动（按成本从低到高）

```bash
# CPU 即可，无需权重（改 RL 代码后必跑）
python tools/test_dpo_loss.py      # 19 checks
python tools/test_grpo_loss.py     # 39 checks
python tools/test_rewards.py       # 44 checks
python tools/test_audio_optional.py

# 改动任何 Markdown / 移动文件后必跑
python tools/check_doc_links.py    # 零断链才能提交

# GPU 冒烟（1 卡 LoRA 峰值 23.3 GB；全参需 8×80GB）
bash script/train/smoke_test_lora.sh

# RL 链路首步恒等式（比单测更硬的正确性判据）
#   DPO 首步 loss ≈ 0.6931 (=-log 0.5)；GRPO 首步 kl==0 且 ratio==1
```

## 4. 复跑实验

各轮实验（DPO 六轮、GRPO R1–R6 及对照）的完整编排脚本在
[script/dev/](./script/dev/README.md)——顺序、并行、判定条件都是当时
真实跑过的记录，仅需替换其中的旧机器绝对路径。训练入口：
`train_dpo.py` / `train_grpo.py`；数据制备脚本在 `tools/make_*.py`。
每个数字的原始出处（训练日志、trainer_state、评测 JSON）在
[artifacts/](./artifacts/README.md)。

## 5. 仓库约定

- 文档在 `docs/00-background` … `docs/05-review` 六级管线目录，
  导航见 [README.md](./README.md#documentation-map-organized-as-a-pipeline)。
- 上游代码风格保持原样，不做无关重构；上游已知缺陷记录在
  [ARCHITECTURE.md §12](./docs/00-background/ARCHITECTURE.md)，别顺手"修"。
- License 仅限学术/研究/教育用途（[License.txt](./License.txt)）。
