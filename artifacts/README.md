# artifacts/ —— 从开发机抢救下来的原始实验产物

> 开发机（及其上 123 GB 的 `vita-outputs/`）已计划回收。这个目录保存了其中
> **有复核价值的小文件**：训练日志、trainer_state、评测结果 JSON/CSV。
> 文档里的每个数字（`EXPERIMENT_LOG.md` §14、`GRPO_DEEP_DIVE.md` §10）
> 都能在这里找到原始出处。
>
> **不在这里的东西**：
> - 模型权重 / checkpoint / 合并后模型 —— 太大，且可按 `REPRODUCE.md` 重新训练
> - 数据集 —— 可用 `tools/make_*_data.py` 系列脚本重新生成
> - **DPO 时代（第 1–6 轮，2026-08 上旬）的原始日志** —— 跑在上一台开发机
>   （`/usr/local/kai/lx`），机器已回收，**原始文件不可恢复**。
>   数字与过程完整保存在 `EXPERIMENT_LOG.md` §3–§9，wandb 云端 run 仍在。

## train/ —— 各轮训练的 log.txt + trainer_state.json

`log.txt` 是 deepspeed 启动器完整 stdout（含完整命令行参数，可直接复原超参）；
`trainer_state.json` 是 HF Trainer 的逐 step 指标（loss、reward、KL、ratio 等）。
个别目录只有 `log.txt`（R2/R3 的 checkpoint 已被磁盘清理），逐 step 指标
在 `log.txt` 里同样有一份。

| 目录 | 轮次 | 一句话 | 文档 |
|---|---|---|---|
| `r1_grpo_rlaif/` | R1 | RLAIF-V + 规则奖励（keyword/length/no_repeat/state_token），奖励可提升但代理性弱 | GRPO_DEEP_DIVE §10 R1 |
| `r2_grpo_rlaif_judge/` | R2 | R1 + LLM Judge 奖励（Qwen2.5-3B 数字 token 期望打分） | GRPO_DEEP_DIVE §10 R2 |
| `r2_probe/` | R2 探针 | R2 的短探针跑（wandb 未登录报错的那次，日志留作排错记录） | — |
| `r3_mu3_aborted/` | R3（中止） | μ=3 样本复用在 RLAIF-V 上的尝试，run_name `grpo-rlaif-v-r3-mu3`，未跑完即转向 CLEVR | GRPO_DEEP_DIVE §9 μ 复用 |
| `r4_grpo_clevr/` | R4 | CLEVR 计数 + 可验证奖励（answer+format），held-out 44.6% → 77.4% | GRPO_DEEP_DIVE §10 R4 |
| `sft_clevr_control/` | SFT 对照 | 同数据预算的 LoRA SFT 对照臂（gold answer 监督） | GRPO_DEEP_DIVE §10「R4 后续验证」 |
| `r5_grpo_arm/` | R5 GRPO 臂 | SFT checkpoint 起点 + 二阶段不相交数据的 GRPO | GRPO_DEEP_DIVE §10 R5 |
| `r5_sft2_arm/` | R5 SFT2 臂 | 同起点同预算的继续 SFT 对照臂 | GRPO_DEEP_DIVE §10 R5 |
| `r6_beta0/` | R6 | β=0 消融：可验证奖励下去掉 KL 项，精度无差异 | GRPO_DEEP_DIVE §10 R6 |

## eval/ —— 评测原始结果

| 文件/目录 | 内容 |
|---|---|
| `clevr_baseline_gate.json` | CLEVR 基线闸门（训练前小样本快测） |
| `clevr_baseline_500.json` | 基座在 held-out 500 上的逐题结果（44.6% 的出处） |
| `r4_heldout.json` | R4 训练后 held-out 500 逐题对比（77.4% 的出处） |
| `bench_regression/` | 通用基准回归（VLMEvalKit 原始输出：MMBench/MME/POPE 的 xlsx+csv）。`baseline/` = 基座，`grpo/` = R1 模型（`compare_r1.txt`，"剂量不足、基准纹丝不动"结论的出处），`grpo_r4/` = R4 模型（`compare_r4.txt`，零回退结论的出处） |
| `phase2/` | 阶段一对照：base vs SFT（held-out）、base vs GRPO/SFT（SuperCLEVR OOD 200） |
| `r5/` | 阶段二对照：SFT 起点 vs GRPO 臂 / SFT2 臂，held-out + OOD 各一组 |
| `r6/` | R6 评测：base vs β=0 模型，held-out + OOD |

逐题 JSON 的格式：每条含 `question / gt / before_output / after_output / before_correct / after_correct`，
可直接用来复算准确率或做 McNemar 检验（工具见 `tools/eval_grpo_heldout.py`）。
