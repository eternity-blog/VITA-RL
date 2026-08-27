# script/dev/ —— 开发机实验编排脚本（历史记录）

这些脚本是各轮实验在开发机上的**实际编排记录**：环境搭建、权重/数据下载、
训练启动、合并与评测的先后顺序和并行安排。开发机已回收，脚本里的绝对路径
（`/data/agent/lixiao29/...`、`/data/agent/conda/envs/vita-rl`）不再可用，
但编排逻辑（哪几步、什么顺序、哪些并行、判定条件）是文档数字的操作出处，
入库存档。

换机器复跑时：路径按 [ENVIRONMENT.md](../../ENVIRONMENT.md) 重建后替换；
`${DEV_HTTP_PROXY:-}` 是内网代理占位符，有代理需求时自行 export，无则留空。

| 脚本 | 作用 | 对应文档 |
|---|---|---|
| `setup_env_dev.sh` | conda 环境搭建（分步安装的自动化版） | ENVIRONMENT.md §2 |
| `setup_vlmeval_deps_dev.sh` | VLMEvalKit 依赖（--no-deps 陷阱的解法） | HANDBOOK.md §5 |
| `download_weights_dev.sh` | 三份权重下载 + config 本地化 | ENVIRONMENT.md §5 |
| `prefetch_lmudata_dev.sh` | 评测集 tsv 预下载（证书过期规避） | EXPERIMENT_LOG.md §11.6 |
| `prepare_rlaif_grpo_data_dev.sh` | RLAIF-V → GRPO 格式数据 | DATASETS.md §3.3 |
| `prepare_round2_dev.sh` | R2（judge 奖励）数据与 judge 模型准备 | GRPO_DEEP_DIVE.md §10 R2 |
| `run_grpo_eval_dev.sh` / `run_r2_eval_dev.sh` | R1/R2 训练后评测编排 | GRPO_DEEP_DIVE.md §10 |
| `run_r4_benchmark_dev.sh` | R4 通用基准回归（VLMEvalKit 三基准） | EXPERIMENT_LOG.md §14.1 |
| `run_phase2_dev.sh` | 阶段一对照：SFT 对照臂 + OOD 评测 | EXPERIMENT_LOG.md §14.1 |
| `run_r5_dev.sh` | R5 三臂对照全流程（数据切分→双臂训练→合并→四组评测） | EXPERIMENT_LOG.md §14.2 |
| `run_r6_beta0_dev.sh` / `run_r6_eval_dev.sh` | R6 β=0 消融训练与评测 | EXPERIMENT_LOG.md §14.3 |
