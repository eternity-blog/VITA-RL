#!/usr/bin/env bash
# R6：KL 消融 —— 与 R4 完全同配置（裸基座起点、stage-1 CLEVR 数据、
#     GPU 2,3,4,6、GRAD_ACC 4、LR 5e-6、400 步），唯一变量 BETA 0.04 → 0。
#
# 动机：R4 里 KL 涨 6 倍但准确率同步涨（leash 没卡住学习也没证据帮忙）；
# DAPO 的立场是可验证奖励下 KL 只会拖后腿。本 run 用实测回答：
#   - 去掉 KL 罚项后 44.6%→77.4% 的收益是否复现/更快？
#   - grpo/kl 指标继续记录（k3 无条件计算），只是不进 loss——
#     看自由漂移到什么量级、是否伴随格式/reward 崩坏。
#
# wandb：不设 WANDB_PROJECT，落 HF Trainer 默认项目 `huggingface`，
# 与 r4（tvyusoul）同项目并排对比。
set -uo pipefail

REPO=/data/agent/lixiao29/VITA-RL-sync
W=/data/agent/lixiao29/vita-weights
OUT=/data/agent/lixiao29/vita-outputs/r6_beta0

export PYTHONPATH=$REPO
export PATH=/data/agent/conda/envs/vita-rl/bin:$PATH
export http_proxy=${DEV_HTTP_PROXY:-}
export https_proxy=${DEV_HTTP_PROXY:-}
export no_proxy=localhost,127.0.0.1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "$OUT"
cd "$REPO"

if ls "$OUT"/grpo-clevr/adapter_model.* >/dev/null 2>&1 || \
   [ -f "$OUT/grpo-clevr/adapter_config.json" ]; then
  echo "adapter already exists in $OUT/grpo-clevr, refusing to overwrite"; exit 0
fi

WEIGHTS_ROOT=$W \
VITA_CLEVR_GRPO_DATA_DIR=$W/clevr_grpo \
GPUS=2,3,4,6 GRAD_ACC=4 LR=5e-6 BETA=0 GROUP_SIZE=8 MAX_NEW=128 \
MAX_STEPS=400 NUM_ITER=1 \
REPORT_TO=wandb WANDB_NAME=grpo-clevr-r6-beta0 \
  bash script/train/grpo_clevr.sh "$OUT" || { echo "R6 TRAIN FAILED"; exit 1; }

echo "R6 DONE. Adapter in $OUT/grpo-clevr"
