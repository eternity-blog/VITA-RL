#!/usr/bin/env bash
# GRPO 训练结束后的自动评测：
#   1. 把 LoRA adapter 合并进 VITA-1.5 基座（tools/merge_and_eval.py）
#   2. baseline 与 GRPO 各自在 POPE / MME / MMBench_DEV_EN_V11 上评测
#      （规则判分，无需 GPT judge；4 张空闲卡并行，POPE 最大单独占卡）
#   3. tools/compare_eval.py 对比并打印 1.96σ 噪声带
set -uo pipefail

REPO=/data/agent/lixiao29/VITA-RL-sync
W=/data/agent/lixiao29/vita-weights
ADAPTER=/data/agent/lixiao29/vita-outputs/grpo-rlaif-v-8gpu
MERGED=$W/VITA-1.5-grpo-rlaif
EVAL_OUT=$REPO/outputs/eval_grpo
PY=/data/agent/conda/envs/vita-rl/bin/python

export PYTHONPATH=$REPO
export LMUData=/data/agent/lixiao29/LMUData
export PATH=/data/agent/conda/envs/vita-rl/bin:$PATH
# 评测全程离线：数据集已预下载，权重全本地，不要让 opencompass 的过期证书搅局
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "$EVAL_OUT"

echo "== step 1: merge adapter -> $MERGED =="
if [ ! -f "$MERGED/config.json" ]; then
  "$PY" "$REPO/tools/merge_and_eval.py" \
    --base "$W/VITA-1.5" \
    --adapter "$ADAPTER" \
    --out "$MERGED" || { echo "MERGE FAILED"; exit 1; }
else
  echo "merged checkpoint already exists, skip"
fi

cd "$REPO/VLMEvalKit"

run_eval() {  # gpu ckpt tag datasets...
  local gpu=$1 ckpt=$2 tag=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu VITA_CKPT=$ckpt "$PY" run.py \
    --data "$@" --model vita_qwen2 \
    --work-dir "$EVAL_OUT/$tag" \
    > "$EVAL_OUT/${tag}_$(echo "$*" | tr ' ' '-').log" 2>&1
  echo "[done] $tag $* (exit $?)"
}

echo "== step 2: run benchmarks on 4 GPUs =="
run_eval 2 "$W/VITA-1.5"  baseline POPE &
run_eval 3 "$MERGED"      grpo     POPE &
run_eval 4 "$W/VITA-1.5"  baseline MME MMBench_DEV_EN_V11 &
run_eval 6 "$MERGED"      grpo     MME MMBench_DEV_EN_V11 &
wait

echo "== step 3: compare =="
"$PY" "$REPO/tools/compare_eval.py" \
  --before "$EVAL_OUT/baseline/vita_qwen2" \
  --after  "$EVAL_OUT/grpo/vita_qwen2" \
  2>&1 | tee "$EVAL_OUT/compare.txt"

echo "== GRPO EVAL DONE =="
