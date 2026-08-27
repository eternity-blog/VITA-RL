#!/usr/bin/env bash
# R4（CLEVR GRPO）通用能力回归：POPE / MME / MMBench_DEV_EN_V11
# 目的：确认 +32.8pt 的专项提升没有以通用能力退化为代价。
# baseline 数字复用 R1 轮已跑好的 outputs/eval_grpo/baseline。
set -uo pipefail

REPO=/data/agent/lixiao29/VITA-RL-sync
W=/data/agent/lixiao29/vita-weights
MERGED=/data/agent/lixiao29/vita-outputs/r4/merged
EVAL_OUT=$REPO/outputs/eval_r4_bench
PY=/data/agent/conda/envs/vita-rl/bin/python

export PYTHONPATH=$REPO
export LMUData=/data/agent/lixiao29/LMUData
export PATH=/data/agent/conda/envs/vita-rl/bin:$PATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "$EVAL_OUT"

[ -f "$MERGED/config.json" ] || { echo "FATAL: merged model missing at $MERGED"; exit 1; }

echo "== r4 benchmark regression: POPE (gpu2) | MME+MMBench (gpu3) =="
cd "$REPO/VLMEvalKit"
( CUDA_VISIBLE_DEVICES=2 VITA_CKPT=$MERGED "$PY" run.py \
    --data POPE --model vita_qwen2 --work-dir "$EVAL_OUT/grpo_r4" \
    > "$EVAL_OUT/r4_POPE.log" 2>&1
  echo "[done] POPE (exit $?)" ) &
( CUDA_VISIBLE_DEVICES=3 VITA_CKPT=$MERGED "$PY" run.py \
    --data MME MMBench_DEV_EN_V11 --model vita_qwen2 --work-dir "$EVAL_OUT/grpo_r4" \
    > "$EVAL_OUT/r4_MME-MMBench.log" 2>&1
  echo "[done] MME+MMBench (exit $?)" ) &
wait

echo "== summarize vs baseline =="
"$PY" "$REPO/tools/compare_eval.py" \
  --before "$REPO/outputs/eval_grpo/baseline/vita_qwen2" \
  --after  "$EVAL_OUT/grpo_r4/vita_qwen2" \
  2>&1 | tee "$EVAL_OUT/compare.txt"

echo "== R4 BENCHMARK REGRESSION DONE =="
touch "$EVAL_OUT/PHASE1_DONE"
