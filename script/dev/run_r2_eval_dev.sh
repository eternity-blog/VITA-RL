#!/usr/bin/env bash
# GRPO 第二轮训练结束后的自动评测：
#   1. 合并 r2 adapter -> VITA-1.5-grpo-r2
#   2. 三路并行：
#      GPU2  held-out 500 条生成式对比（keyword 召回 / judge 分 / win rate）  <- 主效果指标
#      GPU3  POPE（r2 模型；baseline 结果复用第一轮的 outputs/eval_grpo/baseline）
#      GPU4  MME + MMBench_DEV_EN_V11（r2 模型）
#   3. compare_eval.py 汇总
set -uo pipefail

REPO=/data/agent/lixiao29/VITA-RL-sync
W=/data/agent/lixiao29/vita-weights
ADAPTER=/data/agent/lixiao29/vita-outputs/r2/grpo-rlaif-v-8gpu
MERGED=$W/VITA-1.5-grpo-r2
EVAL_OUT=$REPO/outputs/eval_r2
DATA=$W/rlaif_v_grpo_full
PY=/data/agent/conda/envs/vita-rl/bin/python

export PYTHONPATH=$REPO
export LMUData=/data/agent/lixiao29/LMUData
export PATH=/data/agent/conda/envs/vita-rl/bin:$PATH
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

echo "== step 2: heldout + benchmarks in parallel =="
( CUDA_VISIBLE_DEVICES=2 "$PY" "$REPO/tools/eval_grpo_heldout.py" \
    --before "$W/VITA-1.5" --after "$MERGED" \
    --data "$DATA/rlaif_v_grpo_eval.json" \
    --image-root "$DATA/images" \
    --judge "$W/Qwen2.5-3B-Instruct" \
    --out "$EVAL_OUT/heldout_results.json" \
    > "$EVAL_OUT/heldout.log" 2>&1
  echo "[done] heldout (exit $?)" ) &

cd "$REPO/VLMEvalKit"
( CUDA_VISIBLE_DEVICES=3 VITA_CKPT=$MERGED "$PY" run.py \
    --data POPE --model vita_qwen2 --work-dir "$EVAL_OUT/grpo_r2" \
    > "$EVAL_OUT/r2_POPE.log" 2>&1
  echo "[done] POPE (exit $?)" ) &
( CUDA_VISIBLE_DEVICES=4 VITA_CKPT=$MERGED "$PY" run.py \
    --data MME MMBench_DEV_EN_V11 --model vita_qwen2 --work-dir "$EVAL_OUT/grpo_r2" \
    > "$EVAL_OUT/r2_MME-MMBench.log" 2>&1
  echo "[done] MME+MMBench (exit $?)" ) &
wait

echo "== step 3: summarize =="
"$PY" "$REPO/tools/compare_eval.py" \
  --before "$REPO/outputs/eval_grpo/baseline/vita_qwen2" \
  --after  "$EVAL_OUT/grpo_r2/vita_qwen2" \
  2>&1 | tee "$EVAL_OUT/compare.txt"
echo "---- heldout ----"
tail -12 "$EVAL_OUT/heldout.log"

echo "== R2 EVAL DONE =="
