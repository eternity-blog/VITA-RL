#!/usr/bin/env bash
# R6（beta=0 消融）训练后评测：
#   1. 合并 R6 GRPO adapter
#   2. 双路并行：
#      GPU2  held-out 500：base vs R6（对比 R4 的 77.4%）
#      GPU3  SuperCLEVR OOD 200：base vs R6（对比 R4 的 54.5%）
set -uo pipefail

REPO=/data/agent/lixiao29/VITA-RL-sync
W=/data/agent/lixiao29/vita-weights
R6_OUT=/data/agent/lixiao29/vita-outputs/r6_beta0
EVAL_OUT=$REPO/outputs/eval_r6
PY=/data/agent/conda/envs/vita-rl/bin/python

export PYTHONPATH=$REPO
export PATH=/data/agent/conda/envs/vita-rl/bin:$PATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "$EVAL_OUT"

echo "== step 1: merge R6 adapter =="
R6_MERGED=$R6_OUT/merged
if [ ! -f "$R6_MERGED/config.json" ]; then
  "$PY" "$REPO/tools/merge_and_eval.py" \
    --base "$W/VITA-1.5" \
    --adapter "$R6_OUT/grpo-clevr" \
    --out "$R6_MERGED" > "$EVAL_OUT/merge.log" 2>&1 || { echo "MERGE FAILED"; tail -5 "$EVAL_OUT/merge.log"; exit 1; }
fi
echo "merge done"

echo "== step 2: evals in parallel =="
cd "$REPO"
( CUDA_VISIBLE_DEVICES=2 "$PY" tools/eval_grpo_heldout.py \
    --before "$W/VITA-1.5" --after "$R6_MERGED" \
    --data "$W/clevr_grpo/clevr_grpo_heldout.json" \
    --image-root "$W/clevr_grpo/images" \
    --out "$EVAL_OUT/heldout_base_vs_r6.json" \
    > "$EVAL_OUT/heldout_base_vs_r6.log" 2>&1
  echo "[done] heldout base-vs-r6 (exit $?)" ) &
( CUDA_VISIBLE_DEVICES=3 "$PY" tools/eval_grpo_heldout.py \
    --before "$W/VITA-1.5" --after "$R6_MERGED" \
    --data "$W/superclevr_eval/superclevr_test.json" \
    --image-root "$W/superclevr_eval/images" \
    --out "$EVAL_OUT/superclevr_base_vs_r6.json" \
    > "$EVAL_OUT/superclevr_base_vs_r6.log" 2>&1
  echo "[done] superclevr base-vs-r6 (exit $?)" ) &
wait

echo "== summaries =="
for f in heldout_base_vs_r6 superclevr_base_vs_r6; do
  echo "---- $f ----"
  tail -15 "$EVAL_OUT/$f.log"
done

echo "== R6 EVAL DONE =="
touch "$EVAL_OUT/R6_EVAL_DONE"
