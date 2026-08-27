#!/usr/bin/env bash
# Phase 2（在 phase 1 通用回归完成后自动启动）：
#   1. SuperCLEVR test200 -> OOD 评测集（jigsaw-r1/super_clevr，已下载 parquet）
#   2. CLEVR SFT 对照数据：与 R4 GRPO 相同 6,400 个 prompt + gold solution
#   3. LoRA SFT 训练（4 卡，400 步，与 GRPO 同 batch/容量）
#   4. 合并 SFT adapter
#   5. 三路并行评测：
#      GPU2  held-out 500：base vs SFT（answer_accuracy + win rate）
#      GPU3  SuperCLEVR OOD：base vs GRPO(R4)
#      GPU4  SuperCLEVR OOD：base vs SFT
set -uo pipefail

REPO=/data/agent/lixiao29/VITA-RL-sync
W=/data/agent/lixiao29/vita-weights
R4_MERGED=/data/agent/lixiao29/vita-outputs/r4/merged
SFT_OUT=/data/agent/lixiao29/vita-outputs/sft_clevr
EVAL_OUT=$REPO/outputs/eval_phase2
PY=/data/agent/conda/envs/vita-rl/bin/python
MARKER=$REPO/outputs/eval_r4_bench/PHASE1_DONE

export PYTHONPATH=$REPO
export PATH=/data/agent/conda/envs/vita-rl/bin:$PATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "$EVAL_OUT"

echo "== waiting for phase 1 marker: $MARKER =="
for i in $(seq 1 240); do
  [ -f "$MARKER" ] && break
  sleep 60
done
[ -f "$MARKER" ] || { echo "FATAL: phase 1 did not finish within 4h"; exit 1; }
echo "phase 1 done, starting phase 2 at $(date)"

echo "== step 1: SuperCLEVR OOD eval data =="
if [ ! -f "$W/superclevr_eval/superclevr_test.json" ]; then
  "$PY" "$REPO/tools/make_superclevr_eval_data.py" \
    --parquet "$W/superclevr_raw/test.parquet" \
    --out-dir "$W/superclevr_eval" || exit 1
fi

echo "== step 2: CLEVR SFT control data (6400 prompts, seed 42) =="
if [ ! -f "$W/clevr_sft/clevr_sft_train.json" ]; then
  "$PY" "$REPO/tools/make_clevr_sft_data.py" \
    --parquet "$W/clevr_cogen_a_train/data/"'*.parquet' \
    --image-dir "$W/clevr_grpo/images" \
    --out-dir "$W/clevr_sft" --take 6400 --skip-tail 500 || exit 1
fi
[ -e "$W/clevr_sft/images" ] || ln -s "$W/clevr_grpo/images" "$W/clevr_sft/images"

echo "== step 3: LoRA SFT training (GPUs 2,3,4,6; 400 steps) =="
if [ ! -f "$SFT_OUT/sft-clevr/adapter_config.json" ] && \
   [ ! -f "$SFT_OUT/sft-clevr/adapter_model.safetensors" ]; then
  cd "$REPO"
  VITA_CLEVR_SFT_DATA_DIR=$W/clevr_sft \
  WEIGHTS_ROOT=$W \
  GPUS=2,3,4,6 GRAD_ACC=4 \
    bash script/train/sft_clevr.sh "$SFT_OUT" || { echo "SFT TRAIN FAILED"; exit 1; }
fi

echo "== step 4: merge SFT adapter =="
SFT_MERGED=$SFT_OUT/merged
if [ ! -f "$SFT_MERGED/config.json" ]; then
  "$PY" "$REPO/tools/merge_and_eval.py" \
    --base "$W/VITA-1.5" \
    --adapter "$SFT_OUT/sft-clevr" \
    --out "$SFT_MERGED" > "$EVAL_OUT/merge.log" 2>&1 || { echo "MERGE FAILED"; tail -5 "$EVAL_OUT/merge.log"; exit 1; }
fi

echo "== step 5: evals in parallel =="
cd "$REPO"
( CUDA_VISIBLE_DEVICES=2 "$PY" tools/eval_grpo_heldout.py \
    --before "$W/VITA-1.5" --after "$SFT_MERGED" \
    --data "$W/clevr_grpo/clevr_grpo_heldout.json" \
    --image-root "$W/clevr_grpo/images" \
    --out "$EVAL_OUT/heldout_base_vs_sft.json" \
    > "$EVAL_OUT/heldout_base_vs_sft.log" 2>&1
  echo "[done] heldout base-vs-sft (exit $?)" ) &
( CUDA_VISIBLE_DEVICES=3 "$PY" tools/eval_grpo_heldout.py \
    --before "$W/VITA-1.5" --after "$R4_MERGED" \
    --data "$W/superclevr_eval/superclevr_test.json" \
    --image-root "$W/superclevr_eval/images" \
    --out "$EVAL_OUT/superclevr_base_vs_grpo.json" \
    > "$EVAL_OUT/superclevr_base_vs_grpo.log" 2>&1
  echo "[done] superclevr base-vs-grpo (exit $?)" ) &
( CUDA_VISIBLE_DEVICES=4 "$PY" tools/eval_grpo_heldout.py \
    --before "$W/VITA-1.5" --after "$SFT_MERGED" \
    --data "$W/superclevr_eval/superclevr_test.json" \
    --image-root "$W/superclevr_eval/images" \
    --out "$EVAL_OUT/superclevr_base_vs_sft.json" \
    > "$EVAL_OUT/superclevr_base_vs_sft.log" 2>&1
  echo "[done] superclevr base-vs-sft (exit $?)" ) &
wait

echo "== summaries =="
for f in heldout_base_vs_sft superclevr_base_vs_grpo superclevr_base_vs_sft; do
  echo "---- $f ----"
  tail -15 "$EVAL_OUT/$f.log"
done

echo "== PHASE 2 DONE =="
touch "$EVAL_OUT/PHASE2_DONE"
