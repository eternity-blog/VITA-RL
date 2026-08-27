#!/usr/bin/env bash
# R5：阶段二对照实验 —— 同一 SFT checkpoint，同一批新 prompt，
#     监督信号形态是唯一变量。
#
#   Arm A（已有）: SFT-6.4k                     held-out 75.4% / OOD 63.0%
#   Arm C: A + 400 步 SFT   （新 6.4k，gold 直接监督，lr 1e-4，~26min）
#   Arm B: A + 400 步 GRPO  （同一批新 6.4k，verifier 信号，lr 5e-6，~3h15m）
#
# 配平：起点权重、阶段二 prompt 集合、步数、有效 batch、LoRA 容量。
# 不配平（方法自带）：lr、FLOPs（GRPO 每 prompt 8 rollouts）。
#
# 评测（before = SFT-A，after = 各 arm）：CLEVR held-out 500 + SuperCLEVR 200。
set -uo pipefail

REPO=/data/agent/lixiao29/VITA-RL-sync
W=/data/agent/lixiao29/vita-weights
SFT_MERGED=/data/agent/lixiao29/vita-outputs/sft_clevr/merged
OUT_B=/data/agent/lixiao29/vita-outputs/r5_grpo
OUT_C=/data/agent/lixiao29/vita-outputs/r5_sft2
EVAL_OUT=$REPO/outputs/eval_r5
PY=/data/agent/conda/envs/vita-rl/bin/python

export PYTHONPATH=$REPO
export PATH=/data/agent/conda/envs/vita-rl/bin:$PATH
export http_proxy=${DEV_HTTP_PROXY:-}
export https_proxy=${DEV_HTTP_PROXY:-}
export no_proxy=localhost,127.0.0.1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

mkdir -p "$EVAL_OUT" "$OUT_B" "$OUT_C"

echo "== step 0: stage-2 data (one sample, two formats) =="
if [ ! -f "$W/clevr_grpo_s2/clevr_grpo_train.json" ]; then
  "$PY" "$REPO/tools/make_clevr_stage2_data.py" \
    --parquet "$W/clevr_cogen_a_train/data/"'*.parquet' \
    --image-dir "$W/clevr_grpo/images" \
    --out-root "$W" --take 6400 --stage1-take 6400 --skip-tail 500 \
    2>&1 | tee "$EVAL_OUT/stage2_data.log" || exit 1
fi
[ -e "$W/clevr_sft_s2/images" ] || ln -s "$W/clevr_grpo/images" "$W/clevr_sft_s2/images"
[ -e "$W/clevr_grpo_s2/images" ] || ln -s "$W/clevr_grpo/images" "$W/clevr_grpo_s2/images"

echo "== step 1: Arm C -- continued SFT, 400 steps (~26min) =="
if ! ls "$OUT_C"/sft-clevr/adapter_model.* >/dev/null 2>&1 && \
   [ ! -f "$OUT_C/sft-clevr/adapter_config.json" ]; then
  cd "$REPO"
  MODEL_PATH=$SFT_MERGED \
  WEIGHTS_ROOT=$W \
  VITA_CLEVR_SFT_DATA_DIR=$W/clevr_sft_s2 \
  GPUS=2,3,4,6 GRAD_ACC=4 LR=1e-4 \
    bash script/train/sft_clevr.sh "$OUT_C" || { echo "ARM C TRAIN FAILED"; exit 1; }
fi

echo "== step 2: Arm B -- GRPO, 400 steps (~3h15m) =="
if ! ls "$OUT_B"/grpo-clevr/adapter_model.* >/dev/null 2>&1 && \
   [ ! -f "$OUT_B/grpo-clevr/adapter_config.json" ]; then
  cd "$REPO"
  MODEL_PATH=$SFT_MERGED \
  WEIGHTS_ROOT=$W \
  VITA_CLEVR_GRPO_DATA_DIR=$W/clevr_grpo_s2 \
  GPUS=2,3,4,6 GRAD_ACC=4 LR=5e-6 BETA=0.04 GROUP_SIZE=8 MAX_NEW=128 \
  MAX_STEPS=400 NUM_ITER=1 \
  REPORT_TO=wandb WANDB_PROJECT=vita-rl-grpo WANDB_NAME=grpo-clevr-r5-sft2rl \
    bash script/train/grpo_clevr.sh "$OUT_B" || { echo "ARM B TRAIN FAILED"; exit 1; }
fi

echo "== step 3: merge both adapters onto the SFT base =="
for pair in "B:$OUT_B/grpo-clevr:$OUT_B/merged" "C:$OUT_C/sft-clevr:$OUT_C/merged"; do
  arm=${pair%%:*}; rest=${pair#*:}; adapter=${rest%%:*}; merged=${rest#*:}
  if [ ! -f "$merged/config.json" ]; then
    "$PY" "$REPO/tools/merge_and_eval.py" \
      --base "$SFT_MERGED" --adapter "$adapter" --out "$merged" \
      > "$EVAL_OUT/merge_$arm.log" 2>&1 || { echo "MERGE $arm FAILED"; tail -5 "$EVAL_OUT/merge_$arm.log"; exit 1; }
  fi
done

echo "== step 4: four evals in parallel (before = SFT-A) =="
unset http_proxy https_proxy 2>/dev/null || true
cd "$REPO"
run_eval () { # gpu, after, data, image_root, tag
  ( CUDA_VISIBLE_DEVICES=$1 "$PY" tools/eval_grpo_heldout.py \
      --before "$SFT_MERGED" --after "$2" \
      --data "$3" --image-root "$4" \
      --out "$EVAL_OUT/$5.json" \
      > "$EVAL_OUT/$5.log" 2>&1
    echo "[done] $5 (exit $?)" ) &
}
run_eval 2 "$OUT_B/merged" "$W/clevr_grpo/clevr_grpo_heldout.json" "$W/clevr_grpo/images" heldout_sft_vs_grpo
run_eval 3 "$OUT_B/merged" "$W/superclevr_eval/superclevr_test.json" "$W/superclevr_eval/images" superclevr_sft_vs_grpo
run_eval 4 "$OUT_C/merged" "$W/clevr_grpo/clevr_grpo_heldout.json" "$W/clevr_grpo/images" heldout_sft_vs_sft2
run_eval 6 "$OUT_C/merged" "$W/superclevr_eval/superclevr_test.json" "$W/superclevr_eval/images" superclevr_sft_vs_sft2
wait

echo "== summaries =="
for f in heldout_sft_vs_grpo superclevr_sft_vs_grpo heldout_sft_vs_sft2 superclevr_sft_vs_sft2; do
  echo "---- $f ----"
  tail -8 "$EVAL_OUT/$f.log"
done

echo "== R5 DONE =="
touch "$EVAL_OUT/R5_DONE"
