#!/bin/bash
# DPO on real preference data (RLAIF-V), single GPU, LoRA.
#
# The smoke test in dpo_smoke_test.sh proves the code path on 24 synthetic
# pairs; this one trains on real AI-feedback preferences and produces an
# adapter worth evaluating. The differences from the smoke script are all
# about it being a real run rather than a wiring check:
#
#   - saves a checkpoint (the smoke test uses --save_strategy no)
#   - a cosine schedule over the whole set rather than 24 steps
#   - gradient accumulation, so the effective batch is 16 pairs rather than 1
#   - lora_dropout back on: the -log(0.5) check is a startup assertion, and
#     regularisation matters more than keeping it exact past step 1
#
# The first-step loss should still be ~0.693 even with dropout, just not to
# four decimals. If it is far off, the reference policy is wired up wrong --
# stop and fix that before reading anything into the results.
#
# Usage:
#   export VITA_RLAIF_DATA_DIR=/path/to/rlaif_v_dpo
#   export WEIGHTS_ROOT=/path/to/weights
#   bash script/train/dpo_rlaif_v.sh /path/to/output [gpu_id]

set -euo pipefail

OUTPUT_DIR=${1:?"usage: $0 <output_dir> [gpu_id]"}
GPU_ID=${2:-0}

WEIGHTS_ROOT=${WEIGHTS_ROOT:-${VITA_WEIGHTS:-}}
if [ -z "${WEIGHTS_ROOT}" ] && [ -z "${MODEL_PATH:-}" ]; then
    echo "error: set WEIGHTS_ROOT (or VITA_WEIGHTS) to the directory holding" >&2
    echo "       VITA-1.5/ and InternViT-300M-448px/" >&2
    exit 1
fi
MODEL_PATH=${MODEL_PATH:-${WEIGHTS_ROOT}/VITA-1.5}
VISION_TOWER=${VISION_TOWER:-${WEIGHTS_ROOT}/InternViT-300M-448px}
AUDIO_ENCODER=${AUDIO_ENCODER:-${MODEL_PATH}/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning}

for p in "${MODEL_PATH}" "${VISION_TOWER}" "${AUDIO_ENCODER}"; do
    if [ ! -d "${p}" ]; then
        echo "error: not a directory: ${p}" >&2
        exit 1
    fi
done

if [ -z "${VITA_RLAIF_DATA_DIR:-}" ]; then
    echo "error: VITA_RLAIF_DATA_DIR is not set." >&2
    echo "       python tools/make_rlaif_v_data.py --parquet <shard> --out-dir <dir>" >&2
    exit 1
fi

EPOCHS=${EPOCHS:-1}
LR=${LR:-5e-6}
BETA=${BETA:-0.1}
GRAD_ACC=${GRAD_ACC:-16}

MODEL_TYPE=qwen2p5_instruct
OUTPUT_DIR_DPO=${OUTPUT_DIR}/dpo-rlaif-v
mkdir -p "${OUTPUT_DIR_DPO}"

export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Batch size 1 is one preference pair = two sequences through the policy plus
# two under no_grad for the reference. The accumulation steps are what make
# the effective batch reasonable for a 7B.
deepspeed --include "localhost:${GPU_ID}" vita/train/train_dpo.py \
    --deepspeed ./script/deepspeed/zero2.json \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --dpo_beta "${BETA}" \
    --model_name_or_path "${MODEL_PATH}" \
    --model_type $MODEL_TYPE \
    --version qwen2p5_instruct \
    --dataset_use RLAIFV \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --audio_encoder "${AUDIO_ENCODER}" \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter True \
    --image_aspect_ratio square \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR_DPO}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 2 \
    --learning_rate "${LR}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 6200 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee -a "${OUTPUT_DIR_DPO}/log.txt"

echo "Done. Adapter in ${OUTPUT_DIR_DPO}"
