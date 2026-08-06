#!/bin/bash
# Single-GPU LoRA smoke test.
#
# Full-parameter 7B training needs ~98 GB of optimizer state and does not fit
# on one 80 GB card (see REPRODUCE.md#memory-note). LoRA trains a small set of
# adapter matrices instead, which does fit. Upstream ships the LoRA code path
# in train.py but no script that uses it, so this is the first thing that
# exercises it.
#
# Known caveat: find_all_linear_names (train.py:157) excludes mm_projector,
# vision_tower and vision_resampler, but NOT audio_encoder -- so LoRA also
# attaches to whale's Linear layers even with --freeze_audio_encoder True.
# The run below prints the resolved target modules so you can see what
# actually got adapted.
#
# Usage:
#   export VITA_SMOKE_DATA_DIR=/path/to/smoke_data
#   export WEIGHTS_ROOT=/path/to/weights
#   bash script/train/smoke_test_lora.sh /path/to/output [gpu_id]

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

if [ -z "${VITA_SMOKE_DATA_DIR:-}" ]; then
    echo "error: VITA_SMOKE_DATA_DIR is not set." >&2
    echo "       python tools/make_smoke_data.py --out-dir <dir>" >&2
    exit 1
fi

MODEL_TYPE=qwen2p5_instruct
OUTPUT_DIR_FT=${OUTPUT_DIR}/smoke-lora
mkdir -p "${OUTPUT_DIR_FT}"

export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ZeRO-2 rather than ZeRO-3: with LoRA the trainable set is small, so sharding
# parameters buys little and ZeRO-3's gather on save complicates the adapter
# dump. One GPU either way.
deepspeed --include "localhost:${GPU_ID}" vita/train/train.py \
    --deepspeed ./script/deepspeed/zero2.json \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --model_name_or_path "${MODEL_PATH}" \
    --model_type $MODEL_TYPE \
    --version qwen2p5_instruct \
    --dataset_use SmokeTest \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --audio_encoder "${AUDIO_ENCODER}" \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter True \
    --image_aspect_ratio square \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR_FT}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 6200 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee -a "${OUTPUT_DIR_FT}/log.txt"

echo "Done."
