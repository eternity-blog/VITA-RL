#!/bin/bash
# Smoke test for the training pipeline.
#
# This is NOT a reproduction of VITA-1.5 training. It runs a handful of
# optimizer steps on the tiny synthetic dataset produced by
# tools/make_smoke_data.py, purely to verify that the training path works end
# to end: data loading, multimodal token expansion, the collator, forward,
# backward, and checkpoint saving.
#
# Derived from script/train/finetuneTaskNeg_qwen.sh, with cluster-specific
# absolute paths replaced by environment variables.
#
# Usage:
#   export VITA_SMOKE_DATA_DIR=/path/to/smoke_data
#   bash script/train/smoke_test_qwen.sh /path/to/output [num_gpus]

set -euo pipefail

OUTPUT_DIR=${1:?"usage: $0 <output_dir> [num_gpus]"}
NUM_GPUS=${2:-1}

# Weight locations; override in the environment if yours differ.
WEIGHTS_ROOT=${WEIGHTS_ROOT:-/usr/local/kai/lx/weights}
MODEL_PATH=${MODEL_PATH:-${WEIGHTS_ROOT}/VITA-1.5}
VISION_TOWER=${VISION_TOWER:-${WEIGHTS_ROOT}/InternViT-300M-448px}
AUDIO_ENCODER=${AUDIO_ENCODER:-${MODEL_PATH}/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning}

if [ -z "${VITA_SMOKE_DATA_DIR:-}" ]; then
    echo "error: VITA_SMOKE_DATA_DIR is not set." >&2
    echo "       Generate the data first:" >&2
    echo "         python tools/make_smoke_data.py --out-dir <dir>" >&2
    echo "         export VITA_SMOKE_DATA_DIR=<dir>" >&2
    exit 1
fi

MODEL_TYPE=qwen2p5_instruct
OUTPUT_DIR_FT=${OUTPUT_DIR}/smoke-finetune_task_neg
mkdir -p "${OUTPUT_DIR_FT}"

export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))

deepspeed --include "localhost:${GPU_LIST}" vita/train/train.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path "${MODEL_PATH}" \
    --model_type $MODEL_TYPE \
    --version qwen2p5_instruct \
    --dataset_use SmokeTest \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --audio_encoder "${AUDIO_ENCODER}" \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter False \
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
    --learning_rate 1e-6 \
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
