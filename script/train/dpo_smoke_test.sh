#!/bin/bash
# Single-GPU DPO smoke test.
#
# Runs offline Direct Preference Optimization on the synthetic preference
# pairs from tools/make_dpo_smoke_data.py. Like the other smoke tests this
# verifies the code path, not the model: the preferences are trivially
# separable, so the point is that the reward margin moves at all.
#
# What to look for in the output:
#
#   step 1 loss ~= 0.6931   the reference equals the initial policy, so the
#                           DPO logit is zero and the loss is -log(0.5).
#                           A different value means the reference model is
#                           wired up wrong -- that is the sharpest check
#                           available and it is free.
#   rewards/margin          should climb above zero
#   rewards/accuracy        should rise toward 1.0
#
# lora_dropout is 0 here on purpose. With dropout on, the policy pass is
# stochastic while the reference pass (adapter disabled) is not, so the two
# disagree even at initialisation and the -log(0.5) check goes soft. Turn it
# back on for real training if you want the regularisation.
#
# Usage:
#   export VITA_DPO_DATA_DIR=/path/to/dpo_smoke_data
#   export WEIGHTS_ROOT=/path/to/weights
#   bash script/train/dpo_smoke_test.sh /path/to/output [gpu_id]

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

if [ -z "${VITA_DPO_DATA_DIR:-}" ]; then
    echo "error: VITA_DPO_DATA_DIR is not set." >&2
    echo "       python tools/make_dpo_smoke_data.py --out-dir <dir>" >&2
    exit 1
fi

MODEL_TYPE=qwen2p5_instruct
OUTPUT_DIR_DPO=${OUTPUT_DIR}/smoke-dpo
mkdir -p "${OUTPUT_DIR_DPO}"

export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Batch size 1 means one preference pair, i.e. two sequences through the
# model per step, plus two more under no_grad for the reference.
deepspeed --include "localhost:${GPU_ID}" vita/train/train_dpo.py \
    --deepspeed ./script/deepspeed/zero2.json \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --dpo_beta 0.1 \
    --model_name_or_path "${MODEL_PATH}" \
    --model_type $MODEL_TYPE \
    --version qwen2p5_instruct \
    --dataset_use DPOSmokeTest \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --audio_encoder "${AUDIO_ENCODER}" \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter True \
    --image_aspect_ratio square \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR_DPO}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
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
    2>&1 | tee -a "${OUTPUT_DIR_DPO}/log.txt"

echo "Done."
