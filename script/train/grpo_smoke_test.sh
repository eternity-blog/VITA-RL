#!/bin/bash
# Single-GPU GRPO smoke test.
#
# Group Relative Policy Optimization on synthetic text-only prompts. Like the
# other smoke tests this verifies the machinery, not the model: the rewards
# are simple rules, so the point is that the reward mean moves at all.
#
# What to look for:
#
#   grpo/kl at step 1     ~= 0. The policy starts equal to the reference, so
#                          any real KL means the reference is wired up wrong.
#   reward/mean           should trend upward -- the core GRPO signal
#   grpo/advantage_std    ~= 1 on live groups (normalisation working)
#   groups/degenerate_frac  how often every rollout in a group scored the
#                          same. Near 1.0 means the reward cannot tell the
#                          samples apart and almost nothing is training.
#   grpo/ratio            ~= 1 on the first inner step by construction
#
# Batch size is 1 prompt x 8 rollouts = 8 sequences through the model per
# step, plus 8 more under no_grad for the reference, plus the sampling pass.
#
# Usage:
#   export VITA_GRPO_DATA_DIR=/path/to/grpo_smoke_data
#   export WEIGHTS_ROOT=/path/to/weights
#   bash script/train/grpo_smoke_test.sh /path/to/output [gpu_id]

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

if [ -z "${VITA_GRPO_DATA_DIR:-}" ]; then
    echo "error: VITA_GRPO_DATA_DIR is not set." >&2
    echo "       python tools/make_grpo_smoke_data.py --out-dir <dir>" >&2
    exit 1
fi

MODEL_TYPE=qwen2p5_instruct
OUTPUT_DIR_GRPO=${OUTPUT_DIR}/smoke-grpo
mkdir -p "${OUTPUT_DIR_GRPO}"

export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# lora_dropout 0 for the same reason as the DPO script: the reference pass
# runs with the adapter disabled and therefore without dropout, so dropout
# on the policy side would make the step-1 KL nonzero for no good reason.
deepspeed --include "localhost:${GPU_ID}" vita/train/train_grpo.py \
    --deepspeed ./script/deepspeed/zero2.json \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --grpo_group_size 8 \
    --grpo_beta 0.04 \
    --grpo_clip_eps 0.2 \
    --grpo_max_new_tokens 96 \
    --grpo_temperature 1.0 \
    --grpo_top_p 0.95 \
    --reward_fns "keyword:1.0,length:0.5,no_repeat:0.5,state_token:0.5" \
    --model_name_or_path "${MODEL_PATH}" \
    --model_type $MODEL_TYPE \
    --version qwen2p5_instruct \
    --dataset_use GRPOSmokeTest \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --audio_encoder "${AUDIO_ENCODER}" \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter True \
    --image_aspect_ratio square \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR_GRPO}" \
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
    2>&1 | tee -a "${OUTPUT_DIR_GRPO}/log.txt"

echo "Done."
