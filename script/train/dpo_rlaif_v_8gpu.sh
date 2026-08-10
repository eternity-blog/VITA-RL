#!/bin/bash
# Scaled-up DPO on RLAIF-V: more data, larger effective batch, 8 GPUs.
#
# The 3000-pair run moved rewards/accuracy from 0.49 to 0.52 and changed no
# benchmark. The probe explained why: the base model separates these pairs
# with a signal-to-noise ratio of 0.11, so a single pair's gradient is mostly
# noise. This script attacks that on both axes at once, because they are
# different mechanisms:
#
#   more pairs      -- accumulated signal grows as sqrt(n), so 5x the data is
#                      about 2.2x the signal. Necessary but slow.
#   larger batch    -- averaging G pairs before the step cuts the noise in
#                      that step by sqrt(G). This is the lever the previous
#                      run left unused: effective batch 16 against an SNR of
#                      0.11 gives a per-step SNR of only 0.44.
#
# 8 GPUs x batch 1 x 8 accumulation = effective batch 64, four times the
# previous run, for a per-step SNR near 0.9. With 15000 pairs that is 234
# optimizer steps -- fewer than before, but each one far less noisy.
#
# A larger batch also permits a larger learning rate. 2e-5 was already the
# better of the two settings at batch 16; keeping it at batch 64 is
# conservative relative to the usual linear-scaling heuristic.
#
# Usage:
#   export VITA_RLAIF_DATA_DIR=/path/to/rlaif_v_dpo_large
#   export WEIGHTS_ROOT=/path/to/weights
#   bash script/train/dpo_rlaif_v_8gpu.sh /path/to/output

set -euo pipefail

OUTPUT_DIR=${1:?"usage: $0 <output_dir>"}

WEIGHTS_ROOT=${WEIGHTS_ROOT:-${VITA_WEIGHTS:-}}
if [ -z "${WEIGHTS_ROOT}" ] && [ -z "${MODEL_PATH:-}" ]; then
    echo "error: set WEIGHTS_ROOT (or VITA_WEIGHTS)" >&2
    exit 1
fi
MODEL_PATH=${MODEL_PATH:-${WEIGHTS_ROOT}/VITA-1.5}
VISION_TOWER=${VISION_TOWER:-${WEIGHTS_ROOT}/InternViT-300M-448px}
# Default to the audio encoder shipped inside MODEL_PATH, but fall back to the
# released base's copy. A checkpoint produced by our own SFT keeps the frozen
# encoders as absolute paths in its config rather than as subdirectories, so
# deriving this from MODEL_PATH fails for exactly the SFT -> DPO chain this
# script exists to support.
if [ -z "${AUDIO_ENCODER:-}" ]; then
    AUDIO_ENCODER=${MODEL_PATH}/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning
    if [ ! -d "${AUDIO_ENCODER}" ]; then
        AUDIO_ENCODER=${WEIGHTS_ROOT}/VITA-1.5/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning
    fi
fi

for p in "${MODEL_PATH}" "${VISION_TOWER}" "${AUDIO_ENCODER}"; do
    [ -d "${p}" ] || { echo "error: not a directory: ${p}" >&2; exit 1; }
done

if [ -z "${VITA_RLAIF_DATA_DIR:-}" ]; then
    echo "error: VITA_RLAIF_DATA_DIR is not set." >&2
    exit 1
fi

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NGPU=$(awk -F, '{print NF}' <<< "${GPUS}")
GRAD_ACC=${GRAD_ACC:-8}
LR=${LR:-2e-5}
BETA=${BETA:-0.1}
EPOCHS=${EPOCHS:-1}
PORT=${MASTER_PORT:-29677}

# Dataloader workers are shared-memory hungry: each one ships decoded 448x448
# tiles back through /dev/shm. One rank with 4 workers is fine; N ranks with 4
# each is not, and on this box /dev/shm is 512 MB and cannot be remounted
# without privileges. The failure is "DataLoader worker killed by signal: Bus
# error" partway into the first step.
#
# 0 means load in the training process, using no shared memory at all. It
# costs some throughput, but the step here is four 7B forwards -- data loading
# is not the bottleneck. Raise WORKERS if your /dev/shm is large.
WORKERS=${WORKERS:-0}

echo "effective batch = ${NGPU} GPUs x 1 x ${GRAD_ACC} accum = $((NGPU * GRAD_ACC))"
echo "/dev/shm: $(df -h /dev/shm | awk 'NR==2{print $2}'), dataloader workers: ${WORKERS}"

OUTPUT_DIR_DPO=${OUTPUT_DIR}/dpo-rlaif-v-8gpu
mkdir -p "${OUTPUT_DIR_DPO}"

export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# zero2 rather than zero3: the policy and reference are the same weights with
# the adapter toggled, and zero3's parameter sharding makes disable_adapter()
# considerably more delicate. At LoRA's memory profile (~28 GB/GPU here) there
# is no need for zero3.
deepspeed --include "localhost:${GPUS}" --master_port "${PORT}" vita/train/train_dpo.py \
    --deepspeed ./script/deepspeed/zero2.json \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --dpo_beta "${BETA}" \
    --model_name_or_path "${MODEL_PATH}" \
    --model_type qwen2p5_instruct \
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
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate "${LR}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --tf32 True \
    --model_max_length 6200 \
    --gradient_checkpointing True \
    --dataloader_num_workers "${WORKERS}" \
    --lazy_preprocess True \
    --report_to none \
    2>&1 | tee -a "${OUTPUT_DIR_DPO}/log.txt"

echo "Done. Adapter in ${OUTPUT_DIR_DPO}"
