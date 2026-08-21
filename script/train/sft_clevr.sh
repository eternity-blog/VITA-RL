#!/bin/bash
# LoRA SFT on CLEVR counting: the CONTROL ARM for grpo_clevr.sh.
#
# Matched to the GRPO run on everything that defines the data budget:
#   - same 6,400 distinct prompts (sampled from the same train split,
#     held-out tail excluded -- tools/make_clevr_sft_data.py)
#   - same trainable capacity (LoRA r=64 alpha=16 on the LLM only;
#     vision tower / mm_projector / audio encoder frozen)
#   - same effective batch (16) and optimizer steps (400 = 1 epoch)
# and deliberately NOT matched on:
#   - supervision channel: SFT gets the gold answer verbatim, GRPO got
#     8 one-bit pass/fail signals per prompt
#   - lr: 1e-4, a standard LoRA-SFT value, ~20x the GRPO policy lr.
#     Crippling SFT with an RL-sized lr would rig the comparison.
#
# The dataset's solutions are direct answers ("<answer> N </answer>", no
# <think> chain), so SFT learns the answer format head-on. The question the
# comparison answers: given the same prompts and gold resource, is handing
# the answer over (SFT) or judging self-generated attempts (GRPO) the better
# teacher -- in-distribution and, more importantly, OOD (SuperCLEVR).
#
# Usage:
#   export VITA_CLEVR_SFT_DATA_DIR=/path/to/clevr_sft
#   export WEIGHTS_ROOT=/path/to/weights
#   bash script/train/sft_clevr.sh /path/to/output

set -euo pipefail

OUTPUT_DIR=${1:?"usage: $0 <output_dir>"}

WEIGHTS_ROOT=${WEIGHTS_ROOT:-${VITA_WEIGHTS:-}}
if [ -z "${WEIGHTS_ROOT}" ] && [ -z "${MODEL_PATH:-}" ]; then
    echo "error: set WEIGHTS_ROOT (or VITA_WEIGHTS)" >&2
    exit 1
fi
MODEL_PATH=${MODEL_PATH:-${WEIGHTS_ROOT}/VITA-1.5}
VISION_TOWER=${VISION_TOWER:-${WEIGHTS_ROOT}/InternViT-300M-448px}
AUDIO_ENCODER=${AUDIO_ENCODER:-${MODEL_PATH}/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning}

for p in "${MODEL_PATH}" "${VISION_TOWER}" "${AUDIO_ENCODER}"; do
    [ -d "${p}" ] || { echo "error: not a directory: ${p}" >&2; exit 1; }
done

if [ -z "${VITA_CLEVR_SFT_DATA_DIR:-}" ]; then
    echo "error: VITA_CLEVR_SFT_DATA_DIR is not set." >&2
    echo "       python tools/make_clevr_sft_data.py --parquet <shards> \\" >&2
    echo "           --image-dir <clevr_grpo>/images --out-dir <dir>" >&2
    exit 1
fi

GPUS=${GPUS:-2,3,4,6}
NGPU=$(awk -F, '{print NF}' <<< "${GPUS}")
GRAD_ACC=${GRAD_ACC:-4}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-1}
PORT=${MASTER_PORT:-29757}
WORKERS=${WORKERS:-0}
REPORT_TO=${REPORT_TO:-none}

echo "effective batch = ${NGPU} GPUs x 1 x ${GRAD_ACC} accum = $((NGPU * GRAD_ACC))"

OUTPUT_DIR_SFT=${OUTPUT_DIR}/sft-clevr
mkdir -p "${OUTPUT_DIR_SFT}"

export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

deepspeed --include "localhost:${GPUS}" --master_port "${PORT}" vita/train/train.py \
    --deepspeed ./script/deepspeed/zero2.json \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --model_name_or_path "${MODEL_PATH}" \
    --model_type qwen2p5_instruct \
    --version qwen2p5_instruct \
    --dataset_use CLEVRSFT \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --audio_encoder "${AUDIO_ENCODER}" \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter True \
    --tune_mm_mlp_adapter False \
    --freeze_mm_mlp_adapter True \
    --image_aspect_ratio square \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR_SFT}" \
    --num_train_epochs "${EPOCHS}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 1 \
    --learning_rate "${LR}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 6200 \
    --gradient_checkpointing True \
    --dataloader_num_workers "${WORKERS}" \
    --lazy_preprocess True \
    --report_to "${REPORT_TO}" \
    2>&1 | tee -a "${OUTPUT_DIR_SFT}/log.txt"

echo "Done. Adapter in ${OUTPUT_DIR_SFT}"
