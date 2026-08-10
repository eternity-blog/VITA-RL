#!/bin/bash
# Full-parameter SFT on RLAIF-V's chosen responses. 8 GPUs, ZeRO-3.
#
# Why SFT after four rounds of DPO
# --------------------------------
# DPO stalled on this data because ranking two responses needs a judge, and
# every judge available here is too noisy (see EXPERIMENT_LOG 7.0). SFT needs
# no ranking at all -- just good answers -- so the same source becomes usable
# by keeping `chosen` and dropping `rejected`.
#
# It also attacks the measured root cause rather than working around it. The
# probe says the base cannot separate these pairs, which means it does not
# model this distribution well. SFT raises in-distribution capability head-on;
# a later DPO round would then start from a base that finds the same pairs
# easier, because separability is a property of the model, not only the data.
#
# Memory
# ------
# Full-parameter 7B needs ~98 GB on one card (bf16 weights 14 + fp32 master 28
# + AdamW's two moments 56), so single-GPU OOMs while allocating exp_avg_sq.
# ZeRO-3 shards all three across 8 cards and lands near 18 GB each, measured.
# zero2 would NOT be enough here -- it shards optimizer state but replicates
# parameters, unlike the DPO runs where LoRA kept the trainable set tiny.
#
# Learning rate
# -------------
# 1e-6, an order of magnitude below the LoRA DPO runs. Full-parameter updates
# touch all 7B weights, and the released checkpoint is already well
# calibrated -- MME 2353 against the paper's 2362. The failure mode to avoid
# is not underfitting but overwriting general capability with 20k VQA answers.
# Upstream's own finetune scripts use 1e-6 for the same reason.
#
# Usage:
#   export VITA_SFT_DATA_DIR=/path/to/rlaif_v_sft
#   export WEIGHTS_ROOT=/path/to/weights
#   bash script/train/sft_rlaif_v_8gpu.sh /path/to/output

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

if [ -z "${VITA_SFT_DATA_DIR:-}" ]; then
    echo "error: VITA_SFT_DATA_DIR is not set." >&2
    echo "       python tools/make_rlaif_v_sft_data.py --parquet <shard> --out-dir <dir>" >&2
    exit 1
fi

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NGPU=$(awk -F, '{print NF}' <<< "${GPUS}")
GRAD_ACC=${GRAD_ACC:-4}
LR=${LR:-1e-6}
EPOCHS=${EPOCHS:-1}
PORT=${MASTER_PORT:-29701}
# 7 ranks x 4 dataloader workers each overran a 512 MB /dev/shm on this box and
# died with "DataLoader worker killed by signal: Bus error". 0 loads in-process
# and uses no shared memory; the step is dominated by the 7B forward anyway.
WORKERS=${WORKERS:-0}

echo "effective batch = ${NGPU} GPUs x 1 x ${GRAD_ACC} accum = $((NGPU * GRAD_ACC))"
echo "/dev/shm: $(df -h /dev/shm | awk 'NR==2{print $2}'), dataloader workers: ${WORKERS}"

OUTPUT_DIR_SFT=${OUTPUT_DIR}/sft-rlaif-v
mkdir -p "${OUTPUT_DIR_SFT}"

export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Vision tower and audio encoder stay frozen: 20k VQA answers is far too
# little to retrain perception, and unfreezing them is the fastest way to
# damage the benchmarks this experiment is measuring. Only the LLM and the
# mm_projector train.
deepspeed --include "localhost:${GPUS}" --master_port "${PORT}" vita/train/train.py \
    --deepspeed ./script/deepspeed/zero3.json \
    --model_name_or_path "${MODEL_PATH}" \
    --model_type qwen2p5_instruct \
    --version qwen2p5_instruct \
    --dataset_use RLAIFVSFT \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --audio_encoder "${AUDIO_ENCODER}" \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter True \
    --freeze_backbone False \
    --tune_mm_mlp_adapter False \
    --freeze_mm_mlp_adapter False \
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
    --save_steps 200 \
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
    --report_to none \
    2>&1 | tee -a "${OUTPUT_DIR_SFT}/log.txt"

echo "Done. Checkpoint in ${OUTPUT_DIR_SFT}"
