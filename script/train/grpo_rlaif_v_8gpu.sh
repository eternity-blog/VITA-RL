#!/bin/bash
# Real multimodal GRPO on RLAIF-V: image+text prompts, 8 GPUs, LoRA + ZeRO-2.
#
# This is the scaled-up counterpart of grpo_mm_smoke_test.sh. The smoke test
# verifies the vision-fusion path runs; this script trains on real grounded
# VQA and is the deliverable the multimodal GRPO extension was written for.
#
# Why GRPO here, after SFT-then-DPO already worked
# -----------------------------------------------
# DPO needs a *preference* (which of two fixed responses is better), and on
# RLAIF-V that signal was too weak until an SFT round raised separability
# (EXPERIMENT_LOG 9.7). GRPO needs no preference pair: it samples G
# completions per image, scores each with a reward, and ranks them *within
# the group*. The group baseline removes the need for a value network, and
# the reward -- keyword overlap with the gold answer -- is a groundedness
# check the model can actually optimize by describing the image correctly.
#
# Reward design
# -------------
# Default reward_fns is rule-based and needs no extra model:
#   keyword:1.0   fraction of the gold answer's content words the rollout
#                mentions (the core groundedness signal)
#   length:0.4    right-sized answers beat one-word and rambling ones
#   no_repeat:0.3 brake on degenerate sampling loops
#   state_token:0.3  VITA's reply-format token
# For a stronger (model-based) signal, point --judge_model_path at a small
# instruct model and add 'judge:1.0' to --reward_fns; the trainer loads it
# lazily. Not the default because a per-step 7B judge doubles forward cost.
#
# Memory
# ------
# group_size 8 means each step runs, per prompt: 1 vision encode, 8 rollouts
# (generate), 8 policy-logp forwards, 8 reference-logp forwards (adapter off,
# no_grad). LoRA on 7B + InternViT-300M lands ~28 GB/GPU, so ZeRO-2 (not 3)
# is enough -- and disable_adapter() is cleaner without ZeRO-3 sharding,
# same reasoning as dpo_rlaif_v_8gpu.sh.
#
# /dev/shm is 512 MB on this box; 8 ranks x N dataloader workers each ship
# 448x448 tiles through shared memory and OOM. WORKERS=0 loads in-process
# and uses none. The step is dominated by the 7B forwards, not data loading.
#
# What to look for
# ----------------
#   grpo/kl at step 1     ~= 0  (policy == reference at start; nonzero means
#                                the fused-embeddings reference is mis-wired
#                                or mm_projector escaped the freeze)
#   reward/mean           trending up -- the groundedness signal is landing
#   groups/degenerate_frac  low; near 1.0 means the keyword reward cannot
#                                tell rollouts apart (check keyword mining)
#   grpo/ratio            ~= 1 first inner step, by construction
#
# Usage:
#   export VITA_RLAIF_GRPO_DATA_DIR=/path/to/rlaif_v_grpo
#   export WEIGHTS_ROOT=/path/to/weights
#   bash script/train/grpo_rlaif_v_8gpu.sh /path/to/output

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
# released base's copy -- a checkpoint produced by our own SFT keeps frozen
# encoders as absolute paths in its config, so deriving from MODEL_PATH fails
# for exactly the SFT -> RL chain this script supports.
if [ -z "${AUDIO_ENCODER:-}" ]; then
    AUDIO_ENCODER=${MODEL_PATH}/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning
    if [ ! -d "${AUDIO_ENCODER}" ]; then
        AUDIO_ENCODER=${WEIGHTS_ROOT}/VITA-1.5/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning
    fi
fi

for p in "${MODEL_PATH}" "${VISION_TOWER}" "${AUDIO_ENCODER}"; do
    [ -d "${p}" ] || { echo "error: not a directory: ${p}" >&2; exit 1; }
done

if [ -z "${VITA_RLAIF_GRPO_DATA_DIR:-}" ]; then
    echo "error: VITA_RLAIF_GRPO_DATA_DIR is not set." >&2
    echo "       python tools/make_rlaif_v_grpo_data.py --parquet <shards> --out-dir <dir>" >&2
    exit 1
fi

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NGPU=$(awk -F, '{print NF}' <<< "${GPUS}")
GRAD_ACC=${GRAD_ACC:-8}
LR=${LR:-1e-6}
BETA=${BETA:-0.04}
GROUP_SIZE=${GROUP_SIZE:-8}
MAX_NEW=${MAX_NEW:-96}
EPOCHS=${EPOCHS:-1}
PORT=${MASTER_PORT:-29755}
WORKERS=${WORKERS:-0}

echo "effective batch = ${NGPU} GPUs x 1 x ${GRAD_ACC} accum = $((NGPU * GRAD_ACC)) prompts/step"
echo "group_size = ${GROUP_SIZE} -> $((NGPU * GROUP_SIZE)) rollouts/step (vision tower runs once per distinct image)"
echo "/dev/shm: $(df -h /dev/shm | awk 'NR==2{print $2}'), dataloader workers: ${WORKERS}"

OUTPUT_DIR_GRPO=${OUTPUT_DIR}/grpo-rlaif-v-8gpu
mkdir -p "${OUTPUT_DIR_GRPO}"

export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REWARD_FNS=${REWARD_FNS:-"keyword:1.0,length:0.4,no_repeat:0.3,state_token:0.3"}
JUDGE_ARG=""
if [ -n "${JUDGE_MODEL_PATH:-}" ]; then
    JUDGE_ARG="--judge_model_path ${JUDGE_MODEL_PATH}"
    echo "judge model: ${JUDGE_MODEL_PATH} (add 'judge:1.0' to REWARD_FNS to weight it)"
fi

# lora_dropout 0 for the same reason as the DPO/text-GRPO scripts: the
# reference pass runs with the adapter disabled and therefore without
# dropout, so dropout on the policy side would make the step-1 KL nonzero
# for no good reason.
deepspeed --include "localhost:${GPUS}" --master_port "${PORT}" vita/train/train_grpo.py \
    --deepspeed ./script/deepspeed/zero2.json \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --grpo_group_size "${GROUP_SIZE}" \
    --grpo_beta "${BETA}" \
    --grpo_clip_eps 0.2 \
    --grpo_max_new_tokens "${MAX_NEW}" \
    --grpo_temperature 1.0 \
    --grpo_top_p 0.95 \
    --reward_fns "${REWARD_FNS}" \
    ${JUDGE_ARG} \
    --model_name_or_path "${MODEL_PATH}" \
    --model_type qwen2p5_instruct \
    --version qwen2p5_instruct \
    --dataset_use RLAIFVGRPO \
    --vision_tower "${VISION_TOWER}" \
    --mm_projector_type mlp2x_gelu \
    --audio_encoder "${AUDIO_ENCODER}" \
    --freeze_audio_encoder True \
    --freeze_audio_encoder_adapter True \
    --image_aspect_ratio square \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir "${OUTPUT_DIR_GRPO}" \
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
    2>&1 | tee -a "${OUTPUT_DIR_GRPO}/log.txt"

echo "Done. Adapter in ${OUTPUT_DIR_GRPO}"
