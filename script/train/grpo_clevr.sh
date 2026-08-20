#!/bin/bash
# Multimodal GRPO on CLEVR counting with a VERIFIABLE reward (R1-V's recipe).
#
# Why this dataset, after two RLAIF-V rounds
# ------------------------------------------
# RLAIF-V's rewards are proxies (keyword overlap / LLM judge against a
# model-written gold). On open-ended description, 8 rollouts from a decent
# base model are all "fine" and differ in style, so the within-group ranking
# is mostly noise -- observed directly: r2 ran 91 steps with KL up 6x and no
# trend in keyword/judge. Counting has ground truth. The `answer` reward is
# binary right/wrong, the group's pass/fail mix carries the signal, and the
# hacking surface collapses (you cannot fake the correct count). R1-V took
# Qwen2-VL-2B from 48% to 82.5% OOD counting in ~100 steps on this data.
#
# Reward design
# -------------
#   answer:1.0       exact match against reward_meta["answer"] (binary)
#   format:0.3       <think>..</think><answer>..</answer> structure; watch it
#                    saturate first while answer accuracy climbs after --
#                    the classic two-phase R1 curve
# keyword/length/no_repeat/judge/state_token are deliberately absent: no
# proxies needed when the reward is verifiable, and this reproduction is
# text-image only so VITA's audio state-token prefix is not worth guarding.
#
# What to look for
# ----------------
#   reward/answer            = train accuracy; this is THE metric and it
#                              should visibly climb (R1-V baseline: within
#                              ~100 steps)
#   groups/degenerate_frac   all-right or all-wrong groups; moderate values
#                              are normal, near 1.0 means no headroom left
#                              (or none to begin with)
#   grpo/kl at step 1        ~= 0, same invariant as always
#
# Usage:
#   export VITA_CLEVR_GRPO_DATA_DIR=/path/to/clevr_grpo
#   export WEIGHTS_ROOT=/path/to/weights
#   bash script/train/grpo_clevr.sh /path/to/output

set -euo pipefail

OUTPUT_DIR=${1:?"usage: $0 <output_dir>"}

WEIGHTS_ROOT=${WEIGHTS_ROOT:-${VITA_WEIGHTS:-}}
if [ -z "${WEIGHTS_ROOT}" ] && [ -z "${MODEL_PATH:-}" ]; then
    echo "error: set WEIGHTS_ROOT (or VITA_WEIGHTS)" >&2
    exit 1
fi
MODEL_PATH=${MODEL_PATH:-${WEIGHTS_ROOT}/VITA-1.5}
VISION_TOWER=${VISION_TOWER:-${WEIGHTS_ROOT}/InternViT-300M-448px}
if [ -z "${AUDIO_ENCODER:-}" ]; then
    AUDIO_ENCODER=${MODEL_PATH}/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning
    if [ ! -d "${AUDIO_ENCODER}" ]; then
        AUDIO_ENCODER=${WEIGHTS_ROOT}/VITA-1.5/audio-encoder-Qwen2-7B-1107-weight-base-11wh-tunning
    fi
fi

for p in "${MODEL_PATH}" "${VISION_TOWER}" "${AUDIO_ENCODER}"; do
    [ -d "${p}" ] || { echo "error: not a directory: ${p}" >&2; exit 1; }
done

if [ -z "${VITA_CLEVR_GRPO_DATA_DIR:-}" ]; then
    echo "error: VITA_CLEVR_GRPO_DATA_DIR is not set." >&2
    echo "       python tools/make_clevr_grpo_data.py --parquet <shards> --out-dir <dir>" >&2
    exit 1
fi

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NGPU=$(awk -F, '{print NF}' <<< "${GPUS}")
GRAD_ACC=${GRAD_ACC:-8}
LR=${LR:-1e-6}
BETA=${BETA:-0.04}
GROUP_SIZE=${GROUP_SIZE:-8}
# Counting answers are short; <think> reasoning needs some room but nothing
# like open-ended description.
MAX_NEW=${MAX_NEW:-128}
EPOCHS=${EPOCHS:-1}
MAX_STEPS=${MAX_STEPS:--1}
# PPO-style sample reuse (optimizer steps per rollout batch); 1 = on-policy.
NUM_ITER=${NUM_ITER:-1}
PORT=${MASTER_PORT:-29756}
WORKERS=${WORKERS:-0}
REPORT_TO=${REPORT_TO:-none}

echo "effective batch = ${NGPU} GPUs x 1 x ${GRAD_ACC} accum = $((NGPU * GRAD_ACC)) prompts/step"
echo "group_size = ${GROUP_SIZE} -> $((NGPU * GROUP_SIZE)) rollouts/step"

OUTPUT_DIR_GRPO=${OUTPUT_DIR}/grpo-clevr
mkdir -p "${OUTPUT_DIR_GRPO}"

export PYTHONPATH=./
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REWARD_FNS=${REWARD_FNS:-"answer:1.0,format:0.3"}

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
    --grpo_num_iterations "${NUM_ITER}" \
    --grpo_temperature 1.0 \
    --grpo_top_p 0.95 \
    --reward_fns "${REWARD_FNS}" \
    --model_name_or_path "${MODEL_PATH}" \
    --model_type qwen2p5_instruct \
    --version qwen2p5_instruct \
    --dataset_use CLEVRGRPO \
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
    --max_steps "${MAX_STEPS}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "${GRAD_ACC}" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 3 \
    --learning_rate "${LR}" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 2 \
    --tf32 True \
    --model_max_length 6200 \
    --gradient_checkpointing True \
    --dataloader_num_workers "${WORKERS}" \
    --lazy_preprocess True \
    --report_to "${REPORT_TO}" \
    --run_name "${WANDB_NAME:-grpo-clevr}" \
    2>&1 | tee -a "${OUTPUT_DIR_GRPO}/log.txt"

echo "Done. Adapter in ${OUTPUT_DIR_GRPO}"
