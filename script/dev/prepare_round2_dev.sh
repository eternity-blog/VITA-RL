#!/usr/bin/env bash
# GRPO 第二轮准备：judge 模型 + 全量数据（带 gold 供 judge 对照打分）。
set -euo pipefail

export http_proxy=${DEV_HTTP_PROXY:-}
export https_proxy=${DEV_HTTP_PROXY:-}
export no_proxy=localhost,127.0.0.1

W=/data/agent/lixiao29/vita-weights
ENV=/data/agent/conda/envs/vita-rl
PY=$ENV/bin/python
# vita-rl 环境没有 modelscope；借 acestep 环境的 python 只做下载
MS_PY=/data/agent/conda/envs/acestep/bin/python

echo "== 1. judge model: Qwen2.5-3B-Instruct (ModelScope) =="
for attempt in 1 2 3 4 5; do
  "$MS_PY" - <<EOF && break
from modelscope import snapshot_download
snapshot_download("Qwen/Qwen2.5-3B-Instruct",
                  local_dir="$W/Qwen2.5-3B-Instruct")
print("judge model downloaded", flush=True)
EOF
  echo "download interrupted, retrying in 15s..."
  sleep 15
done

echo "== 2. verify judge loads under transformers 4.41.1 (CPU) =="
"$PY" - <<EOF
import sys
sys.path.insert(0, "/data/agent/lixiao29/VITA-RL-sync")
from vita.train.rewards import JudgeReward
j = JudgeReward("$W/Qwen2.5-3B-Instruct", device="cpu")
s_good = j("What color is the sky in the image?", "The sky is blue with a few clouds.",
           {"gold": "The sky in the picture is blue with scattered white clouds."})
s_bad  = j("What color is the sky in the image?", "There are three dogs playing chess.",
           {"gold": "The sky in the picture is blue with scattered white clouds."})
print(f"judge OK: good={s_good:.3f} bad={s_bad:.3f}")
assert s_good > s_bad, "judge cannot separate an obviously right/wrong pair"
EOF

echo "== 3. full data with gold in reward_meta =="
cd /data/agent/lixiao29/VITA-RL-sync
RAW=$W/rlaif_v_raw
"$PY" tools/make_rlaif_v_grpo_data.py \
  --parquet "$RAW"/RLAIF-V-Dataset_000.parquet "$RAW"/RLAIF-V-Dataset_001.parquet \
            "$RAW"/RLAIF-V-Dataset_003.parquet "$RAW"/RLAIF-V-Dataset_004.parquet \
  --out-dir "$W/rlaif_v_grpo_full" --limit 999999

echo "== ROUND2 PREP DONE =="
