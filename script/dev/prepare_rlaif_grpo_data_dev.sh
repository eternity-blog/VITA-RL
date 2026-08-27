#!/usr/bin/env bash
# 下载 RLAIF-V 的 4 个 parquet 分片（与 DPO 实验同源：000/001/003/004），
# 转换成 GRPO prompt-only 格式。ModelScope 走国内代理 ~16MB/s。
set -euo pipefail

export http_proxy=${DEV_HTTP_PROXY:-}
export https_proxy=${DEV_HTTP_PROXY:-}
export no_proxy=localhost,127.0.0.1

W=/data/agent/lixiao29/vita-weights
RAW=$W/rlaif_v_raw
OUT=$W/rlaif_v_grpo
ENV=/data/agent/conda/envs/vita-rl
mkdir -p "$RAW"

"$ENV/bin/python" -c "import pyarrow" 2>/dev/null || \
  "$ENV/bin/pip" install --only-binary=:all: pyarrow

for s in 000 001 003 004; do
  f="$RAW/RLAIF-V-Dataset_${s}.parquet"
  if [ -s "$f" ]; then echo "already have $f"; continue; fi
  echo "== downloading shard $s =="
  curl -fL --retry 5 --retry-delay 5 -C - -o "$f" \
    "https://modelscope.cn/datasets/OpenBMB/RLAIF-V-Dataset/resolve/master/RLAIF-V-Dataset_${s}.parquet"
done
ls -la "$RAW"

echo "== converting to GRPO prompt format =="
cd /data/agent/lixiao29/VITA-RL-sync
"$ENV/bin/python" tools/make_rlaif_v_grpo_data.py \
  --parquet "$RAW"/RLAIF-V-Dataset_000.parquet "$RAW"/RLAIF-V-Dataset_001.parquet \
            "$RAW"/RLAIF-V-Dataset_003.parquet "$RAW"/RLAIF-V-Dataset_004.parquet \
  --out-dir "$OUT" --limit 8000

echo "== DATA PREP DONE =="
