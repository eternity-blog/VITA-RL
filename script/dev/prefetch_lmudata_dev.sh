#!/usr/bin/env bash
# 预下载 VLMEvalKit 基准数据集 tsv（POPE / MME / MMBench_DEV_EN_V11），
# 训练结束后评测就不用再等下载。LMUData 放 Ceph 持久目录。
set -uo pipefail

export http_proxy=${DEV_HTTP_PROXY:-}
export https_proxy=${DEV_HTTP_PROXY:-}
export no_proxy=localhost,127.0.0.1

export LMUData=/data/agent/lixiao29/LMUData
mkdir -p "$LMUData"

REPO=/data/agent/lixiao29/VITA-RL-sync
export PYTHONPATH=$REPO
export VITA_CKPT=/data/agent/lixiao29/vita-weights/VITA-1.5
PY=/data/agent/conda/envs/vita-rl/bin/python

cd "$REPO/VLMEvalKit"

echo "== sanity: import vlmeval =="
"$PY" -c "
from vlmeval.config import supported_VLM
assert 'vita_qwen2' in supported_VLM, sorted(supported_VLM)[:5]
print('vlmeval import OK, vita_qwen2 registered')
"

for DS in POPE MME MMBench_DEV_EN_V11; do
  echo "== prefetch $DS =="
  for attempt in 1 2 3 4 5; do
    "$PY" -c "
from vlmeval.dataset import build_dataset
d = build_dataset('$DS')
print('$DS ready, n =', len(d.data))
" && break
    echo "retry $DS in 10s..."
    sleep 10
  done
done

ls -lh "$LMUData"
echo "== LMUDATA PREFETCH DONE =="
