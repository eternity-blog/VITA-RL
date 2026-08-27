#!/usr/bin/env bash
# 下载 VITA-1.5 (~19.6GB) 和 InternViT-300M (~0.3GB) 到 Ceph（重启不丢）。
# HF 大文件走海外代理只有 ~90KB/s，hf-mirror ~540KB/s，ModelScope 走国内代理 ~9MB/s，
# 与旧机器结论一致（PROJECT_SUMMARY.md：ModelScope 是唯一返回真实字节的路径）。
# 借用 acestep 环境的 modelscope 1.34.0；支持断点续传，外层再套重试。
set -uo pipefail

export http_proxy=${DEV_HTTP_PROXY:-}
export https_proxy=${DEV_HTTP_PROXY:-}
export no_proxy=localhost,127.0.0.1

W=/data/agent/lixiao29/vita-weights
mkdir -p "$W"
PY=/data/agent/conda/envs/acestep/bin/python

for attempt in $(seq 1 30); do
  echo "== attempt $attempt =="
  "$PY" - <<EOF && break
from modelscope import snapshot_download
w = "$W"
snapshot_download("OpenGVLab/InternViT-300M-448px", local_dir=f"{w}/InternViT-300M-448px")
print("InternViT done", flush=True)
snapshot_download("VITA-MLLM/VITA-1.5", local_dir=f"{w}/VITA-1.5")
print("VITA-1.5 done", flush=True)
EOF
  echo "download interrupted, retrying in 15s..."
  sleep 15
done

echo "== verifying shards =="
"$PY" - <<EOF
import json, os
w = "$W/VITA-1.5"
idx = json.load(open(f"{w}/model.safetensors.index.json"))
missing = [f for f in set(idx["weight_map"].values()) if not os.path.exists(f"{w}/{f}")]
print("missing shards:", missing or "none")
EOF
echo "== WEIGHTS ALL DONE =="
