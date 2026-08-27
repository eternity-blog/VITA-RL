#!/usr/bin/env bash
# 在 KML 开发机上为 VITA-RL 搭建 conda 环境（按 REPRODUCE.md 的分阶段顺序）。
# 环境建在 /data/agent/conda/envs/vita-rl（Ceph，重启不丢）。
set -euo pipefail

PROXY_CN=${DEV_HTTP_PROXY:-}
PROXY_OV=http://oversea-squid1.jp.txyun:11080
NO_PROXY=localhost,127.0.0.1

ENV=/data/agent/conda/envs/vita-rl
PIP="$ENV/bin/pip"

# pip 源是清华镜像（国内），conda-forge 和 GitHub 在海外
cn() { http_proxy=$PROXY_CN https_proxy=$PROXY_CN no_proxy=localhost,127.0.0.1
ov() { http_proxy=$PROXY_OV https_proxy=$PROXY_OV no_proxy=localhost,127.0.0.1

if [ ! -x "$ENV/bin/python" ]; then
  echo "== stage 0: conda create python=3.10 =="
  ov /data/agent/conda/bin/conda create -y -n vita-rl python=3.10
fi

echo "== stage 1: torch stack =="
cn "$PIP" install --only-binary=:all: "pillow==10.4.0" \
    torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1

echo "== stage 2: core deps =="
cn "$PIP" install --only-binary=:all: "numpy<2" "xformers==0.0.27" \
    "transformers==4.41.1" accelerate decord Jinja2 ninja tqdm

echo "== stage 3: imports missing from requirements.txt =="
cn "$PIP" install --only-binary=:all: timm einops PyYAML \
    "opencv-python-headless==4.10.0.84" "soundfile==0.12.1" librosa \
    sentencepiece protobuf six

echo "== stage 4: re-pin numpy (numba/librosa 会拉回 numpy 2.x) =="
cn "$PIP" install --only-binary=:all: "numpy==1.26.4" "numba<0.61" "llvmlite<0.44"

echo "== stage 5: deepspeed + peft (训练 / LoRA / GRPO) =="
cn "$PIP" install "deepspeed==0.14.4"
cn "$PIP" install --only-binary=:all: "peft==0.11.1"

echo "== stage 6: flash-attn 预编译 wheel =="
# GitHub 走海外代理只有 ~100KB/s，实际做法是在本地 Mac 下载后 scp 上来。
FA_WHL=/data/agent/lixiao29/flash_attn-2.5.9.post1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
if [ -f "$FA_WHL" ]; then
  "$PIP" install "$FA_WHL"
else
  ov "$PIP" install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
fi

echo "== verify =="
"$ENV/bin/python" - <<'EOF'
import torch, numpy, transformers
print('torch', torch.__version__, '| cuda', torch.version.cuda)
print('numpy', numpy.__version__, '| transformers', transformers.__version__)
print('GPUs', torch.cuda.device_count(), '| bf16', torch.cuda.is_bf16_supported())
import flash_attn
from flash_attn import flash_attn_func
q = torch.randn(2,64,8,64, dtype=torch.bfloat16, device='cuda')
print('flash-attn kernel ok:', tuple(flash_attn_func(q,q,q).shape))
import torchaudio; print('audio backends:', torchaudio.list_audio_backends())
EOF

echo "== ALL DONE =="
