#!/usr/bin/env bash
# 为 vita-rl 环境安装 VLMEvalKit 依赖（HANDBOOK §"VLMEvalKit" 的 --no-deps 原则）。
# 关键：绝不能直接 pip install -r VLMEvalKit/requirements.txt（会升级 transformers/torch）。
# 本机清华源有 antlr4-python3-runtime 4.9.3，可以解掉旧机器上"未解决"的 omegaconf 坑。
set -euo pipefail

export http_proxy=${DEV_HTTP_PROXY:-}
export https_proxy=${DEV_HTTP_PROXY:-}
export no_proxy=localhost,127.0.0.1

ENV=/data/agent/conda/envs/vita-rl
PIP="$ENV/bin/pip"

echo "== VLMEvalKit direct deps (--no-deps) =="
"$PIP" install --only-binary=:all: --no-deps \
  pandas openpyxl portalocker rich sty tabulate tiktoken validators \
  xlsxwriter "omegaconf==2.3.1" imageio matplotlib python-dotenv "openai==1.3.5"

echo "== sdist-only packages =="
"$PIP" install --no-deps timeout-decorator "antlr4-python3-runtime==4.9.3"

echo "== transitive deps missed by --no-deps =="
"$PIP" install --only-binary=:all: --no-deps \
  pytz python-dateutil tzdata six markdown-it-py mdurl pygments \
  contourpy cycler fonttools kiwisolver pyparsing et-xmlfile \
  regex httpx httpcore h11 anyio sniffio distro annotated-types \
  pydantic pydantic-core typing-inspection jiter

echo "== moviepy 2.x + shim (mvbench.py 无条件 import moviepy.editor) =="
"$PIP" install --only-binary=:all: --no-deps moviepy proglog imageio-ffmpeg decorator

SP="$ENV/lib/python3.10/site-packages"
cat > "$SP/moviepy_editor_shim.py" <<'EOF'
"""moviepy 2.x removed moviepy.editor / moviepy.config_defaults; VLMEvalKit
mvbench.py imports the former and does attribute access on the latter
(moviepy.config_defaults.LOGGER_LEVEL = ...), and is imported unconditionally
by dataset/__init__.py. Alias both so image benchmarks are not held hostage
by a video benchmark. Registered via moviepy_editor_shim.pth."""
import sys
import types

try:
    import moviepy

    sys.modules.setdefault("moviepy.editor", moviepy)
    moviepy.editor = moviepy

    _cfg = types.ModuleType("moviepy.config_defaults")
    sys.modules.setdefault("moviepy.config_defaults", _cfg)
    moviepy.config_defaults = _cfg
except Exception:
    pass
EOF
echo "import moviepy_editor_shim" > "$SP/moviepy_editor_shim.pth"

echo "== verify core stack untouched =="
"$ENV/bin/python" -c "
import torch, transformers, numpy
assert torch.__version__.startswith('2.3.1'), torch.__version__
assert transformers.__version__ == '4.41.1', transformers.__version__
assert numpy.__version__ == '1.26.4', numpy.__version__
print('core stack OK:', torch.__version__, transformers.__version__, numpy.__version__)
import omegaconf; print('omegaconf', omegaconf.__version__)
import moviepy.editor; print('moviepy.editor shim OK')
import pandas, rich, tiktoken; print('vlmeval deps OK')
"

echo "== VLMEVAL DEPS DONE =="
