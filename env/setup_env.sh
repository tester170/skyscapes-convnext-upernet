#!/usr/bin/env bash
# Creates the exact environment of the SkyScapes experiments in a separate Python 3.11 venv:
# PyTorch 2.1.0+cu121, MMCV 2.1.0 (official OpenMMLab wheel), MMEngine 0.10.7, MMSegmentation 1.2.2 and every other package
# pinned in env/requirements.lock (installed without dependency resolution, so versions cannot drift), plus the official
# DINOv3 code (facebookresearch/dinov3 at the commit used in the experiment) in vendor/dinov3 for the ConvNeXt backbone.
# Works on Google Colab (any Python in the notebook kernel) and on any Linux x86_64 with an NVIDIA driver >= 525.
# Usage: bash env/setup_env.sh [env_dir=/content/py311]   (idempotent: does nothing if the env already passes the check)
set -euo pipefail
export MPLBACKEND=Agg   # notebook kernels export an inline matplotlib backend unknown to this environment
BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV="${1:-/content/py311}"
PY="$ENV/bin/python"
DINO="$BUNDLE/vendor/dinov3"
DINO_COMMIT=6876159a11b4df116f30f667f8c9888617df0751
if [ "$(git -C "$DINO" rev-parse HEAD 2>/dev/null)" != "$DINO_COMMIT" ]; then
  echo "== official DINOv3 code (github.com/facebookresearch/dinov3, commit ${DINO_COMMIT:0:7})"
  rm -rf "$DINO"; mkdir -p "$DINO"
  git -C "$DINO" init -q
  git -C "$DINO" fetch -q --depth 1 https://github.com/facebookresearch/dinov3.git "$DINO_COMMIT"
  git -C "$DINO" checkout -q FETCH_HEAD
fi
if [ -x "$PY" ] && "$PY" "$BUNDLE/env/check_env.py" --quiet; then
  echo "environment is ready: $ENV"; exit 0
fi
# uv (Python and package installer): an installed one, else from pip, else the official standalone installer into
# .uv/ of the repository (plain Linux without pip, e.g. a fresh Ubuntu 24.04)
if command -v uv >/dev/null 2>&1; then UV=uv
elif python3 -m uv --version >/dev/null 2>&1; then UV="python3 -m uv"
else
  echo "== installing uv (Python and package installer)"
  if python3 -m pip install -q "uv>=0.4" 2>/dev/null      || python3 -m pip install -q --user --break-system-packages "uv>=0.4" 2>/dev/null; then UV="python3 -m uv"
  else
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$BUNDLE/.uv" INSTALLER_NO_MODIFY_PATH=1 sh >/dev/null
    UV="$BUNDLE/.uv/uv"
  fi
fi
echo "== creating Python 3.11 environment in $ENV"
$UV venv --python 3.11 --seed "$ENV"
echo "== installing pinned packages (PyTorch 2.1.0 cu121, MMEngine, MMSegmentation, ...): ~2-3 min"
$UV pip install --python "$PY" --no-deps --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu121 -r "$BUNDLE/env/requirements.lock"
echo "== installing MMCV 2.1.0 (prebuilt for torch 2.1 / CUDA 12.1)"
WHEEL="$BUNDLE/wheels/mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl"
if [ -f "$WHEEL" ] || python3 "$BUNDLE/scripts/fetch.py" mmcv --out "$WHEEL"; then   # project release, else OpenMMLab
  $UV pip install --python "$PY" --no-deps "$WHEEL"
else   # last resort: compile MMCV 2.1.0 against the installed torch (needs nvcc; ~30 min)
  echo "== OpenMMLab server unavailable: building MMCV 2.1.0 from source (about 30 minutes)"
  $UV pip install --python "$PY" ninja==1.11.1.1
  MMCV_WITH_OPS=1 FORCE_CUDA=1 MAX_JOBS="$(nproc)" PATH="$ENV/bin:/usr/local/cuda/bin:$PATH" \
    $UV pip install --python "$PY" --no-deps --no-build-isolation --no-binary mmcv mmcv==2.1.0
fi
"$PY" "$BUNDLE/env/check_env.py"
