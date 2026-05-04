#!/bin/bash
# Oscar setup for Continual World dependencies + optional editable install.
#
# Usage:
#   source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
#   conda activate homeorl
#   USE_CONDA=1 bash scripts/setup_cw_env_oscar.sh
#
# Optional:
#   CW_REPO=/path/to/continual_world USE_CONDA=1 bash scripts/setup_cw_env_oscar.sh
#   CW_GIT_URL=https://github.com/awarelab/continual_world.git USE_CONDA=1 bash scripts/setup_cw_env_oscar.sh
#   Ensure GPU is used with Pytorch: 
#   INSTALL_TORCH_CUDA=1 USE_CONDA=1 bash scripts/setup_cw_env_oscar.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

# -----------------------------
# 1. Activate / verify Python env
# -----------------------------
if [[ "${USE_CONDA:-0}" == "1" ]]; then
  if [[ -z "${CONDA_PREFIX:-}" ]] || ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: USE_CONDA=1 requires an activated conda environment." >&2
    echo "Run:" >&2
    echo "  source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh" >&2
    echo "  conda activate homeorl" >&2
    exit 1
  fi

  ENV_PREFIX="$CONDA_PREFIX"

  echo "Using conda env: ${CONDA_DEFAULT_ENV:-?}"
  echo "CONDA_PREFIX: $CONDA_PREFIX"
else
  if command -v module >/dev/null 2>&1; then
    module load python/3.10.12 2>/dev/null || module load python/3.11.0 2>/dev/null || true
    module load gcc/11.3.0 2>/dev/null || module load gcc/10.2 2>/dev/null || true
  fi

  if [[ -d .venv ]]; then
    source .venv/bin/activate
  else
    python3 -m venv .venv
    source .venv/bin/activate
  fi

  ENV_PREFIX="$VIRTUAL_ENV"
fi

echo "Final python: $(command -v python)"
python --version
echo "Final pip: $(python -m pip --version)"

# -----------------------------
# 2. Check MuJoCo 2.0 install
# -----------------------------
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco200"

if [[ ! -d "$MUJOCO_PY_MUJOCO_PATH" ]]; then
  echo "ERROR: MuJoCo 2.0 not found at $MUJOCO_PY_MUJOCO_PATH" >&2
  echo "Install MuJoCo 2.0 there, then rerun." >&2
  exit 1
fi

if [[ ! -f "$HOME/.mujoco/mjkey.txt" ]]; then
  echo "ERROR: MuJoCo license key not found at $HOME/.mujoco/mjkey.txt" >&2
  echo "MuJoCo 2.0 + mujoco-py<2.1 expects this file." >&2
  exit 1
fi

echo "Checking MuJoCo files..."
ls "$MUJOCO_PY_MUJOCO_PATH" >/dev/null
ls "$MUJOCO_PY_MUJOCO_PATH/bin" >/dev/null

# -----------------------------
# 3. Oscar Mesa / OSMesa config
# -----------------------------
# This is the Mesa path that contains:
#   include/GL/osmesa.h
#   lib/libOSMesa.so
#
# We found this on Oscar via:
#   find /oscar/rt/9.6/25 -name osmesa.h
#   find /oscar/rt/9.6/25 -name "libOSMesa*"
export MESA_PREFIX="/oscar/rt/9.6/25/x86_64_v3/mesa-25.0.5-2kqswd2l3fkmb62exxaxvj7ykx63pnft"

if [[ ! -f "$MESA_PREFIX/include/GL/osmesa.h" ]]; then
  echo "ERROR: osmesa.h not found at $MESA_PREFIX/include/GL/osmesa.h" >&2
  echo "Try: module avail mesa && module load mesa/25.0.5-2kqs" >&2
  exit 1
fi

if [[ ! -f "$MESA_PREFIX/lib/libOSMesa.so" ]]; then
  echo "ERROR: libOSMesa.so not found at $MESA_PREFIX/lib/libOSMesa.so" >&2
  exit 1
fi

# Build-time include paths.
export CPATH="$MESA_PREFIX/include:$ENV_PREFIX/include:${CPATH:-}"
export C_INCLUDE_PATH="$MESA_PREFIX/include:$ENV_PREFIX/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$MESA_PREFIX/include:$ENV_PREFIX/include:${CPLUS_INCLUDE_PATH:-}"

# Link/runtime library paths.
export LIBRARY_PATH="$MESA_PREFIX/lib:$ENV_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$MESA_PREFIX/lib:$ENV_PREFIX/lib:$MUJOCO_PY_MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"

# Headless MuJoCo rendering.
export MUJOCO_GL=osmesa

# Fix old mujoco-py source:
# mujoco_py/gl/osmesashim.c uses printf without explicitly including stdio.h.
export CFLAGS="-include stdio.h ${CFLAGS:-}"

# Prefer gcc over clang; clang failed on implicit printf declaration.
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

echo "Checking OSMesa..."
echo "MESA_PREFIX=$MESA_PREFIX"
test -f "$MESA_PREFIX/include/GL/osmesa.h" && echo "ok osmesa.h"
test -f "$MESA_PREFIX/lib/libOSMesa.so" && echo "ok libOSMesa.so"

echo "Compiler:"
echo "CC=$CC"
"$CC" --version | head -n 1

# -----------------------------
# 4. Install Python dependencies
# -----------------------------
python -m pip install -U pip

# Main project dependencies.
python -m pip install -r requirements.txt

# Oscar GPU nodes currently expose CUDA 12.9 via driver 575.x.
# Avoid torch + cu130 because it can make torch.cuda.is_available() false.
if [[ "${INSTALL_TORCH_CUDA:-0}" == "1" ]]; then
  python -m pip uninstall -y torch torchvision torchaudio
  python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio
fi

# Build deps for old mujoco-py.
# setuptools/wheel are pinned lower because old mujoco-py can break with newer build tooling.
python -m pip install \
  "cython<3" \
  "setuptools<70" \
  "wheel<0.42" \
  fasteners \
  cffi \
  glfw \
  imageio

# Install mujoco-py first, without deps, so requirements-cw will see it already satisfied.
PIP_NO_BUILD_ISOLATION=1 python -m pip install \
  --no-build-isolation \
  --no-deps \
  "mujoco-py>=2.0,<2.1"

# Continual World dependencies.
python -m pip install -r requirements-cw.txt

# Continual World editable install:
# - Use CW_REPO when provided
# - Else use local ./continual_world when present
# - Else clone from CW_GIT_URL and install editable
if [[ -z "${CW_REPO:-}" ]]; then
  if [[ -d "continual_world" ]]; then
    CW_REPO="$(pwd)/continual_world"
  else
    CW_GIT_URL="${CW_GIT_URL:-https://github.com/awarelab/continual_world.git}"
    echo "continual_world repo not found; cloning from: $CW_GIT_URL"
    git clone "$CW_GIT_URL" continual_world
    CW_REPO="$(pwd)/continual_world"
  fi
fi
python -m pip install -e "$CW_REPO"

# MetaWorld / continual_world need classic gym step API; metaworld may pull a newer `gym` otherwise.
python -m pip install "gym>=0.20,<0.26"

# -----------------------------
# 5. Smoke tests
# -----------------------------
python - <<'PY'
import importlib

for m in ("mujoco_py", "metaworld", "pandas", "matplotlib", "seaborn"):
    importlib.import_module(m)
    print("ok", m)

try:
    importlib.import_module("tensorflow")
    print("ok tensorflow")
except Exception as e:
    print("tensorflow import failed:", repr(e))
    raise

try:
    importlib.import_module("continualworld")
    print("ok continualworld")
except ImportError:
    print("skip continualworld (set CW_REPO=... for editable install)")
PY

echo "Setup complete."