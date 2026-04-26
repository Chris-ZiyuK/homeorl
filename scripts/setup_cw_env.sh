#!/usr/bin/env bash
# Continual World optional deps. Usage: bash scripts/setup_cw_env.sh
# With editable CW: CW_REPO=/path/to/continual_world bash scripts/setup_cw_env.sh

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -d .venv ]]; then source .venv/bin/activate
else python3 -m venv .venv && source .venv/bin/activate
fi

if [[ ! -d "$HOME/.mujoco/mujoco200" ]]; then
  echo "mujoco-py expects MuJoCo at: $HOME/.mujoco/mujoco200"
  echo "Install MuJoCo 2.0 there first, then rerun this script."
  echo "Reference: https://github.com/openai/mujoco-py#install-mujoco"
  exit 1
fi

# mujoco-py builds native extensions and needs a GCC toolchain.
# Respect user-provided CC/CXX when set; otherwise require a visible gcc.
if [[ -n "${CC:-}" ]]; then
  if ! command -v "$CC" >/dev/null 2>&1; then
    echo "CC is set but not found: $CC"
    echo "Set CC/CXX to valid compiler binaries, then rerun."
    exit 1
  fi
else
  if ! command -v gcc >/dev/null 2>&1; then
    echo "GCC not found in PATH; mujoco-py build will fail."
    echo "On macOS: brew install gcc"
    echo "Then set compiler env vars, e.g.:"
    echo "  export CC=gcc-14"
    echo "  export CXX=g++-14"
    exit 1
  fi
fi

pip install -U pip
# Install base deps first.
pip install -r requirements.txt
# mujoco-py is legacy; force compatible build toolchain and preinstall deps it
# imports at build time when isolation is disabled.
pip install "cython<3" "setuptools<70" "wheel<0.42" fasteners cffi glfw imageio
PIP_NO_BUILD_ISOLATION=1 pip install --no-build-isolation --no-deps "mujoco-py>=2.0,<2.1"
# Install the rest of CW stack (tensorflow/metaworld/etc.).
pip install -r requirements-cw.txt
[[ -n "${CW_REPO:-}" ]] && pip install -e "$CW_REPO"


python - <<'PY'
import importlib
for m in ("tensorflow", "mujoco_py", "metaworld", "pandas", "matplotlib", "seaborn"):
    importlib.import_module(m)
    print("ok", m)
try:
    importlib.import_module("continualworld")
    print("ok continualworld")
except ImportError:
    print("skip continualworld (install with CW_REPO=... bash scripts/setup_cw_env.sh)")
PY
