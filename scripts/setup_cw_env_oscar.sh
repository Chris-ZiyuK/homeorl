#!/bin/bash
# Oscar setup for Continual World dependencies + optional editable install.
# Usage:
#   bash scripts/setup_cw_env_oscar.sh
#   CW_REPO=/path/to/continual_world bash scripts/setup_cw_env_oscar.sh

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_DIR"

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

if [[ ! -d "$HOME/.mujoco/mujoco200" ]]; then
  echo "ERROR: MuJoCo 2.0 not found at $HOME/.mujoco/mujoco200" >&2
  echo "Install MuJoCo 2.0 there, then rerun." >&2
  exit 1
fi

if [[ -n "${CC:-}" ]]; then
  command -v "$CC" >/dev/null 2>&1 || { echo "ERROR: CC not found: $CC" >&2; exit 1; }
fi

pip install -U pip
pip install -r requirements.txt
pip install "cython<3" "setuptools<70" "wheel<0.42" fasteners cffi glfw imageio
PIP_NO_BUILD_ISOLATION=1 pip install --no-build-isolation --no-deps "mujoco-py>=2.0,<2.1"
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
    print("skip continualworld (set CW_REPO=... for editable install)")
PY
