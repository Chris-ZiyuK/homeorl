#!/bin/bash
# ============================================================
# Oscar 环境准备脚本
# 
# 功能：创建 venv，安装所有依赖，运行 sanity check
# 用法：bash scripts/setup_oscar_env.sh
# ============================================================

set -euo pipefail

echo "============================================"
echo "  HACE × Crafter — Oscar Environment Setup"
echo "============================================"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
echo "Project dir: $PROJECT_DIR"

# ── Step 1: Load modules ──────────────────────────────────────
echo ""
echo "─── Step 1: Loading modules ───"
module load python/3.11.0 2>/dev/null || module load python/3.10.12 2>/dev/null || {
    echo "WARNING: Could not load python module. Using system python."
}

# Check if CUDA is available (needed for GPU training)
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || {
    echo "WARNING: Could not load CUDA module. GPU training may not work."
}

echo "Python: $(which python3) ($(python3 --version))"

# ── Step 2: Create virtual environment ────────────────────────
echo ""
echo "─── Step 2: Creating virtual environment ───"
if [ -d ".venv" ]; then
    echo "Virtual environment already exists at .venv/"
    echo "Activating..."
    source .venv/bin/activate
else
    echo "Creating new virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "Created and activated .venv/"
fi

echo "Python (venv): $(which python) ($(python --version))"

# ── Step 3: Install PyTorch (GPU) ─────────────────────────────
echo ""
echo "─── Step 3: Installing PyTorch (CUDA) ───"
# Check if torch is already installed with CUDA
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "PyTorch with CUDA already installed. Skipping."
else
    echo "Installing PyTorch with CUDA support..."
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

# ── Step 4: Install project dependencies ──────────────────────
echo ""
echo "─── Step 4: Installing project dependencies ───"
pip install -r requirements.txt

# ── Step 5: Verify installations ──────────────────────────────
echo ""
echo "─── Step 5: Verifying installations ───"

FAIL=0

# Check each critical package
for pkg in crafter gymnasium numpy torch matplotlib stable_baselines3; do
    # Try __version__ first, fall back to importlib.metadata
    if python -c "
import $pkg
try:
    v = $pkg.__version__
except AttributeError:
    from importlib.metadata import version
    v = version('$pkg'.replace('_', '-'))
print(f'  ✓ {\"$pkg\":20s} {v}')
" 2>/dev/null; then
        :
    else
        echo "  ✗ $pkg — NOT INSTALLED"
        FAIL=1
    fi
done

# Check CUDA
echo ""
if python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null; then
    :
else
    echo "  WARNING: Could not check CUDA status"
fi

if [ "$FAIL" -eq 1 ]; then
    echo ""
    echo "✗ Some packages failed to install. Fix errors above before proceeding."
    exit 1
fi

# ── Step 6: Run Crafter sanity test ───────────────────────────
echo ""
echo "─── Step 6: Running Crafter sanity test ───"
echo "(This creates a Crafter env, runs a few steps, checks wrapper...)"
echo ""

python experiments/crafter/test_wrapper.py

echo ""
echo "─── Step 7: Running Gymnasium adapter test ───"
python experiments/crafter/test_gymnasium_adapter.py

# ── Done ──────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  ✓ Environment setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Submit smoke test:"
echo "       sbatch experiments/crafter/run_crafter_smoke_test.sh"
echo ""
echo "    2. Check job status:"
echo "       squeue -u \$USER"
echo ""
echo "    3. After completion, check results:"
echo "       ls experiments/crafter/results/smoke_*/"
echo "============================================"
