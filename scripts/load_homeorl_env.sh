#!/bin/bash
# source ~/homeorl/scripts/load_homeorl_env.sh
# cd ~/homeorl

source /oscar/rt/9.6/25/spack/x86_64_v3/anaconda3-2023.09-0-aqbcryind6ewgctu7wijluakv5mo3lo5/etc/profile.d/conda.sh
conda activate homeorl

export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco200"
export MESA_PREFIX="/oscar/rt/9.6/25/x86_64_v3/mesa-25.0.5-2kqswd2l3fkmb62exxaxvj7ykx63pnft"

export LD_LIBRARY_PATH="$MESA_PREFIX/lib:$CONDA_PREFIX/lib:$MUJOCO_PY_MUJOCO_PATH/bin:${LD_LIBRARY_PATH:-}"
export MUJOCO_GL=osmesa

export CPATH="$MESA_PREFIX/include:$CONDA_PREFIX/include:${CPATH:-}"
export LIBRARY_PATH="$MESA_PREFIX/lib:$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export CFLAGS="-include stdio.h ${CFLAGS:-}"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

echo "homeorl environment loaded"
