#!/usr/bin/env bash
# Editable COOM install + empty COOM/COOM/__init__.py when missing (needed for
# `import COOM...` on some Python versions / clone layouts; same fix as local).
#
# Usage (from anywhere, venv activated):
#   bash scripts/install_coom_editable.sh /path/to/COOM

set -euo pipefail

REPO="${1:?Usage: bash scripts/install_coom_editable.sh /path/to/COOM-clone}"

if [ ! -d "$REPO/COOM" ]; then
  echo "ERROR: expected package dir at $REPO/COOM (is REPO the COOM git root?)" >&2
  exit 1
fi

INIT="$REPO/COOM/__init__.py"
if [ ! -f "$INIT" ]; then
  : >"$INIT"
  echo "Created $INIT (empty; enables import COOM.* on some setups)"
fi

pip install -e "$REPO"
echo "Installed COOM editable from: $REPO"
