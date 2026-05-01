#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${1:-taac2026-ubuntu2204-cu126.tar.gz}"
PREFIX="${2:-$HOME/envs/taac2026}"

if [ -e "$PREFIX" ]; then
  echo "Target env already exists: $PREFIX"
  echo "Please remove it manually or choose another prefix."
  exit 1
fi

mkdir -p "$PREFIX"
tar -xzf "$ARCHIVE" -C "$PREFIX"

"$PREFIX/bin/conda-unpack"

echo "Installed to: $PREFIX"
echo
echo "Activate with:"
echo "source $PREFIX/bin/activate"
echo
"$PREFIX/bin/python" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
