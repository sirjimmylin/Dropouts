#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
KERNEL_NAME="${KERNEL_NAME:-final-project-local}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-Python (Final-Project Local)}"

echo "[1/4] Upgrading pip"
"$PYTHON_BIN" -m pip install --upgrade pip

echo "[2/4] Installing core notebook dependencies"
"$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements-notebook-local.txt"

echo "[3/4] Installing rasterio (optional, used by BigEarthNet)"
if ! "$PYTHON_BIN" -m pip install rasterio; then
  echo "[warn] rasterio install failed; continuing with tifffile fallback."
fi

echo "[4/4] Registering VSCode/Jupyter kernel"
"$PYTHON_BIN" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

echo "Local notebook environment setup complete."
echo "Kernel: $KERNEL_DISPLAY_NAME (name: $KERNEL_NAME)"
