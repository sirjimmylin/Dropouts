#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install \
  torchvision \
  scikit-learn \
  pillow \
  tqdm \
  pandas \
  tifffile

# Optional: BigEarthNet loader prefers rasterio when available.
python -m pip install rasterio || echo "rasterio install failed; falling back to tifffile."

echo "Environment setup complete."
