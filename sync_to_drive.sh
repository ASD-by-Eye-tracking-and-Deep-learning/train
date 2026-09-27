#!/usr/bin/env bash
# One-way mirror: local git repo -> Google Drive `code/` folder, so Colab can
# open and run train_all.ipynb against the same src/ package. Local is the
# source of truth; this script never reads back from Drive.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVE_DIR="${1:-$HOME/GoogleDrive/Duc n Huyen/ASD/train/code}"

mkdir -p "$DRIVE_DIR"

rsync -av --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.pth' \
  --exclude '*.h5' \
  --exclude 'artifacts/' \
  --exclude 'uv.lock' \
  "$REPO_DIR/" "$DRIVE_DIR/"

echo "Synced $REPO_DIR -> $DRIVE_DIR"
