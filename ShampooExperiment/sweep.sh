#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Python venv setup with uv

# $HOME has a small quota on PACE and was causing "Disk quota exceeded" errors
# when uv tried to cache packages / download a managed Python interpreter there.
# Redirect uv's cache and managed-Python installs to project storage instead,
# which has a much larger quota.
export UV_CACHE_DIR="$PWD/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/.uv-python"
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR"

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

source "$HOME/.local/bin/env"

# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu --frozen
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

python3 QuadGridSearch.py