# -----------------------------------------------------------------------------
# Python venv setup with uv

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu --frozen
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

python3 QuadGridSearch.py