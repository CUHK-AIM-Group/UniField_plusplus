#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install 'vllm==0.19.0' --extra-index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
