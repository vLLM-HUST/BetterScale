#!/usr/bin/env bash
set -euo pipefail
: "${PYTHON_BIN:?set a torch-capable Python interpreter}"
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$HERE/../../src" TORCH_DEVICE_BACKEND_AUTOLOAD=0
exec "$PYTHON_BIN" -m unittest discover -s "$HERE" -p test_state.py -v
