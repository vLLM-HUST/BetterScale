#!/usr/bin/env bash
# Deliberately external dependency, not a vendored or mutable installed allocator.
set -euo pipefail
: "${LIVEINFERENCE_ROOT:?set an explicit LiveInference checkout}"
: "${PYTHON_BIN:?set a torch-capable Python interpreter}"
expected=05ac15419c0e73650e687ceb9daffeb7874865f0
[[ $(git -C "$LIVEINFERENCE_ROOT" rev-parse HEAD) = "$expected" ]] || {
  echo "LiveInference revision differs from qualified $expected" >&2; exit 1;
}
[[ -z $(git -C "$LIVEINFERENCE_ROOT" status --porcelain -- src/livemodule) ]] || {
  echo 'LiveInference State implementation has unqualified local changes' >&2; exit 1;
}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$LIVEINFERENCE_ROOT/src" TORCH_DEVICE_BACKEND_AUTOLOAD=0
exec "$PYTHON_BIN" -m unittest discover -s "$HERE" -p test_state.py -v
