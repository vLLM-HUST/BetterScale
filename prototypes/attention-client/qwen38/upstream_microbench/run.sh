#!/usr/bin/env bash
# Source the host's private environment first. No global installs or upgrades.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEVICE=${1:?physical device}
KIND=${2:?indexer or attention}
OUTPUT=${3:?fresh output directory}
: "${QWEN38_PYTHON:?private Python interpreter}"
: "${QWEN38_OVERLAY:?qualified LiveInfer overlay}"
: "${QWEN38_UPSTREAM:?pinned fetch_sources.py output}"
: "${QWEN38_ADMISSION:?isolated admit_subset.py}"
export PYTHONPATH="$QWEN38_OVERLAY:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="$DEVICE" TASK_QUEUE_ENABLE=0 OMP_NUM_THREADS=2
exec "$QWEN38_PYTHON" "$QWEN38_ADMISSION" \
  --devices "$DEVICE" --output "$OUTPUT" --wait-seconds "${QWEN38_WAIT_SECONDS:-1200}" \
  -- "$QWEN38_PYTHON" "$HERE/probe.py" --device "$DEVICE" \
  --upstream "$QWEN38_UPSTREAM" --kind "$KIND" "${@:4}"
