#!/usr/bin/env bash
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
CAP="$ROOT/runs/qwen38-wire-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$CAP/admission-helper" "$CAP/source"
cp "$ROOT/prototypes/attention-client/device-service/admit_subset.py" "$CAP/admission-helper/"
cp "$ROOT"/prototypes/attention-client/qwen38/*.py "$CAP/source/"
printf '%s\n' "$CAP" > /tmp/betterscale-qwen38-wire-capsule
source /usr/local/Ascend/cann-9.0.1/set_env.sh
export PYTHONPATH="$CAP/source:$ROOT/prototypes/attention-client/roles:$ROOT/runs/qwen38-native-runtime-20260917/overlay:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2 TASK_QUEUE_ENABLE=0 PYTHONDONTWRITEBYTECODE=1
PY=${QWEN38_PYTHON:-/workspace/my-ascend-workspace/runs/liveinfer-online/20260908-donor-local-runtime/env/bin/python}
DEVICES=${1:-1,3,4,5,6}
if (( $# )); then shift; fi
exec "$PY" "$CAP/admission-helper/admit_subset.py" --devices "$DEVICES" --output "$CAP/run" --wait-seconds "${QWEN38_WAIT_SECONDS:-1200}" -- "$PY" "$CAP/source/launch_wire.py" --devices "$DEVICES" --directory "/tmp/q38-wire-$$" --artifacts "$CAP/roles" --build "${QWEN38_BUILD:-$ROOT/runs/qwen38-server-build-20260917T0348-lifetime}" "$@"
