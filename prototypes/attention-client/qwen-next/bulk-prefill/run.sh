#!/bin/bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)
build=${BULK_BUILD:-"$repo/runs/bulk-prefill-1024-build"}
runtime=/workspace/my-ascend-workspace/runs/liveinfer-online/20260908-donor-local-runtime/env/bin/python
devices=${1:-1,3,4}
stamp=$(date -u +%Y%m%dT%H%M%SZ)
capsule="$repo/runs/bulk-prefill-$stamp"
mkdir -p "$capsule/source" "$capsule/build" "$capsule/admission"
# Admission snapshots its sibling Python files and prepends them to PYTHONPATH.
# Isolate it so it cannot shadow this experiment's matched runtime allocation ABI.
cp "$repo/prototypes/attention-client/device-service/admit_subset.py" "$capsule/admission/"
cp -r "$build/." "$capsule/build/"
cp "$repo/prototypes/attention-client/qwen-next/bulk-prefill/probe.py" "$capsule/source/"
cp "$repo/prototypes/attention-client/qwen-next/"{next_weights.py,settings.py} "$capsule/source/"
cp "$repo/prototypes/attention-client/joint/common.py" "$capsule/source/"
set +u; source /usr/local/Ascend/cann-9.0.1/set_env.sh; set -u
export PERSISTENT_BUILD="$capsule/build"
export PYTHONPATH="$capsule/build/source:$capsule/source${PYTHONPATH:+:$PYTHONPATH}"
export DEVICE_SERVICE_INTERNAL_PIPELINE=1 DEVICE_SERVICE_FINE_PACK=1 DEVICE_SERVICE_EARLY_DOWN=1 DEVICE_SERVICE_EARLY_RETURN=1 DEVICE_SERVICE_ROUTE_PULL=1
export DEVICE_SERVICE_RESIDENT_MOVES=0 DEVICE_SERVICE_MOVE_QUANTUM=0 DEVICE_SERVICE_SEGMENT_TAIL_EXPERTS=0
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 TASK_QUEUE_ENABLE=0
printf '%s\n' "$capsule"
exec "$runtime" "$capsule/admission/admit_subset.py" --devices "$devices" --wait-seconds 180 --output "$capsule/run" -- "$runtime" "$capsule/source/probe.py" --devices "$devices" --out "$capsule/results"
