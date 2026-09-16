#!/bin/bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
devices=${1:-0,1,2}
set +u; source /usr/local/Ascend/cann-9.0.1/set_env.sh; set -u
runtime=/workspace/my-ascend-workspace/runs/liveinfer-online/20260908-donor-local-runtime/env/bin/python
stamp=$(date -u +%Y%m%dT%H%M%SZ)
capsule="$repo/runs/device-service-$stamp"
mkdir -p "$capsule/source" "$capsule/build"
cp "$repo/prototypes/attention-client/device-service/"*.{py,cpp,sh} "$capsule/source/"
cp "$repo/prototypes/attention-client/device-service/admit_subset.py" "$capsule/source/launch.py"
cp "$repo/runs/attention-device-service-build/"{launch.so,queue_service.o} "$capsule/build/"
export ASCEND_RT_VISIBLE_DEVICES="$devices" OMP_NUM_THREADS=2 TASK_QUEUE_ENABLE=0
export PROBE_HELPERS=/workspace/my-ascend-workspace/runs/query-gang/20260911-lhtb-long-real-gang-v1/harness
export PYTHONPATH="$capsule/source${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
printf '%s\n' "$capsule"
exec "$runtime" "$capsule/source/launch.py" --devices "$devices" --output "$capsule/run" -- "$runtime" "$capsule/source/probe.py" --out "$capsule/run/measurements" --build "$capsule/build"
