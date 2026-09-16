#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
runtime=/workspace/my-ascend-workspace/runs/liveinfer-online/20260908-donor-local-runtime/env
: "${PROBE_CAPSULE:?fresh capsule required}"
mkdir -p "$(dirname "$PROBE_CAPSULE")"
mkdir "$PROBE_CAPSULE"
mkdir "$PROBE_CAPSULE/source" "$PROBE_CAPSULE/result"
cp "$repo/prototypes/owned-wave/attention_probe.py" "$PROBE_CAPSULE/source/"
cp "$repo/prototypes/attention-client/device-service/admit_subset.py" "$PROBE_CAPSULE/source/admit.py"
set +u; source /usr/local/Ascend/cann-9.0.1/set_env.sh; set -u
export ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 TASK_QUEUE_ENABLE=1
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export OWNED_OUTPUT="$PROBE_CAPSULE/result"
exec "$runtime/bin/python" "$PROBE_CAPSULE/source/admit.py" --devices 0 --wait-seconds 600 \
 --output "$PROBE_CAPSULE/run" -- "$runtime/bin/python" "$PROBE_CAPSULE/source/attention_probe.py"
