#!/bin/bash
# One-card native expert control; requires the pinned lab runtime.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
devices=${1:-2,3}
set +u; source /usr/local/Ascend/cann-9.0.1/set_env.sh; set -u
runtime=/workspace/my-ascend-workspace/runs/liveinfer-online/20260908-donor-local-runtime/env
# Reuse the already-tested fail-closed subset lease/occupancy launcher.
admission="$repo/prototypes/attention-client/device-service/admit_subset.py"
test -f "$admission"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
capsule="$repo/runs/dfc-ep2-control-$stamp"
mkdir -p "$capsule/source/joint" "$capsule/source/device" "$capsule/build"
cp "$repo/prototypes/attention-client/device-service/"*.py "$capsule/source/device/"
cp "$repo/prototypes/attention-client/"*.py "$capsule/source/"
cp "$repo/prototypes/attention-client/joint/"*.py "$capsule/source/joint/"
cp "$admission" "$capsule/source/launch.py"
git -C "$repo" rev-parse HEAD > "$capsule/base-commit.txt"
export ASCEND_RT_VISIBLE_DEVICES="$devices" OMP_NUM_THREADS=2 TASK_QUEUE_ENABLE=0
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 HCCL_BUFFSIZE=256 HCCL_OP_EXPANSION_MODE=AIV
export VLLM_HOST_IP=127.0.0.1 MASTER_ADDR=127.0.0.1
export PYTHONPATH="$capsule/source/device:$capsule/source/joint:$capsule/source${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
export ASCEND_CUSTOM_OPP_PATH="$runtime/lib/python3.12/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export PROBE_HELPERS=/workspace/my-ascend-workspace/runs/query-gang/20260911-lhtb-long-real-gang-v1/harness
# The pinned donor A2 package omits BF16 DFC. Load the existing lab A2
# build ONLY in this process; never install it over the donor environment.
export ASCEND_CUSTOM_OPP_PATH=/workspace/my-ascend-workspace/stateharbor/build/lib.linux-aarch64-cpython-312/livemodule/arch/ascend/_native/opp/vendors/custom_transformer
export DFC_EXTENSION=/workspace/my-ascend-workspace/stateharbor/build/lib.linux-aarch64-cpython-312/livemodule/arch/ascend/_native/vllm_ascend_C.cpython-312-aarch64-linux-gnu.so
export LD_LIBRARY_PATH="$(dirname "$DFC_EXTENSION"):$ASCEND_CUSTOM_OPP_PATH/op_api/lib:$LD_LIBRARY_PATH"
printf '%s\n' "$ASCEND_CUSTOM_OPP_PATH" > "$capsule/dfc-provider.txt"
printf '%s\n' "$capsule"
exec "$runtime/bin/python" "$capsule/source/launch.py" --devices "$devices" --output "$capsule/run" -- "$runtime/bin/python" "$capsule/source/device/dfc_probe.py" --out "$capsule/results"
