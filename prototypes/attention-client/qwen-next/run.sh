#!/bin/bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
set +u; source /usr/local/Ascend/cann-9.0.1/set_env.sh; set -u
runtime=/workspace/my-ascend-workspace/runs/liveinfer-online/20260908-donor-local-runtime/env
devices=${1:-0,1,2,3,4,5}
# Reject stale geometry/config builds before allocating any device resources.
"$runtime/bin/python" - "$PERSISTENT_BUILD" "$DEVICE_SERVICE_SOURCE_BUILD" <<'PYABI'
import json, sys
from pathlib import Path
expected = dict(server_config_words=27, client_config_words=15, hidden=2048, inner=512, topk=10, owners=4)
for root in sys.argv[1:]:
    assert json.loads((Path(root)/"abi.json").read_text()) == expected, root
assert Path(sys.argv[1]).resolve() == Path(sys.argv[2]).resolve(), "Use one matched build closure"
PYABI
stamp=$(date -u +%Y%m%dT%H%M%SZ)
capsule="$repo/runs/qwen-next-$stamp"
mkdir -p "$capsule/source/next" "$capsule/source/roles" "$capsule/source/device" "$capsule/source/joint" "$capsule/build"
cp "$repo/prototypes/attention-client/qwen-next/"*.py "$capsule/source/next/"
cp "$repo/prototypes/attention-client/roles/"*.py "$capsule/source/roles/"
cp "$repo/prototypes/attention-client/device-service/"*.py "$capsule/source/device/"
cp "$repo/prototypes/attention-client/joint/"*.py "$capsule/source/joint/"
cp "$repo/prototypes/attention-client/"*.py "$capsule/source/"
cp "$DEVICE_SERVICE_SOURCE_BUILD/"{queue_service.o,launch.so} "$capsule/build/"
cp -r "$PERSISTENT_BUILD" "$capsule/persistent-build"
export PERSISTENT_BUILD="$capsule/persistent-build" DEVICE_SERVICE_BUILD="$capsule/build"
export PYTHONPATH="$capsule/source/next:$capsule/source/roles:$capsule/source/device:$capsule/source/joint:$capsule/source${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${NEXT_PROFILE:-0}" == "1" ]]; then
  export DEVICE_SERVICE_PROFILE="$capsule/profile"
fi
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 TASK_QUEUE_ENABLE=0
export HCCL_CONNECT_TIMEOUT=120 HCCL_EXEC_TIMEOUT=120 HCCL_BUFFSIZE=256 HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_HOST_IP=127.0.0.1 MASTER_ADDR=127.0.0.1 DEVICE_SERVICE_SEGMENTED=0
export ASCEND_CUSTOM_OPP_PATH="$runtime/lib/python3.12/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer"
export LD_LIBRARY_PATH="$ASCEND_CUSTOM_OPP_PATH/op_api/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True VLLM_ASCEND_ENABLE_FLASHCOMM1=0
export PROBE_HELPERS=/workspace/my-ascend-workspace/runs/query-gang/20260911-lhtb-long-real-gang-v1/harness
socket_dir="/tmp/qnext-$stamp"
printf '%s\n' "$socket_dir" > "$capsule/socket-directory.txt"
git -C "$repo" rev-parse HEAD > "$capsule/base-commit.txt"
printf '%s\n' "$capsule"
if [[ "${NEXT_NATIVE_ONLY:-0}" == "1" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="$devices" LOCAL_EXPERT_RESULT="$capsule/native.json"
  exec "$runtime/bin/python" "$capsule/source/device/admit_subset.py" --devices "$devices" --output "$capsule/run" -- timeout 180 "$runtime/bin/python" "$capsule/source/next/next_native_probe.py"
fi
if [[ "${NEXT_LEAF:-0}" == "1" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="$devices" LOCAL_EXPERT_RESULT="$capsule/leaf.json"
  leaf=next_leaf.py
  if [[ "${NEXT_LEAF_CONCURRENCY:-0}" == "1" ]]; then leaf=concurrency_leaf.py; fi
  exec "$runtime/bin/python" "$capsule/source/device/admit_subset.py" --devices "$devices" --output "$capsule/run" -- timeout 120 "$runtime/bin/python" "$capsule/source/next/$leaf"
fi
exec "$runtime/bin/python" "$capsule/source/device/admit_subset.py" --devices "$devices" --output "$capsule/run" -- "$runtime/bin/python" "$capsule/source/next/next_launch.py" --devices "$devices" --directory "$socket_dir" --artifacts "$capsule/roles"
